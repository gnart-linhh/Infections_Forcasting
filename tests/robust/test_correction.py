"""Loss alignment, bounded inference, isolation, caches and explicit comparisons."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.robust.policy import (FEATURES,CORRECTION_MODELS,MedianCorrection,RankPolicy,
    correction_scale,normalized_residual,bounded_correction,candidate_features)
from experiments.robust.correction_experiment import (choose_correction,issue,cache_records,WeightedForecast)
from experiments.robust.pipeline import load_config
from experiments.robust.policy_experiment import prepare_records
from experiments.robust.explain import DirectForecast,explain_experiment
from experiments.robust.evaluation import paired_metrics,cluster_comparisons
from experiments.robust.data import ROOT,ForecastCase
from experiments.robust.method import RobustSMOA


def fixture():
    rng=np.random.default_rng(91);n=120
    frame=pd.DataFrame(rng.normal(size=(n,6)),columns=FEATURES)
    frame['horizon']=np.tile(np.arange(1,5),30);frame['history_length']=6
    frame['dataset']='synthetic_training';frame['family']=np.where(np.arange(n)<60,'a','b')
    frame['region']=[f'curve_{i//4}' for i in range(n)];frame['scale']=10.
    for j,v in enumerate([2.,10.,30.]):frame[f'candidate_{j}']=v;frame[f'rank_{j}']=v;frame[f'identity_{j}']=j
    frame['prediction']=10.;frame['weighted']=11.;frame['actual']=10.+2*(frame.relative_growth>0)
    frame['history_sha256']='a'*64
    for name in ['q025','q100','q250','q500','q750','q900','q975']:frame[name]=np.nan
    return frame


def engine():
    rng=np.random.default_rng(12)
    return RobustSMOA(rng.normal(size=(50,5))*4,rng.normal(size=(50,4))*5,closest=16)


class CorrectionTests(unittest.TestCase):
    def test_normalized_residual_and_extreme_bounds(self):
        frame=fixture().iloc[:3].copy()
        frame['scale']=[100.,1.,1.];frame['actual']=[110.,-18.,290.]
        np.testing.assert_array_equal(correction_scale(frame),[100.,28.,28.])
        np.testing.assert_array_equal(normalized_residual(frame),[1.,-1.,10.])
        # The training target remains untruncated; only inference is bounded.
        np.testing.assert_array_equal(bounded_correction(frame,[1e20,-1e20,1e20]),[30.,2.,30.])
        np.testing.assert_array_equal(bounded_correction(frame,[1e20,-1e20,1e20],0),[10.,10.,10.])
        with self.assertRaises(ValueError):bounded_correction(frame,[np.nan]*3)
        with self.assertRaises(ValueError):bounded_correction(frame,[0]*3,1.1)

    def test_deterministic_training_and_no_real_fitting(self):
        with threadpool_limits(1):
            frame=fixture();a=MedianCorrection(42).fit(frame);b=MedianCorrection(42).fit(frame)
            np.testing.assert_array_equal(a.predict(frame),b.predict(frame))
            self.assertGreater(np.ptp(a.predict(frame)),0)
            with self.assertRaises(ValueError):MedianCorrection().fit(frame.assign(dataset='COVID19_early'))
            with self.assertRaises(ValueError):choose_correction(frame,frame.assign(dataset='COVID19_early'),{}, {})

    def test_rejected_candidate_names_the_original_and_reasons(self):
        cfg=load_config(ROOT/'configs/experiments.json');cfg['median_correction']['small_per_seed_family']=1
        training=fixture();validation=training.assign(dataset='synthetic_validation',actual=training.candidate_1)
        records=[dict(curve_id=c,family='a',seed_key=[42,1,0,0]) for c in training.region.unique()]
        with threadpool_limits(1):
            fit,frozen,table,_=choose_correction(training,validation,{'training_records':records},cfg)
        self.assertEqual(frozen['selected_method'],'original_smoa');self.assertFalse(frozen['candidate_deployed'])
        self.assertTrue(table.rejection_reasons.str.contains('raw_mae').all())
        np.testing.assert_array_equal(fit.predict(validation,frozen['rho']),validation.candidate_1)

    def test_eligible_selection_uses_raw_mae_and_stays_synthetic(self):
        cfg=load_config(ROOT/'configs/experiments.json');cfg['median_correction']['small_per_seed_family']=1
        training=fixture().assign(actual=12.)
        records=[dict(curve_id=c,family='a',seed_key=[42,1,0,0]) for c in training.region.unique()]
        validation=training.assign(dataset='synthetic_validation')
        with threadpool_limits(1):
            fit,frozen,table,_=choose_correction(training,validation,{'training_records':records},cfg)
        self.assertEqual(frozen['selected_method'],'bounded_median_blend')
        self.assertTrue(frozen['candidate_deployed']);self.assertEqual(frozen['rho'],1.)
        self.assertLess(np.mean(abs(fit.predict(validation,1.)-12.)),.01)

    def test_point_training_preparation_does_not_simulate_uncertainty(self):
        case=ForecastCase('x','synthetic_training','curve',np.arange(6.)+10,6,6,'early')
        with patch('experiments.forecasting.uncertainty.interval_quantiles',side_effect=AssertionError('NB invoked')):
            frame,_=prepare_records(engine(),[case],load_config(ROOT/'configs/experiments.json'),point_only=True)
        self.assertEqual(len(frame),4);self.assertTrue(frame.q500.isna().all());self.assertNotIn('actual',frame)

    def test_issued_registry_pairing_and_no_label_dependence(self):
        frame=fixture();cfg=load_config(ROOT/'configs/experiments.json')
        with threadpool_limits(1):
            median=MedianCorrection().fit(frame);rank=RankPolicy().fit(frame)
            a=issue(frame,rank,median,0,cfg);b=issue(frame.assign(actual=1e12),rank,median,0,cfg)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        self.assertEqual(set(a.model),set(CORRECTION_MODELS));self.assertNotIn('actual',a)
        np.testing.assert_array_equal(a.query('model=="selected_forecast"').prediction,frame.candidate_1)
        # Explicit registry works with descriptive model names on a common sample.
        a['case_id']=np.tile(np.arange(len(frame)),len(CORRECTION_MODELS));a['actual']=10.;a['forecast_origin']='x'
        metrics=paired_metrics(a,models=CORRECTION_MODELS,baseline_model='original_smoa')
        self.assertEqual(len(metrics),5);self.assertTrue(metrics.n.eq(len(frame)).all())
        with self.assertRaises(ValueError):paired_metrics(pd.concat([a,a.iloc[:1]]),models=CORRECTION_MODELS,baseline_model='original_smoa')

    def test_direct_wrappers_match_reported_points_and_rematch(self):
        eng=engine();history=np.arange(6.)+10;frame=fixture()
        with threadpool_limits(1):
            median=MedianCorrection().fit(frame)
            for policy in [WeightedForecast(),median]:
                wrapper=DirectForecast(eng,dict(temperature=4.,beta=0.),4,policy=policy)
                observed=wrapper(history)[0];expected=policy.forecast(eng,history)[0][3]
                np.testing.assert_array_equal(observed,[expected,expected-eng.match(history).baseline[3]])
            before=eng.match_calls;wrapper(history+1);self.assertEqual(eng.match_calls,before+1)

    def test_no_nontrivial_contrast_is_not_plotted(self):
        case=ForecastCase('x','COVID19_early','a',np.arange(6.)+10,6,6,'early')
        rows=[dict(case_id='x',horizon=4,model=m,prediction=10.,actual=12.,phase='rising',growth_rate=1.,dispersion=1.,history_length=6) for m in ['original_smoa','median_correction']]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);explain_experiment(None,[case],pd.DataFrame(rows),{}, {},dict(xai=dict(horizon=4,max_cases=5)),out,
                baseline_model='original_smoa',forecast_model='median_correction')
            self.assertEqual(json.loads((out/'explanation_metadata.json').read_text())['status'],'skipped_zero_adjustment')
            self.assertEqual(list(out.glob('*.png')),[])

    def test_cache_reuse_and_source_invalidation(self):
        cfg=load_config(ROOT/'configs/experiments.json');case=ForecastCase('x','synthetic_training','curve',np.arange(6.)+10,6,6,'early')
        frame,_=prepare_records(engine(),[case],cfg,point_only=True)
        truth=frame[['case_id','horizon','target_date']].assign(actual=12.)
        synthetic={'training_records':[dict(curve_id='curve',family='a',seed_key=[42,1,0,0])]}
        from experiments.runtime import file_hash as real_hash
        with tempfile.TemporaryDirectory() as tmp:
            eng=engine();out=Path(tmp)
            a,reused=cache_records(eng,[case],truth,synthetic,'training',out,cfg,{})
            self.assertFalse(reused);calls=eng.match_calls
            b,reused=cache_records(eng,[case],truth,synthetic,'training',out,cfg,{})
            self.assertTrue(reused);self.assertEqual(eng.match_calls,calls);pd.testing.assert_frame_equal(a,b)
            with patch('experiments.robust.correction_experiment.file_hash',side_effect=lambda p:'changed' if Path(p).name=='policy.py' else real_hash(p)):
                _,reused=cache_records(eng,[case],truth,synthetic,'training',out,cfg,{})
            self.assertFalse(reused);self.assertEqual(len(list(out.glob('*.pkl'))),2)

    def test_cluster_bootstrap_aggregates_counts(self):
        # Clusters have unequal sizes: pooled delta is (1+1+10)/3, not 5.5.
        rows=[]
        for i,(region,delta) in enumerate([('a',1.),('a',1.),('b',10.)]):
            for m,point in [('original_smoa',10.),('median_correction',10.+delta)]:
                rows.append(dict(case_id=str(i),horizon=1,actual=10.,region=region,model=m,prediction=point,dataset='fixture'))
        result=cluster_comparisons(pd.DataFrame(rows),draws=100,baseline_model='original_smoa')
        row=result.set_index('model').loc['median_correction']
        self.assertEqual(row.mae_delta_vs_original,4.);self.assertEqual(row.lower,1.);self.assertEqual(row.upper,10.)

if __name__=='__main__':unittest.main()
