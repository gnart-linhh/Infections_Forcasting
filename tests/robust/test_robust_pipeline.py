"""Future isolation, curve provenance, schema pairing, and direct SHAP checks."""
import copy
from contextlib import closing
import inspect
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.robust.data import (ForecastCase, array_hash, load_real_cases, load_real_outcomes,
                                     attach_outcomes, ROOT)
from experiments.robust.method import RobustSMOA
from experiments.robust.pipeline import load_config, forecast_cases
from experiments.robust.tuning import verify_disjoint, select_parameters, NAMESPACE, ROLES
from experiments.robust.explain import DirectForecast, background_histories
from experiments.robust.evaluation import paired_metrics
from generator import SyntheticCurve


def fixture_engine():
    rng = np.random.default_rng(12)
    return RobustSMOA(rng.normal(size=(50, 5))*4, rng.normal(size=(50, 4))*5, closest=16)


def database(path, outcome):
    history = np.array([5., 8., 9., 12., 16., 19.])
    with closing(sqlite3.connect(path)) as c:
        c.execute('CREATE TABLE cases(case_id,series,origin,T,onset,window_start,history,history_sha256,data_cutoff,issuance_date,domain,arm)')
        c.execute('INSERT INTO cases VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                  (1,'fixture','2020-02-08',6,'2020-01-04','2020-01-04',history.astype('<f8').tobytes(),
                   array_hash(history),'2020-02-09','2020-02-10','real','A'))
        c.execute('CREATE TABLE observations(case_id,horizon,target,actual)')
        c.executemany('INSERT INTO observations VALUES(?,?,?,?)',
                       [(1,h,(pd.Timestamp('2020-02-08')+pd.Timedelta(weeks=h)).date().isoformat(),outcome+h) for h in range(1,5)])
        c.commit()


class PipelineTests(unittest.TestCase):
    def test_identical_observed_histories_different_future_datasets_identical_forecasts(self):
        cfg = load_config(ROOT/'configs/experiments.json')
        engine = fixture_engine()
        frozen = dict(temperature=.5, beta=.3)
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, outcome in enumerate([10., 1e8]):
                path = Path(tmp)/f'{i}.sqlite';database(path, outcome)
                config = copy.deepcopy(cfg);config['evaluation']['database'] = str(path)
                cases = load_real_cases(config)
                issued, _ = forecast_cases(engine, cases, frozen, draws=100)
                self.assertNotIn('actual', issued)
                observations = load_real_outcomes(config, cases)
                scored = attach_outcomes(issued, observations)
                results.append((issued, scored))
        pd.testing.assert_frame_equal(results[0][0], results[1][0], check_exact=True)
        self.assertFalse(results[0][1].actual.equals(results[1][1].actual))

    def test_no_outcome_parameter_or_global_training_interface(self):
        self.assertNotIn('actual', inspect.signature(ForecastCase).parameters)
        self.assertEqual(list(inspect.signature(forecast_cases).parameters), ['engine','cases','frozen','seed','draws'])
        with self.assertRaises(TypeError):
            ForecastCase('x','y','z',np.arange(6.),1,1,'early', actual=10)

    def test_synthetic_selector_rejects_target_disease_cases(self):
        cases = [ForecastCase('1','COVID19_early','A',np.arange(6.)+10,1,1,'early')]
        with self.assertRaisesRegex(ValueError, 'independent synthetic'):
            select_parameters(fixture_engine(), cases, pd.DataFrame(), load_config(ROOT/'configs/experiments.json'), {})

    def test_whole_curve_disjointness_and_modified_provenance(self):
        metadata = {}
        for i, role in enumerate(ROLES):
            values = np.arange(20.)+i*100
            metadata[role] = [SyntheticCurve(values,'seasonal','weekly','counts',True)]
            metadata[role+'_records'] = [dict(sha256=array_hash(values),seed_key=[42,NAMESPACE,ROLES[role],0,0,0],onset=0)]
        verify_disjoint(metadata, {array_hash(np.arange(20.)+1000)})
        with self.assertRaises(ValueError):verify_disjoint(metadata, {array_hash(np.arange(20.))})
        changed = copy.deepcopy(metadata)
        changed['background'] = changed['validation']
        with self.assertRaises(ValueError):verify_disjoint(changed, set())

    def test_direct_shap_wrapper_rematches_and_explains_final_count_and_contrast(self):
        engine = fixture_engine();frozen = dict(temperature=.5,beta=.3)
        wrapper = DirectForecast(engine, frozen, 4)
        a = np.arange(6.)+10
        expected, _ = engine.variants(a,.5,.3)
        result = wrapper(a)[0]
        np.testing.assert_array_equal(result,[expected['M3'][3],expected['M3'][3]-expected['M0'][3]])
        before = engine.match_calls
        wrapper(np.vstack([a,a+2]))
        self.assertEqual(engine.match_calls,before+1)
        import shap
        background = np.vstack([a*.8,a*1.2])
        explain = shap.KernelExplainer(wrapper,background)
        values = np.asarray(explain.shap_values(a[None,:],nsamples=64,l1_reg=0,silent=True))
        np.testing.assert_allclose(np.asarray(explain.expected_value)+values[0].sum(axis=0),wrapper(a)[0],atol=1e-7)
        self.assertGreater(engine.match_calls,before+2)

    def test_background_uses_independent_count_sequences_and_observed_scale(self):
        curves = [SyntheticCurve(np.arange(20.)+i+1,'seasonal','weekly','counts',True) for i in range(4)]
        synthetic = dict(background=curves,background_records=[dict(onset=0) for _ in curves])
        a = background_histories(synthetic,6,np.arange(6.)+10,3,42)
        np.testing.assert_allclose(a.mean(axis=1),12.5)
        self.assertTrue((a>=0).all());self.assertEqual(a.shape,(3,6))
        np.testing.assert_array_equal(a,background_histories(synthetic,6,np.arange(6.)+10,3,42))

    def test_pairing_duplicates_missing_models_and_wrong_targets(self):
        case = ForecastCase('x','fixture','A',np.arange(6.)+10,6,6,'early')
        issued, _ = forecast_cases(fixture_engine(),[case],dict(temperature=1,beta=0),draws=50)
        outcomes = pd.DataFrame([dict(case_id='x',horizon=h,target_date=f'step_{6+h}',actual=10+h) for h in range(1,5)])
        scored = attach_outcomes(issued,outcomes)
        self.assertTrue(paired_metrics(scored).n.eq(4).all())
        incomplete = scored.loc[~(scored.model.eq('M1') & scored.horizon.eq(1))]
        self.assertTrue(paired_metrics(incomplete).n.eq(3).all())
        with self.assertRaises(ValueError):paired_metrics(pd.concat([scored,scored]))
        with self.assertRaises(ValueError):attach_outcomes(issued,outcomes.assign(target_date='wrong'))

    def test_reproducibility_and_invalid_history_lengths(self):
        case = ForecastCase('x','fixture','A',np.arange(6.)+10,6,6,'early')
        a = forecast_cases(fixture_engine(),[case],dict(temperature=1.,beta=.1),seed=42,draws=50)[0]
        b = forecast_cases(fixture_engine(),[case],dict(temperature=1.,beta=.1),seed=42,draws=50)[0]
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        for n in [5,7,12]:
            with self.assertRaises(ValueError):ForecastCase('x','fixture','A',np.arange(n),6,6,'early')


if __name__ == '__main__':unittest.main()
