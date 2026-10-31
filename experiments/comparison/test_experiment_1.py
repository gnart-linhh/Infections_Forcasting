"""Science checks for limited-history fitting, common masks and bank-loss aggregation."""
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import experiments.comparison.experiment_1 as e1
from generator import SyntheticCurve
from model import forecast


class ComparisonTests(unittest.TestCase):
    def test_baseline_definitions_and_common_floor(self):
        z=np.arange(1.,7.)
        np.testing.assert_array_equal(e1.baseline('Persistence',z)[0],[6]*4)
        np.testing.assert_array_equal(e1.baseline('MA3',z)[0],[5]*4)
        np.testing.assert_array_equal(e1.baseline('Drift',z)[0],[7,8,9,10])
        point,info=e1.baseline('Drift',[6,5,4,3,2,1])
        np.testing.assert_array_equal(point,[1]*4);self.assertEqual(info['clipped_predictions'],4)
        np.testing.assert_array_equal(e1.baseline('Persistence',[0]*6)[0],[1]*4)

    def test_SES_constant_history_and_endpoint(self):
        p,info=e1.ses([7]*6)
        np.testing.assert_array_equal(p,[7]*4);self.assertEqual(info['alpha'],0.)
        p,info=e1.ses(np.arange(1.,7.))
        self.assertEqual(info['alpha'],1.)
        np.testing.assert_array_equal(p,[6]*4)

    def test_SES_fits_only_conditional_history_loss(self):
        z=np.array([9.,5.,12.,2.,8.,7.,10.,4.])
        p,info=e1.ses(z);alpha=info['alpha'];level=z[0];loss=0.
        for observed in z[1:]:
            loss+=(observed-level)**2;level=alpha*observed+(1-alpha)*level
        self.assertAlmostEqual(p[0],level,places=10)
        self.assertAlmostEqual(info['conditional_SSE'],loss,places=8)
        losses=[]
        for a in np.linspace(0,1,101):
            l=z[0];v=0.
            for y in z[1:]:v+=(y-l)**2;l=a*y+(1-a)*l
            losses.append(v)
        self.assertLessEqual(loss,min(losses)+1e-8)

    def test_synthetic_onset_weekly_count_gate_and_followup(self):
        values=np.r_[np.zeros(3)+1e-10,10.,np.arange(11.,60.)]
        curve=SyntheticCurve(values,'seasonal','weekly','counts',True)
        self.assertEqual(e1.synthetic_eligibility(curve),('eligible',3))
        self.assertEqual(e1.synthetic_eligibility(SyntheticCurve(values,'seasonal','monthly','counts',True))[0],'monthly')
        self.assertEqual(e1.synthetic_eligibility(SyntheticCurve(values,'seasonal','weekly','proportion',True))[0],'proportion')
        self.assertEqual(e1.synthetic_eligibility(SyntheticCurve(values[:20],'seasonal','weekly','counts',True))[0],'insufficient_followup')
        self.assertGreater(e1.RNG_ROOT,2000000+3*e1.N)

    def test_case_has_exact_history_and_core_GAM_query(self):
        raw=np.array([0.,3.,8.,12.,9.,14.])
        c=e1.make_case(0,'synthetic','B','x','seasonal',0,'24',23,6,'1',24,'19',raw)
        np.testing.assert_array_equal(e1.unpack(c[13]),raw)
        self.assertEqual(c[14],e1.digest_array(np.maximum(raw,1)))
        rng=np.random.default_rng(4);x=rng.normal(size=(60,5));y=rng.normal(size=(60,4))
        point,ids,distance=forecast(raw,x,y,closest=8,smoothing='gam')
        selected,d=e1.ExactNeighbors(x).query(e1.unpack(c[22]),8)
        np.testing.assert_array_equal(ids,selected[0]);np.testing.assert_allclose(distance,d[0])
        expected=np.maximum(1,max(raw[-1],1)+np.cumsum(np.median(y[selected[0]],axis=0)))
        np.testing.assert_array_equal(point,expected)
        with self.assertRaises(ValueError):e1.make_case(0,'real','A','x','',-1,'2020-01-04',0,8,'',0,'',raw)

    def test_common_mask_removes_entire_unit_across_T_and_banks(self):
        cases=[];fits=[];ident=0
        expected=[(m,-1) for m in e1.BASELINES]+[('sMOA',s) for s in e1.SEEDS]
        for state in ['x','y']:
            for T in e1.LENGTHS:
                cases.append(dict(case_id=ident,domain='real',arm='B',series=state,origin='2021-01-02',T=T))
                fits.extend(dict(model=m,bank_seed=s,case_id=ident,status='fit_failed' if state=='x' and T==16 and s==2 else 'ok') for m,s in expected)
                ident+=1
        keep=e1.common_cases(pd.DataFrame(cases),pd.DataFrame(fits),expected)
        self.assertEqual(keep,set(range(5,10)))

    def test_MASE_zero_is_undefined_without_dropping_MAE(self):
        g=pd.DataFrame(dict(prediction=[2.,10.],actual=[0.,5.],mase_den=[0.,2.],normalization=[1.,2.],case_id=[0,1]))
        s=e1.score_metrics(g)
        self.assertEqual(s['MAE'],3.5);self.assertEqual(s['MASE'],2.5)
        self.assertEqual(s['MASE_undefined'],1);self.assertEqual(s['MASE_defined'],1)

    def test_pairing_averages_bank_losses_not_predictions(self):
        rows=[]
        for case,state in enumerate(['x','y']):
            for seed,point in enumerate([1.,5.,9.]):
                rows.append(dict(model='sMOA',bank_seed=seed,case_id=case,horizon=1,prediction=point,actual=5.,
                    domain='real',arm='A',T=6,cluster=state,phase='stable',case_level='low'))
            rows.append(dict(model='Persistence',bank_seed=-1,case_id=case,horizon=1,prediction=7.,actual=5.,
                domain='real',arm='A',T=6,cluster=state,phase='stable',case_level='low'))
        p=e1.paired_tables(pd.DataFrame(rows));row=p.loc[p.dimension.eq('overall')].iloc[0]
        self.assertAlmostEqual(row.delta_MAE,2/3)
        self.assertAlmostEqual(row.lo,2/3);self.assertAlmostEqual(row.hi,2/3)

    def test_RMSE_seed_summary_does_not_pool_squared_errors(self):
        rows=[]
        for seed,end in enumerate([2.,4.,6.]):
            for case,point in enumerate([0.,end]):
                rows.append(dict(model='sMOA',bank_seed=seed,case_id=case,horizon=1,prediction=point,actual=0.,
                    domain='real',arm='A',T=6,phase='stable',case_level='low',normalization=1.,mase_den=1.))
        s=e1.summarize_scores(pd.DataFrame(rows));row=s.loc[s.statistic.eq('mean')&s.dimension.eq('overall')].iloc[0]
        self.assertAlmostEqual(row.RMSE,np.mean([np.sqrt(2),np.sqrt(8),np.sqrt(18)]))
        self.assertNotAlmostEqual(row.RMSE,np.sqrt(56/6))
        self.assertEqual(row.seeds,3)

    def test_dependency_missing_never_emits_fake_ARIMA_forecast(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);db=e1.connect(out);e1.initialize(db)
            case=e1.make_case(0,'real','B','x','',-1,'2020-07-04',0,16,'2020-03-07',18,'2020-03-21',np.arange(1.,17.))
            with db:
                db.execute('INSERT INTO cases VALUES ('+','.join(['?']*26)+')',case)
                db.executemany('INSERT INTO observations VALUES (?,?,?,?)',[(0,h,str(h),1.) for h in range(1,5)])
            with patch.object(e1.importlib.util,'find_spec',return_value=None):e1.run_baselines(db,out)
            self.assertEqual(db.execute("SELECT status FROM fits WHERE model='ARIMA'").fetchone()[0],'dependency_missing')
            self.assertEqual(db.execute("SELECT COUNT(*) FROM predictions WHERE model IN ('ARIMA','Holt')").fetchone()[0],0)
            with self.assertRaises(sqlite3.IntegrityError):
                with db:e1.insert_fit(db,'Persistence',-1,0,'ok',1.,{},[1]*4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM predictions WHERE model='Persistence'").fetchone()[0],4)
            db.close()

    def test_cluster_bootstrap_does_not_gain_precision_by_copying_rows(self):
        frame=pd.DataFrame(dict(cluster=['a','a','b','b'],smoa_loss=[1.,2.,3.,4.],baseline_loss=[2.,2.,2.,2.]))
        a=e1.paired_bootstrap(frame);b=e1.paired_bootstrap(pd.concat([frame]*5,ignore_index=True))
        for field in ['delta_MAE','lo','hi','relative_lo','relative_hi']:self.assertAlmostEqual(a[field],b[field])


if __name__=='__main__':unittest.main()
