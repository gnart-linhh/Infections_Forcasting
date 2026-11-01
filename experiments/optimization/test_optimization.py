import csv
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from bayes import GaussianProcess,decode,expected_improvement,minimize_objective,NAMES
from objective import embeddings,evaluate,generate_bank,origin_indices
from generator import SyntheticCurve


def curve(values):
    return SyntheticCurve(np.asarray(values,float),'seasonal','weekly','counts',True)


class BayesianTests(unittest.TestCase):
    def test_expected_improvement_is_for_minimization(self):
        self.assertAlmostEqual(float(expected_improvement(0,1,0)),1/np.sqrt(2*np.pi))
        self.assertGreater(expected_improvement(-1,1,0),expected_improvement(1,1,0))
        np.testing.assert_allclose(expected_improvement([-2,2],[0,0],0),[2,0])

    def test_integer_ceiling_and_bounds(self):
        bounds=[[5,20],[2,7],[10,100]]
        np.testing.assert_array_equal(decode([0,.1,1],bounds),[5,3,100])

    def test_gp_posterior_has_finite_uncertainty(self):
        model=GaussianProcess(np.array([[0.],[.5],[1.]]),np.array([1.,0.,1.]))
        mean,std=model.predict([[0.],[.5],[1.],[.25]])
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertTrue(np.all(std>=0))
        np.testing.assert_allclose(mean[:3],[1,0,1],atol=.05)

    def test_bo_returns_best_evaluated_point_and_reproduces(self):
        def objective(params,trial):
            values=np.array([params[n] for n in NAMES])
            return float(np.sum((values-[3,4,2])**2)),{}
        bounds=[[1,5],[1,6],[1,4]]
        best,trace=minimize_objective(objective,bounds,4,2,13)
        best_again,trace_again=minimize_objective(objective,bounds,4,2,13)
        self.assertEqual(best,best_again);self.assertEqual(trace,trace_again)
        self.assertEqual(len(trace),6)
        self.assertEqual(sum(r['phase']=='ei' for r in trace),2)
        self.assertEqual(len({tuple(r[n] for n in NAMES) for r in trace}),6)
        selected=min(trace,key=lambda r:r['validation_mae'])
        self.assertEqual(best,{n:selected[n] for n in NAMES})
        self.assertLessEqual(selected['validation_mae'],min(r['validation_mae'] for r in trace[:4]))

    def test_invalid_space_and_budget_rejected(self):
        with self.assertRaises(ValueError):
            minimize_objective(lambda p,i:(0,{}),[[1,2],[1,2],[1,2]],9,0)

    def test_resume_keeps_lhs_ei_candidates_and_rng_identical(self):
        def objective(params,trial):
            return float(sum((params[n]-2)**2 for n in NAMES)),{}
        bounds=[[1,5],[1,6],[1,4]]
        expected,trace=minimize_objective(objective,bounds,4,2,13)
        for stop in [2,5]:
            saved={}
            def checkpoint(state):
                saved.update(copy.deepcopy(state))
                if len(state['history'])==stop: raise InterruptedError('simulated interruption')
            with self.assertRaises(InterruptedError):
                minimize_objective(objective,bounds,4,2,13,on_checkpoint=checkpoint)
            called=[]
            def resumed_objective(params,trial):
                called.append(trial)
                return objective(params,trial)
            actual,resumed=minimize_objective(resumed_objective,bounds,4,2,13,resume_state=saved)
            self.assertEqual(actual,expected);self.assertEqual(resumed,trace)
            self.assertEqual(called,list(range(stop,6)))


class ObjectiveTests(unittest.TestCase):
    def test_parallel_gam_preprocessing_preserves_all_records_and_score(self):
        curves=[curve([2,4,3,9,14,5,7,12,19,11]),curve([3,8,6,4,16,12,20,13,8,21])]
        x=np.array([[1.,2.],[2.,1.],[-2.,1.],[0.,3.]])
        y=np.array([[1.,3.],[-1.,4.],[3.,-2.],[0.,2.]])
        serial=evaluate(curves,x,y,2,2,smoothing='gam',workers=1)
        parallel=evaluate(curves,x,y,2,2,smoothing='gam',workers=2)
        self.assertEqual(serial,parallel)

    def test_constant_history_filter_matches_both_forecast_branches(self):
        values=[curve([0]*20),curve(np.arange(20,dtype=float))]
        for mode,expected_rows in [('upstream',10),('pipeline',11)]:
            x,y=embeddings(values,5,4,mode)
            self.assertEqual(len(x),expected_rows)
            self.assertFalse(np.all(x==0,axis=1).any())

    def test_upstream_diff_y_omits_bridge(self):
        values=[curve([1,3,6,10,15,21])]
        x,y=embeddings(values,2,2,'upstream')
        np.testing.assert_array_equal(x,[[2,3]])
        np.testing.assert_array_equal(y,[[5,6]])
        xp,yp=embeddings(values,2,2,'pipeline')
        np.testing.assert_array_equal(xp,[[2,3],[3,4]])
        np.testing.assert_array_equal(yp,[[4,5],[5,6]])

    def test_origin_range_matches_r_and_sampling_is_unique(self):
        np.testing.assert_array_equal(origin_indices(10,2,4,0),[3,4,5])
        np.testing.assert_array_equal(origin_indices(10,2,4,2),[3,5])

    def test_mae_reconstruction_plus_100_and_future_alignment(self):
        score,rows=evaluate([curve([1,3,6,10,15,21])],np.array([[2,3]]),np.array([[4,5]]),2,1,0,'none')
        self.assertEqual(score,0)
        self.assertEqual([r['actual'] for r in rows],[110,115])
        self.assertEqual([r['forecast'] for r in rows],[110,115])
        self.assertEqual(np.mean([r['baseline_abs_error'] for r in rows]),6.5)
        with self.assertRaises(ValueError):
            evaluate([curve([1,3,6,10,15,21])],np.array([[2,3]]),np.array([[4,5]]),2,2)

    def test_seed_namespaces_are_disjoint_and_reproducible(self):
        def fake_generator(family,rng):
            return curve(rng.uniform(1,2,10))
        with patch('objective.gen_curve',fake_generator):
            first,audit=generate_bank(1,13,0,0)
            repeat,_=generate_bank(1,13,0,0)
            _,validation=generate_bank(1,13,1,0)
            _,heldout=generate_bank(1,13,2,0)
        np.testing.assert_array_equal(first[0].ts,repeat[0].ts)
        keys=[tuple(r['seed_key']) for r in audit+validation+heldout]
        self.assertEqual(len(keys),len(set(keys)))

    def test_cli_logs_ei_exports_best_library_and_independent_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            subprocess.run([sys.executable,str(Path(__file__).with_name('run_optimization.py')),
                '--num-curves-range','1','2','--k-range','2','3','--closest-range','1','2',
                '--initial-points','2','--iterations','1','--max-origins-per-curve','2',
                '--test-curves-per-family','1','--smoothing','none','--output-dir',str(out)],
                check=True,capture_output=True,text=True)
            trace=json.loads((out/'trials.json').read_text())
            selected=json.loads((out/'best_params.json').read_text())
            result=json.loads((out/'summary.json').read_text())
            self.assertEqual(selected,result['best_params'])
            self.assertEqual(result['ei_trials'],1)
            self.assertEqual(result['best_validation_mae'],min(r['validation_mae'] for r in trace))
            from bank import read_audit
            summary_audit=json.loads((out/'generation_audit.json').read_text())
            audit=[entry for item in summary_audit for entry in read_audit(out/item['audit'])]
            self.assertEqual(len(audit),len({tuple(r['seed_key']) for r in audit}))
            self.assertTrue(any(r['role']==2 for r in audit))
            with (out/'heldout_forecasts.csv').open() as handle:
                records=list(csv.DictReader(handle))
            self.assertAlmostEqual(result['heldout_test']['mae'],np.mean([float(r['abs_error']) for r in records]))
            self.assertTrue((out/'best_library/curves.csv').is_file())
            self.assertEqual(json.loads((out/'metadata.json').read_text())['status'],'complete')
            self.assertFalse(json.loads((out/'config.json').read_text())['full_search_budget'])


class FixedValidationTests(unittest.TestCase):
    def test_fixed_validation_reuses_seed_bank_and_target_grid(self):
        from objective import SyntheticObjective
        def fake_generator(family,rng):
            return curve(100+np.cumsum(rng.uniform(1,2,30)))
        with patch('objective.gen_curve',fake_generator):
            objective=SyntheticObjective(seed=13,max_origins=3,smoothing='none',mode='upstream',
                validation_protocol='fixed',validation_curves_per_family=2,min_history=8)
            one,first=objective(dict(num_curves=2,k=2,closest=1),0)
            two,second=objective(dict(num_curves=2,k=2,closest=1),1)
        self.assertEqual(one,two)
        self.assertEqual(first,second)
        val=[r for r in objective.audit if r['role']==1]
        self.assertEqual({tuple(r['seed_key']) for r in val if r['evaluation_trial']==0},
                         {tuple(r['seed_key']) for r in val if r['evaluation_trial']==1})
        np.testing.assert_array_equal(origin_indices(30,2,4,3,min_history=8),
                                      origin_indices(30,7,4,3,min_history=8))

    def test_best_bank_stays_with_lowest_validation_score(self):
        from objective import SyntheticObjective
        banks=[[curve([i+1,i+2,i+3,i+4,i+5,i+6])]*3 for i in range(4)]
        audits=[[dict(accepted=True,seed_key=[i])] *3 for i in range(4)]
        with patch('objective.generate_bank',side_effect=list(zip(banks,audits))), \
             patch('objective.embeddings',return_value=(np.ones((1,2)),np.ones((1,4)))), \
             patch('objective.evaluate',side_effect=[(1.,[dict(baseline_abs_error=2)]),(5.,[dict(baseline_abs_error=2)])]):
            objective=SyntheticObjective()
            objective(dict(num_curves=1,k=2,closest=1),0)
            objective(dict(num_curves=1,k=2,closest=1),1)
        self.assertEqual(objective.best_score,1.)
        self.assertIs(objective.best_bank,banks[0])


if __name__=='__main__':
    unittest.main()
