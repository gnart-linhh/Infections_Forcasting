import unittest
import numpy as np
import pandas as pd
from experiments.robust.method import RobustSMOA
from experiments.robust.policy import candidate_features,rank_labels,convex_forecast,RankPolicy,FEATURES
from experiments.robust.policy_experiment import select_policy
from experiments.robust.explain import DirectForecast

class PolicyTests(unittest.TestCase):
    def engine(self):
        y=np.array([[100.,-100.,0.,0.],[-100.,100.,0.,0.],[100.,100.,0.,0.]])
        return RobustSMOA(np.zeros((3,5)),y,closest=3,smoothing='none')

    def test_cumulative_quantiles_original_not_necessarily_middle(self):
        engine=self.engine();history=np.repeat(10.,6)
        x,candidates,ranked,order=candidate_features(engine,history)
        np.testing.assert_array_equal(candidates[:,1],engine.match(history).baseline)
        np.testing.assert_allclose(candidates[:,0],np.maximum(0,np.quantile(10+np.cumsum(engine.y,axis=1),.25,axis=0)))
        self.assertEqual(order[1,-1],1)
        self.assertTrue((np.diff(ranked,axis=1)>=0).all());self.assertTrue((ranked>=0).all())
        self.assertEqual(engine.match_calls,1)
        for t in [6,8,11]:
            a=candidate_features(engine,np.zeros(t))[0]
            self.assertEqual(a.shape,(4,6));self.assertTrue(np.isfinite(a).all())

    def test_ties_original_preference_and_rho_zero(self):
        ranked=np.array([[2.,2.,2.],[1.,2.,3.]])
        ordering=np.array([[0,1,2],[0,2,1]])
        np.testing.assert_array_equal(rank_labels(ranked,ordering,[10.,2.5]),[1,2])
        p=np.array([[.1,.2,.7],[.2,.6,.2]])
        original=np.array([2.,3.])
        np.testing.assert_array_equal(convex_forecast(original,ranked,p,0),original)
        self.assertEqual(convex_forecast(original,ranked,p,1)[0],2.)
        np.testing.assert_array_equal(convex_forecast([2.],[[2.,2.,2.]],[[.333333,.333333,.333333]],1),[2.])
        with self.assertRaises(ValueError):convex_forecast(original,ranked,p*2,1)

    def fixture(self):
        rng=np.random.default_rng(42);n=90
        frame=pd.DataFrame(rng.normal(size=(n,6)),columns=FEATURES)
        frame['horizon']=np.tile([1,2,3],30);frame['history_length']=6
        frame['dataset']='synthetic_training';frame['region']=[f'curve_{i//3}' for i in range(n)]
        frame['family']='fixture';frame['scale']=10.
        for j in range(3):frame[f'rank_{j}']=j+1.;frame[f'identity_{j}']=j
        frame['prediction']=2.;frame['actual']=rng.choice([1.,2.,3.],n)
        return frame

    def test_policy_deterministic_probabilities_and_no_real_fitting(self):
        frame=self.fixture();a=RankPolicy(3,42).fit(frame);b=RankPolicy(3,42).fit(frame)
        pa=a.probabilities(frame[FEATURES]);np.testing.assert_array_equal(pa,b.probabilities(frame[FEATURES]))
        np.testing.assert_allclose(pa.sum(axis=1),1);self.assertTrue((pa>=0).all())
        with self.assertRaises(ValueError):RankPolicy().fit(frame.assign(dataset='COVID19_early'))
        with self.assertRaises(ValueError):select_policy(frame,frame.assign(dataset='COVID19_early'),{'policy':{}},)

    def test_exact_direct_wrapper_executes_policy_and_rematches(self):
        fit=RankPolicy().fit(self.fixture());engine=self.engine();history=np.arange(6.)+10
        wrapper=DirectForecast(engine,dict(temperature=4,beta=0),2,policy=fit,rho=.5)
        result=wrapper(history)[0];expected=fit.forecast(engine,history,.5)[0][1]
        self.assertEqual(result[0],expected);self.assertEqual(result[1],expected-engine.match(history).baseline[1])
        old=engine.match_calls;wrapper(history+2);self.assertEqual(engine.match_calls,old+1)

    def test_future_labels_do_not_change_policy_predictions_and_no_change_selection(self):
        from experiments.robust.policy_experiment import predict_records
        frame=self.fixture();fit=RankPolicy().fit(frame)
        frame['weighted']=frame.prediction;frame['history_sha256']='0'*64
        for q in ['q025','q100','q250','q500','q750','q900','q975']:frame[q]=0.
        config=dict(seed=42,uncertainty=dict(draws=50),policy=dict(depths=[3],rhos=[0,1],minimum_improvement=.01,rmse_tolerance=.05,underprediction_tolerance=.02))
        a=predict_records(frame,fit,.25,config)
        b=predict_records(frame.assign(actual=1e9),fit,.25,config)
        pd.testing.assert_frame_equal(a,b,check_exact=True);self.assertNotIn('actual',a)
        _,frozen,_=select_policy(frame,frame.assign(dataset='synthetic_validation',actual=frame.prediction),config)
        self.assertEqual(frozen['rho'],0.)

if __name__=='__main__':unittest.main()
