"""Checks for rank/tie handling, exact joins and chronology-safe risk diagnostics."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiment_5 import spearman,distance_bins,auc,average_precision,fit_risk,join_instances,partial_spearman,risk_diagnostics,replay_distances


class AnalogueQualityTests(unittest.TestCase):
    def test_spearman_ties_and_constant_values(self):
        self.assertAlmostEqual(spearman([1,1,3,4],[10,10,30,40]),1)
        self.assertAlmostEqual(spearman([1,2,3],[3,2,1]),-1)
        self.assertTrue(np.isnan(spearman([0,0,0],[1,2,3])))

    def test_bins_do_not_use_future_distribution_or_split_ties(self):
        values,cuts=distance_bins([0,0,0,0],[0,0,10])
        self.assertEqual(cuts.tolist(),[0])
        self.assertEqual(values.tolist(),[1,1,2])
        _,again=distance_bins([0,0,0,0],[0,1000000])
        np.testing.assert_array_equal(cuts,again)

    def test_auroc_and_ap_count_score_ties_as_one_threshold(self):
        self.assertEqual(auc([0,1],[.5,.5]),.5)
        self.assertEqual(average_precision([0,1],[.5,.5]),.5)
        self.assertEqual(auc([0,1],[0,1]),1)
        self.assertEqual(average_precision([0,1],[0,1]),1)
        self.assertTrue(np.isnan(auc([0,0],[.1,.2])))

    def test_risk_scaling_fitted_only_on_training(self):
        x=np.arange(20.)[:,None];y=(x[:,0]>9).astype(int)
        first,fit=fit_risk(x,y,np.array([[3.],[17.]]))
        second,other=fit_risk(x,y,np.array([[3.],[17.],[1e10]]))
        np.testing.assert_allclose(first,second[:2],rtol=0,atol=1e-12)
        self.assertEqual(fit['train_center'],other['train_center'])
        self.assertLess(first[0],first[1])
        p,status=fit_risk(x,np.zeros(20),x)
        self.assertEqual(status['status'],'constant_training_labels')
        self.assertTrue((p==0).all())

    def test_joins_reject_duplicates_missing_cases_and_shifted_dates(self):
        cases=pd.DataFrame([dict(series='A',forecast_origin='2020-04-04',normalization=10)])
        forecasts=pd.DataFrame([dict(series='A',forecast_origin='2020-04-04',target_end_date='2020-04-11',horizon=1,
            prediction=30,actual=10,normalization=10)])
        result=join_instances(cases,forecasts)
        self.assertEqual(result.absolute_error.iloc[0],20)
        self.assertTrue(result.extreme.iloc[0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):join_instances(pd.concat([cases,cases]),forecasts)
        with self.assertRaisesRegex(ValueError,'missing'):join_instances(cases,forecasts.assign(series='B'))
        with self.assertRaisesRegex(ValueError,'horizons'):join_instances(cases,forecasts.assign(target_end_date='2020-04-18'))

    def test_partial_rank_correlation_detects_fully_explained_scale(self):
        x=np.arange(1,101.)
        data=pd.DataFrame(dict(mean_distance_scaled=x,normalized_absolute_error=x,level_scale=x,recent_cases=x,
            growth=0,volatility=.1,history_weeks=30,phase='stable'))
        self.assertAlmostEqual(spearman(x,x),1)
        self.assertTrue(np.isnan(partial_spearman(data)))

    def test_exact_replay_measures_mean_and_rejects_changed_points(self):
        cases=pd.DataFrame(dict(series=['A','B'],forecast_origin=['2020-04-04']*2))
        contexts=dict(series=np.array(['A','B']),origin=np.array(['2020-04-04']*2),last=np.array([10.,10.]),
            query_k5=np.array([[0.]*5,[2.]*5]))
        x=np.array([[0.]*5,[1.]*5,[2.]*5]);y=np.array([[1.,2.,3.,4.],[3.,4.,5.,6.],[5.,6.,7.,8.]])
        matches=cases.assign(minimum_distance=0.,median_distance=2.5,maximum_distance=5.)
        rows=[]
        for state,points in [('A',[12,15,19,24]),('B',[14,19,25,32])]:
            for h,p in enumerate(points,1):rows.append(dict(series=state,forecast_origin='2020-04-04',horizon=h,prediction=p))
        forecasts=pd.DataFrame(rows)
        result=replay_distances(cases,contexts,x,y,matches,forecasts,m=2)
        self.assertTrue(result.mean_distance.eq(2.5).all())
        with self.assertRaisesRegex(ValueError,'point differs'):
            replay_distances(cases,contexts,x,y,matches,forecasts.assign(prediction=-1),m=2)

    def test_risk_training_uses_historical_vintage_not_future_final_labels(self):
        rows=[];panel=[]
        for i,state in enumerate(['A','B','C','D']):
            for split,origin in [('train','2020-11-07'),('test','2021-01-23')]:
                for h in [1,2,3,4]:
                    target=(pd.Timestamp(origin)+pd.Timedelta(weeks=h)).date().isoformat()
                    rows.append(dict(series=state,forecast_origin=origin,target_end_date=target,horizon=h,split=split,
                        prediction=100,actual=50,normalization=10,mean_distance_scaled=.1+i,level_scale=100+i,
                        recent_cases=100+i,growth=.1*i,volatility=.2,history_weeks=40+i,phase='rising' if i%2 else 'stable',extreme=bool(i%2)))
                    if split=='train':panel.append(dict(series=state,forecast_origin='2021-01-23',target_end_date=target,value=0 if i%2 else 100))
        data=pd.DataFrame(rows);history=pd.DataFrame(panel)
        _,predictions,fits=risk_diagnostics(data,history)
        changed=data.copy();changed.loc[changed.split.eq('train'),'actual']=999999
        changed.loc[changed.split.eq('train'),'extreme']=True
        _,again,other=risk_diagnostics(changed,history)
        pd.testing.assert_frame_equal(predictions,again)
        self.assertEqual(fits,other)
        self.assertTrue(all(v['max_training_target']<='2020-12-31' for v in fits.values()))


if __name__=='__main__':unittest.main()
