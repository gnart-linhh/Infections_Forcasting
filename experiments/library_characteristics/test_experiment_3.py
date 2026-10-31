"""Sampling controls, unchanged forecast core, paired scores and transactional resume."""
import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import experiments.library_characteristics.experiment_3 as e3
from experiments.forecasting.model import forecast, load_library
from generator import SyntheticCurve


class LibraryTests(unittest.TestCase):
    def test_curve_budget_nested_mixture_and_reproducibility(self):
        families = np.repeat(np.arange(3), 15)
        with patch.object(e3,'CURVES',6):
            mixed = e3.select_curves(families,0,'mixed')
            self.assertEqual(len(mixed),6)
            np.testing.assert_array_equal(mixed,e3.select_curves(families,0,'mixed'))
            self.assertTrue(np.all(np.diff(mixed)>0))
            self.assertEqual(np.bincount(families[mixed]).tolist(),[2,2,2])
            for i,lib in enumerate(e3.LIBRARIES[:3]):
                single = e3.select_curves(families,0,lib)
                self.assertEqual(len(single),6)
                self.assertTrue(set(mixed[families[mixed]==i]).issubset(single))
            self.assertFalse(np.array_equal(mixed,e3.select_curves(families,1,'mixed')))

    def test_window_budget_nested_families_and_source_order(self):
        codes = np.concatenate([np.repeat(i*e3.N+np.arange(1,11),3) for i in range(3)])
        with patch.object(e3,'WINDOWS',12):
            mixed = e3.select_windows(codes,1,'mixed')
            self.assertEqual(len(mixed),12)
            self.assertTrue(np.all(np.diff(mixed)>0))
            family = (codes-1)//e3.N
            self.assertEqual(np.bincount(family[mixed]).tolist(),[4,4,4])
            for i,lib in enumerate(e3.LIBRARIES[:3]):
                single = e3.select_windows(codes,1,lib)
                self.assertEqual(len(single),12)
                self.assertTrue(set(mixed[family[mixed]==i]).issubset(single))
                local = np.flatnonzero(family==i)
                np.testing.assert_array_equal(single,local[e3.select_windows(codes[local],1,lib)])

    def test_insufficient_budget_raises_without_replacement(self):
        with patch.object(e3,'CURVES',6),patch.object(e3,'WINDOWS',12):
            with self.assertRaises(ValueError): e3.select_curves([0,0,0],0,'rollercoaster')
            with self.assertRaises(ValueError): e3.select_windows(np.arange(1,6),0,'rollercoaster')

    def test_aggregate_matches_original_forecast_core(self):
        rng = np.random.default_rng(15)
        x,y = rng.normal(size=(100,5)),rng.normal(size=(100,4))
        history = np.array([12.,15.,14.,19.,16.,20.])
        point,ids,_ = forecast(history,x,y,closest=15,smoothing='none')
        np.testing.assert_array_equal(point,e3.aggregate(history[-1],y[ids]))

    def test_embedding_matches_core_omit_bridge_and_curve_boundaries(self):
        curves = [SyntheticCurve(np.arange(1.,25.)**2,'sir_rollercoaster','weekly','counts',False),
                  SyntheticCurve(np.arange(3.,30.)**1.5,'seasonal','weekly','counts',True)]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'curves.csv'
            with path.open('w') as h:
                writer=csv.writer(h);writer.writerow(['curve_id','step','value'])
                for ident,c in enumerate(curves,1):
                    writer.writerows((ident,j,float(v)) for j,v in enumerate(c.ts,1))
            xx,yy,ids,_=load_library(path,5,5)
            x,y,codes=e3.embed(curves,[1,2],5,root/'scratch',h=4)
            np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy[:,1:])
            np.testing.assert_array_equal(codes,ids.astype(int))
            del x,y,codes

    def test_equal_distance_ties_keep_original_source_order(self):
        x=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        index=e3.ExactNeighbors(x)
        ids,distance=index.query([[0.,0.]],3)
        np.testing.assert_array_equal(ids[0],[0,1,2])
        np.testing.assert_array_equal(distance[0],[1.,1.,1.])

    def test_score_rmse_is_mean_of_seed_rmse_not_pooled(self):
        rows=[]
        for seed,error in enumerate([[0.,2.],[0.,4.],[0.,6.]]):
            for j,value in enumerate(error):
                rows.append(dict(seed=seed,regime='full',library='mixed',case_id=j,horizon=1,phase='stable',
                    case_level='low',history_stage='early',error=value,absolute_error=abs(value),
                    normalized_absolute_error=abs(value),normalized_minimum_distance=.1,normalized_median_distance=.2))
        scores,summary=e3.score_tables(pd.DataFrame(rows))
        row=summary.loc[summary.dimension.eq('overall')].iloc[0]
        self.assertAlmostEqual(row.RMSE_mean,np.mean([np.sqrt(2),np.sqrt(8),np.sqrt(18)]))
        self.assertNotAlmostEqual(row.RMSE_mean,np.sqrt(np.mean(np.array([0,2,0,4,0,6])**2)))
        self.assertEqual(row.targets_per_seed,2)

    def test_paired_deltas_use_same_regime_seed_and_target(self):
        rows=[]
        for seed in range(3):
            for lib,err in [('mixed',10.),('rollercoaster',12.)]:
                for j,state in enumerate(['A','B']):
                    rows.append(dict(seed=seed,regime='equal_windows',library=lib,case_id=j,horizon=1,
                        series=state,phase='stable',case_level='low',history_stage='early',actual=100.,absolute_error=err))
        result=e3.paired_comparisons(pd.DataFrame(rows))
        row=result.loc[result.dimension.eq('overall')].iloc[0]
        self.assertEqual(row.delta_MAE,2.);self.assertEqual(row.unique_targets,2)
        self.assertEqual(row.lo,2.);self.assertEqual(row.hi,2.)

    def test_database_completion_is_transactional(self):
        with tempfile.TemporaryDirectory() as temp:
            db=e3.connect(temp);e3.initialize(db)
            predictions=[(j,h,2.) for j in range(2500) for h in range(1,5)]
            matches=[(j,0.,1.,2.) for j in range(2500)]
            e3.save_run(db,0,'full','mixed',predictions,matches,{'reused':True})
            with self.assertRaises(sqlite3.IntegrityError):
                e3.save_run(db,0,'full','mixed',predictions,matches,{})
            self.assertEqual(db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],10000)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM runs').fetchone()[0],1)
            with self.assertRaises(ValueError):e3.save_run(db,1,'full','mixed',predictions[:-1],matches,{})
            db.close()


if __name__=='__main__':unittest.main()
