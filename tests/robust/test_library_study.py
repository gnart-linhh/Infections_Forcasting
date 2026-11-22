import unittest
import numpy as np
from experiments.robust.library_study import balanced_rows,dynamics_bins,embed_new,instantiate
from experiments.robust.method import RobustSMOA
from generator import change_point_sir

class LibraryTests(unittest.TestCase):
    def test_balanced_exact_budget_unique_reproducible_sparse(self):
        labels=np.repeat([0,1,2],[2,50,80])
        a=balanced_rows(labels,30,42)
        np.testing.assert_array_equal(a,balanced_rows(labels,30,42))
        self.assertEqual(len(a),30);self.assertEqual(len(np.unique(a)),30)
        np.testing.assert_array_equal(np.bincount(labels[a]),[2,14,14])
        with self.assertRaises(ValueError):balanced_rows(labels,200,42)

    def test_bin_definitions_only_history(self):
        x=np.array([[1,2,3,4,5],[-1,-2,-3,-4,-5],[1,-1,1,-1,0.]])
        b=dynamics_bins(x)
        self.assertEqual(b.dtype,np.dtype('u1'))
        self.assertTrue(np.all((b>=0)&(b<12)));self.assertNotEqual(b[0],b[1])
        np.testing.assert_array_equal(b,dynamics_bins(x*1000))

    def test_generator_conservation_semantics_reproducibility_units(self):
        a,states,meta=change_point_sir(np.random.default_rng(42),return_state=True)
        b=change_point_sir(np.random.default_rng(42))
        np.testing.assert_array_equal(a.ts,b.ts)
        np.testing.assert_allclose(states.sum(axis=1),1,atol=1e-7)
        self.assertGreaterEqual(states.min(),-1e-8)
        self.assertEqual((a.time_cadence,a.scale),('weekly','counts'))
        self.assertEqual(len(a.ts),104);self.assertTrue(np.isfinite(a.ts).all());self.assertTrue((a.ts>=0).all())
        constant=change_point_sir(np.random.default_rng(1),change_times=[],multipliers=[])
        unchanged=change_point_sir(np.random.default_rng(1),change_times=[20],multipliers=[1.])
        np.testing.assert_allclose(constant.ts,unchanged.ts,atol=1)
        with self.assertRaises(ValueError):change_point_sir(np.random.default_rng(1),change_times=[10,10],multipliers=[1,1])

    def test_embedding_bridge_and_library_cache_isolation(self):
        curve=change_point_sir(np.random.default_rng(42))
        x,y,c=embed_new([curve]);self.assertEqual(x.shape[1],5);self.assertEqual(y.shape[1],4)
        self.assertEqual(x.dtype,np.dtype('<f8'));self.assertTrue((c==0).all())
        base=RobustSMOA(x,y,closest=5)
        a=instantiate(base,np.arange(min(20,len(x))),np.empty((0,5)),np.empty((0,4)))
        history=np.arange(6.)+10
        a.variants(history,4,0);self.assertEqual(a.match_calls,1);self.assertEqual(base.match_calls,0)
        first=a.match(history).indices.copy();a.variants(history,.25,0)
        np.testing.assert_array_equal(first,a.match(history).indices)
        window=np.lib.stride_tricks.sliding_window_view(np.maximum(curve.ts,1e-8),11)
        from model import valid_history_windows
        expected=np.diff(window[valid_history_windows(window[:,:6])],axis=1)
        np.testing.assert_array_equal(y,expected[:,6:])
        from experiments.robust.explain import DirectForecast
        wrapper=DirectForecast(a,dict(temperature=4.,beta=0.),4,reference_engine=base)
        result=wrapper(history)[0]
        predicted=a.variants(history,4.,0.)[0]['M3'][3]
        self.assertEqual(result[0],predicted)
        self.assertEqual(result[1],predicted-base.match(history).baseline[3])
        before=(a.match_calls,base.match_calls)
        wrapper(history+1)
        self.assertEqual((a.match_calls,base.match_calls),(before[0]+1,before[1]+1))

if __name__=='__main__':unittest.main()
