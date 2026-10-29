"""Mathematical checks of the smoother; numerical R parity is not claimed."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.interpolate import make_smoothing_spline

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/forecasting'))
from smoothing import thin_plate_gcv, _basis


class SmoothingTests(unittest.TestCase):
    def test_affine_null_space_and_translation_scale_equivariance(self):
        for n in [6,7,20,41]:
            line=7+3*np.arange(n)
            np.testing.assert_allclose(thin_plate_gcv(line),line,rtol=1e-10,atol=1e-9)
        y=np.array([1,2,5,12,9,8,15,16,11,9,4,2.],float)
        np.testing.assert_allclose(thin_plate_gcv(3*y+100),3*thin_plate_gcv(y)+100,rtol=1e-7)

    def test_rank_and_gcv_match_hat_matrix_definition(self):
        y=np.random.default_rng(41).normal(size=31)
        fitted,info=thin_plate_gcv(y,return_info=True)
        self.assertEqual(info['basis_size'],round(len(y)/2))
        b,d=_basis(len(y),info['basis_size'])
        weights=1/(1+np.exp(info['log_lambda'])*d)
        hat=(b*weights)@b.T
        np.testing.assert_allclose(fitted,hat@y,atol=1e-10)
        self.assertAlmostEqual(info['edf'],np.trace(hat))
        self.assertAlmostEqual(info['gcv'],len(y)*sum((y-fitted)**2)/(len(y)-np.trace(hat))**2)
        self.assertLessEqual(info['edf'],info['basis_size'])
        self.assertGreaterEqual(info['edf'],2)

    def test_rank_truncation_changes_full_smoothing_spline(self):
        y=np.array([0,3,0,8,0,20,0,6,0,11,0,7.])
        fitted=thin_plate_gcv(y)
        legacy=make_smoothing_spline(np.arange(len(y),dtype=float),y)(np.arange(len(y)))
        self.assertFalse(np.allclose(fitted,legacy))


if __name__=='__main__': unittest.main()
