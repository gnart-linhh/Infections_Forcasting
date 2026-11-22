"""Regression, weighted aggregation, and regularization invariants."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments.robust.method import (RobustSMOA, original_forecast, similarity_weights,
                                      weighted_median, regularize)
from experiments.robust.evaluation import point_metrics, interval_metrics
from smoothing import thin_plate_gcv


def engine_fixture():
    rng = np.random.default_rng(42)
    return RobustSMOA(rng.normal(size=(70, 5)), rng.normal(size=(70, 4))*5, closest=20)


class MethodTests(unittest.TestCase):
    def test_original_regression_and_truncated_gam_at_all_lengths(self):
        engine = engine_fixture()
        for length in [6, 8, 11]:
            history = 20+np.arange(length)**2
            expected = original_forecast(history, engine.x, engine.y, engine.closest, 'gam', neighbor_index=engine.index)[0]
            with patch('model.thin_plate_gcv', wraps=thin_plate_gcv) as smooth:
                actual, _ = engine.variants(history, None, 0)
                self.assertEqual(len(smooth.call_args.args[0]), length)
            for value in actual.values():
                np.testing.assert_array_equal(value, expected)

    def test_uniform_weights_even_numpy_median(self):
        data = np.array([[0., 5.], [2., 3.], [8., 1.], [10., 9.]])
        np.testing.assert_array_equal(weighted_median(data, [1., 1., 1., 1.]), np.median(data, axis=0))
        self.assertEqual(weighted_median([0., 2., 8., 10.], [1., 1., 1., 1.]), 5.)

    def test_weighted_median_known_mass_and_half_boundary(self):
        self.assertEqual(weighted_median([10., 1., 3.], [.8, .1, .1]), 10.)
        self.assertEqual(weighted_median([0., 10.], [.5, .5]), 5.)
        self.assertEqual(weighted_median([0., 2., 8.], [0., .5, .5]), 5.)
        self.assertEqual(weighted_median([1., 1., 1.], [.1, .3, .6]), 1.)

    def test_weights_equal_zero_spread_extreme_temperatures_and_scale(self):
        np.testing.assert_array_equal(similarity_weights([0., 0., 0.], .1), [1/3]*3)
        d = np.array([3., 4., 6., 100.])
        np.testing.assert_allclose(similarity_weights(d, .5), similarity_weights(d*1e150, .5))
        np.testing.assert_allclose(similarity_weights(d, 1e-300), [1., 0., 0., 0.])
        np.testing.assert_allclose(similarity_weights(d, 1e300), [.25]*4)
        self.assertEqual(similarity_weights([0., 0., 0., 1.], .1).sum(), 1.)

    def test_weighted_extreme_finite_numbers(self):
        self.assertTrue(np.isfinite(weighted_median([1e308, 1e308], [1., 1.])))
        self.assertEqual(weighted_median([-1e308, 1e308], [1., 1.]), 0.)
        self.assertTrue(np.isfinite(similarity_weights([1e308, 1e307], 1e-300)).all())

    def test_invalid_weight_inputs_rejected(self):
        for d in [[np.nan], [np.inf], [-1.], []]:
            with self.assertRaises(ValueError):similarity_weights(d, 1.)
        for temperature in [0., -1., np.inf, np.nan]:
            with self.assertRaises(ValueError):similarity_weights([0., 1.], temperature)
        for values, weights in [([1., np.inf], [1., 1.]), ([1., 2.], [0., 0.]), ([1., 2.], [-1., 2.])]:
            with self.assertRaises(ValueError):weighted_median(values, weights)

    def test_regularization_identity_monotonicity_and_large_values(self):
        point = np.array([2., 40., 90., 100.])
        np.testing.assert_array_equal(regularize(point, 10, [1., 2., 3., 4.], 0)[0], point)
        np.testing.assert_array_equal(regularize(point, 10, np.zeros(4), 1)[0], point)
        value, lam = regularize(point, 10, [0., 1., 10., 100.], .5)
        self.assertTrue((np.diff(lam) <= 0).all())
        self.assertTrue(((lam > 0) & (lam <= 1)).all())
        self.assertTrue((value >= 0).all())
        self.assertAlmostEqual(value[1], 30.)
        large, tiny = regularize([1e308], 1e308, [1e308], 1e308)
        self.assertTrue(np.isfinite(large).all());self.assertGreater(tiny[0], 0)

    def test_dispersion_uses_cumulative_count_changes_and_window_scale(self):
        engine = engine_fixture()
        history = np.array([10., 20., 30., 40., 50., 60.])
        matched = engine.match(history)
        cumulative = np.cumsum(matched.future, axis=1)
        expected = (np.quantile(cumulative, .75, axis=0)-np.quantile(cumulative, .25, axis=0))/35.
        np.testing.assert_array_equal(matched.dispersion, expected)
        self.assertEqual(matched.scale, 35.)

    def test_rematching_for_changed_input_cache_for_identical_input(self):
        engine = engine_fixture()
        a = np.arange(6.)+10
        engine.match(a); self.assertEqual(engine.match_calls, 1)
        engine.match(a.copy()); self.assertEqual(engine.match_calls, 1)
        engine.match(a+1); self.assertEqual(engine.match_calls, 2)
        engine.match(np.arange(8.)+10); self.assertEqual(engine.match_calls, 3)

    def test_hand_computed_point_and_interval_metrics(self):
        m = point_metrics([1., 3.], [2., 1.])
        self.assertEqual(m['mae'], 1.5);self.assertEqual(m['bias'], -.5)
        self.assertAlmostEqual(m['rmse'], np.sqrt(2.5))
        score = interval_metrics([3.], [[0., 1., 2., 3., 4., 5., 6.]])
        self.assertAlmostEqual(score['wis'], (.025*6+.1*4+.25*2)/3.5)
        self.assertEqual(score['coverage_95'], 1.)
        self.assertEqual(score['width_80'], 4.)
        with self.assertRaises(ValueError):interval_metrics([3.], [[6., 5., 4., 3., 2., 1., 0.]])


if __name__ == '__main__':unittest.main()
