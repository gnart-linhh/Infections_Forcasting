"""Checks for metric conventions, alignment, history availability and UQ provenance."""
import unittest
import numpy as np
import pandas as pd
from experiments.probabilistic.experiment_6 import (
    KEYS, LONG_QUANTILES, quantile_matrix, interval_arrays, wis_values,
    origin_features, history_bin, calibration_bin, verify_calibration,
    state_interval, partial_rho,
)
from experiments.forecasting.uncertainty import QUANTILES, scores


class ProbabilisticTests(unittest.TestCase):
    def test_inclusive_boundary_and_source_convention(self):
        values = np.arange(23., dtype=float)[None, :]
        a = interval_arrays(values, [1.])
        self.assertTrue(a['covered'][0, 1])
        self.assertFalse(a['source_covered'][0, 1])
        self.assertFalse(a['lower_miss'][0, 1])
        b = interval_arrays(values, [21.])
        self.assertTrue(b['covered'][0, 1])
        self.assertTrue(b['source_covered'][0, 1])
        self.assertEqual(a['width'][0, 1], 20.)

    def test_wis_exact_original_grid_not_23_quantile_average(self):
        values = (LONG_QUANTILES**2*1000)[None, :]
        indexes = [np.flatnonzero(np.isclose(LONG_QUANTILES, q))[0] for q in QUANTILES]
        for y in [-4., 0., 300., 10000.]:
            self.assertAlmostEqual(wis_values(values, [y])[0], scores(values[0, indexes], y)['wis'], places=11)

    def test_quantiles_reordered_by_keys_and_invalid_rows_rejected(self):
        keys = pd.DataFrame([['A','2020-01-04','2020-01-11',1], ['B','2020-01-04','2020-01-11',1]], columns=KEYS)
        rows = [dict(**row, model='smoa', quantile=q, value=q+(100 if row['series']=='B' else 0))
                for row in keys.to_dict('records') for q in LONG_QUANTILES]
        frame = pd.DataFrame(rows).sample(frac=1, random_state=1)
        result = quantile_matrix(frame, keys.iloc[::-1])
        self.assertGreater(result[0, 0], 100)
        for invalid in [pd.concat([frame, frame.iloc[[0]]]), frame.iloc[:-1], frame.assign(value=np.nan)]:
            with self.assertRaises(ValueError): quantile_matrix(invalid, keys)
        crossing = frame.copy(); crossing.loc[crossing.index[0], 'value'] = -1e9
        with self.assertRaises(ValueError): quantile_matrix(crossing, keys)

    def test_features_require_exact_vintage_and_ignore_future_vintages(self):
        dates = pd.date_range('2020-01-04', periods=8, freq='7D').strftime('%Y-%m-%d').tolist()
        origin = dates[-1]
        archive = pd.DataFrame(dict(as_of=[origin]*8, target_end_date=dates, value=np.arange(8)+100.))
        baseline = origin_features(archive, origin, dates)
        future = archive.assign(as_of='2020-03-07', value=1e9)
        self.assertEqual(baseline, origin_features(pd.concat([archive, future]), origin, dates))
        with self.assertRaises(ValueError): origin_features(archive.iloc[:-1], origin, dates)
        with self.assertRaises(ValueError): origin_features(archive.assign(as_of='2020-03-07'), origin, dates)

    def test_fixed_history_and_calibration_bins(self):
        self.assertEqual(history_bin(26), 'early <=26 weeks')
        self.assertEqual(history_bin(30), '27–52 weeks')
        self.assertEqual(history_bin(52), '27–52 weeks')
        self.assertEqual(history_bin(53), '53–104 weeks')
        self.assertEqual(history_bin(105), '105+ weeks')
        self.assertEqual(calibration_bin(49), '24–49 pairs')
        self.assertEqual(calibration_bin(50), '50–99 pairs')
        self.assertEqual(calibration_bin(100), '100+ pairs')

    def test_retrospective_counts_include_future_exclude_negative_and_current(self):
        f = pd.DataFrame([dict(series='A', forecast_origin='2020-02-01', target_end_date='2020-02-29', horizon=4,
            forecast=5., true_value=10., nb_size=2., calibration_n=1, negative_calibration_excluded=1)])
        c = pd.DataFrame([dict(series='A', forecast_origin='2020-02-01', horizon=4, model='smoa', nb_size=2.,
            method='mle', calibration_n=1, excluded_negative=1, calibration_policy='figure2_retrospective', future_calibration_n=1)])
        warmup = pd.DataFrame([dict(series='A', forecast_origin='2020-01-25', target_end_date='2020-02-22', horizon=4, true_value=10.),
                               dict(series='A', forecast_origin='2020-01-04', target_end_date='2020-02-01', horizon=4, true_value=-1.)])
        result = verify_calibration(f, c, warmup)
        self.assertEqual(result.future_calibration_n.iloc[0], 1)
        for wrong in [c.assign(future_calibration_n=0), c.assign(calibration_n=2), c.assign(calibration_policy='matured')]:
            with self.assertRaises(ValueError): verify_calibration(f, wrong, warmup)

    def test_cluster_interval_retains_within_state_rows(self):
        # Repeating identical within-state histories does not create new independent evidence.
        frame = pd.DataFrame(dict(series=['A','A','B','B'], value=[0.,1.,2.,3.]))
        interval = state_interval(frame, ['value'])
        repeat = state_interval(pd.concat([frame]*10), ['value'])
        np.testing.assert_allclose(interval, repeat)

    def test_context_adjustment_removes_shared_rank_trend(self):
        rng = np.random.default_rng(10)
        context = np.linspace(0, 1, 300)
        x = context+rng.normal(0, .03, 300)
        y = context+rng.normal(0, .03, 300)
        adjusted = partial_rho(x, y, np.column_stack([np.ones(300), context]))
        self.assertLess(abs(adjusted), .15)


if __name__ == '__main__': unittest.main()
