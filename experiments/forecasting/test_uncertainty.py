import unittest
import numpy as np
from uncertainty import dispersion, fit_size, interval_quantiles, scores


class UncertaintyTests(unittest.TestCase):
    def test_wis_matches_weighted_interval_formula(self):
        q = np.array([1,2,3,4,5,6,7], dtype=float)
        actual = 8
        interval_scores = [(hi-lo) + 2/alpha*max(lo-actual, 0) + 2/alpha*max(actual-hi, 0)
                           for alpha, lo, hi in [(0.05,1,7), (0.2,2,6), (0.5,3,5)]]
        expected = (0.5*abs(actual-4) + sum(alpha/2*s for alpha,s in zip([0.05,0.2,0.5], interval_scores)))/3.5
        result = scores(q, actual)
        self.assertAlmostEqual(result['wis'], expected)
        self.assertAlmostEqual(result['wis'], result['sharpness']+result['overprediction']+result['underprediction'])
        self.assertEqual(result['coverage_95'], 0)

    def test_only_matured_outcomes_enter_fit(self):
        prior = [dict(horizon=j, target_end_date='2020-01-01', forecast=1000.0, true_value=1100.0)
                 for _ in range(6) for j in range(1,5)]
        prior += [dict(horizon=1, target_end_date='2020-12-31', forecast=1000.0, true_value=1e8)]
        size, method, n = dispersion(prior, '2020-02-01', 1)
        changed = [dict(r, true_value=1) if r['target_end_date'] > '2020-02-01' else r for r in prior]
        self.assertEqual((size, method, n), dispersion(changed, '2020-02-01', 1))
        self.assertEqual(n, 6)
        self.assertEqual(method, 'mle')

    def test_initial_sizes_and_threshold(self):
        for h, expected in enumerate([20,10,6,0.3], 1):
            self.assertEqual(dispersion([], '2020-01-01', h), (expected, 'initial', 0))
        prior = [dict(horizon=j, target_end_date='2020-01-01', forecast=1000., true_value=1100.)
                 for _ in range(5) for j in range(1,5)]
        self.assertEqual(dispersion(prior, '2020-02-01', 1)[1], 'initial')

    def test_size_recovery_on_negative_binomial_data(self):
        rng = np.random.default_rng(47)
        actuals = rng.negative_binomial(8, 8/(8+1000), 5000)
        fitted = fit_size(np.full(5000, 1000), np.maximum(actuals, 1))
        self.assertAlmostEqual(fitted, 8, delta=0.6)

    def test_quantile_recentring_and_reproducibility(self):
        a = interval_quantiles(1000, 10, np.random.default_rng(13))
        b = interval_quantiles(1000, 10, np.random.default_rng(13))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a[3], 1000)
        self.assertTrue(np.all(np.diff(a) >= 0))
        self.assertEqual(scores(a, 1000)['coverage_95'], 1)

    def test_low_forecasts_use_observed_mean_rule(self):
        # Upstream substitutes observed means when point predictions are <100.
        self.assertEqual(fit_size([2,3,4], [150,180,220]), fit_size([150,180,220], [150,180,220]))


class BacktestIntegrationTests(unittest.TestCase):
    def test_future_truth_cannot_change_issued_quantiles(self):
        import csv
        from datetime import date, timedelta
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        script = Path(__file__).with_name('run_backtest.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic = root/'synthetic.csv'
            synthetic.write_text('curve_id,step,value\n' + ''.join(f'a,{i+1},{1000+10*i}\n' for i in range(30)))
            truth = root/'truth.csv'
            original = [1000+20*i for i in range(28)]
            outputs = []
            for attempt in range(2):
                values = original.copy()
                if attempt:
                    # Seventh origin is index 8 (k=2): change only its future.
                    values[9:] = [v*50 for v in values[9:]]
                truth.write_text('target_end_date,value\n' + ''.join(
                    f'{date(2020,1,4)+timedelta(weeks=i)},{v}\n' for i,v in enumerate(values)))
                out = root/f'out{attempt}'
                subprocess.run([sys.executable, str(script), '--synthetic', str(synthetic), '--truth', str(truth),
                    '--k', '2', '--smoothing', 'none', '--start-date', '2020-01-04', '--max-origins', '7',
                    '--probabilistic', '--output-dir', str(out)], check=True, capture_output=True, text=True)
                with (out/'quantiles.csv').open() as handle:
                    outputs.append(list(csv.DictReader(handle)))
                with (out/'forecasts.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([int(r['calibration_n']) for r in rows[-4:]], [6,5,4,3])
                self.assertTrue(all(r['dispersion_method']=='mle' for r in rows[-4:]))
            self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
