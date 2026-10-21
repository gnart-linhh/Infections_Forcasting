import csv
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from snapshots import SnapshotTruth, calibration_rows
from run_backtest import parser, run_experiment
from run_benchmark import aggregate


class SnapshotTests(unittest.TestCase):
    def test_exact_vintage_and_complete_history_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'snapshots.csv'
            path.write_text('as_of,target_end_date,value\n2020-01-11,2020-01-04,4\n2020-01-11,2020-01-11,5\n')
            snapshots = SnapshotTruth(path)
            with self.assertRaisesRegex(ValueError, 'Missing exact'):
                snapshots.history(date(2020,1,4), [date(2020,1,4)])
            with self.assertRaisesRegex(ValueError, 'match all weekly'):
                snapshots.history(date(2020,1,11), [date(2020,1,11)])
            self.assertEqual(snapshots.history(date(2020,1,11), [date(2020,1,4),date(2020,1,11)]).tolist(), [4,5])

    def test_duplicate_and_future_target_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'snapshots.csv'
            for rows in ['2020-01-04,2020-01-11,5\n',
                         '2020-01-04,2020-01-04,5\n2020-01-04,2020-01-04,6\n']:
                path.write_text('as_of,target_end_date,value\n'+rows)
                with self.assertRaises(ValueError):
                    SnapshotTruth(path)

    def test_calibration_refreshes_revised_outcomes(self):
        prior = [dict(horizon=1, target_end_date='2020-01-04', true_value=9999),
                 dict(horizon=1, target_end_date='2020-01-18', true_value=8888)]
        result = calibration_rows(prior, date(2020,1,11), [date(2020,1,4)], [300])
        self.assertEqual(result[0]['true_value'], 300)
        self.assertNotEqual(result[1]['true_value'], result[1]['true_value'])  # NaN is never fitted
        self.assertEqual(prior[0]['true_value'], 9999)

    def test_asof_backtest_ignores_final_values_for_history_and_mle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            days = [date(2020,1,4)+timedelta(weeks=i) for i in range(24)]
            synthetic = root/'curves.csv'
            synthetic.write_text('curve_id,step,value\n'+''.join(f'a,{i+1},{1000+i*15}\n' for i in range(30)))
            archive = root/'asof.csv'
            with archive.open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['as_of','target_end_date','value'])
                for origin in range(2,10):
                    w.writerows((days[origin],days[j],1000+j*20+origin) for j in range(origin+1))
            quantiles, forecasts = [], []
            for attempt, multiplier in enumerate([1,50]):
                truth = root/f'truth{attempt}.csv'
                truth.write_text('target_end_date,value\n'+''.join(f'{d},{multiplier*(1000+j*30)}\n' for j,d in enumerate(days)))
                out = root/f'out{attempt}'
                args = parser().parse_args(['--synthetic',str(synthetic), '--truth',str(truth),
                    '--as-of-truth',str(archive), '--k','2', '--smoothing','none', '--probabilistic',
                    '--warmup-start-date',str(days[2]), '--start-date',str(days[8]), '--max-origins','2',
                    '--output-dir',str(out)])
                summary = run_experiment(args)
                self.assertEqual(summary['warmup_origins'],6)
                self.assertEqual(summary['origins'],2)
                self.assertEqual(summary['history_mode'],'as_of')
                quantiles.append((out/'quantiles.csv').read_text())
                with (out/'forecasts.csv').open() as f:
                    forecasts.append(list(csv.DictReader(f)))
                self.assertEqual([int(r['calibration_n']) for r in forecasts[-1][:4]],[6,5,4,3])
                self.assertTrue(all(r['dispersion_method']=='mle' for r in forecasts[-1]))
                with (out/'warmup_forecasts.csv').open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))),24)
            self.assertEqual(quantiles[0],quantiles[1])
            self.assertNotEqual(forecasts[0][0]['wis'],forecasts[1][0]['wis'])
            self.assertEqual(forecasts[0][0]['baseline'],str(float(1000+8*20+8)))


class AggregationTests(unittest.TestCase):
    def test_micro_weighting_and_macro_ratios_are_distinct(self):
        result = aggregate([dict(series='A',origins=10,overall=dict(mae=2.,baseline_mae=4.)),
                            dict(series='B',origins=1,overall=dict(mae=10.,baseline_mae=5.))])
        self.assertAlmostEqual(result['pooled']['mae'],30/11)
        self.assertEqual(result['relative']['mae']['mean_series_ratio'],1.25)
        self.assertEqual(result['relative']['mae']['series_better_than_baseline'],1)

    def test_zero_baseline_and_duplicate_series(self):
        row = dict(series='A',origins=1,overall=dict(mae=1.,baseline_mae=0.))
        self.assertEqual(aggregate([row])['relative']['mae']['excluded_zero_baseline'],1)
        self.assertIsNone(aggregate([row])['relative']['mae']['mean_series_ratio'])
        with self.assertRaises(ValueError):
            aggregate([row,row])


class TruthPolicyTests(unittest.TestCase):
    def test_negative_revisions_require_explicit_policy(self):
        from run_backtest import load_truth
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'truth.csv'
            p.write_text('target_end_date,value\n2020-01-04,-100\n2020-01-11,20\n')
            with self.assertRaisesRegex(ValueError,'Negative weekly revisions'):
                load_truth(p)
            _, values = load_truth(p, 'clip')
            self.assertEqual(values.tolist(),[1,20])


if __name__ == '__main__':
    unittest.main()
