import csv
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/reproduction'))
from experiment_0 import schedule,figure2_embeddings,figure2_initialization,select_examples
import uncertainty


class ExperimentZeroTests(unittest.TestCase):
    def test_source_uq_policy_is_explicit_and_includes_unmatured_labels(self):
        prior=[dict(horizon=h,target_end_date='2020-02-01',forecast=1000.,baseline=1000.,true_value=1100.)
               for _ in range(6) for h in range(1,5)]
        prior.append(dict(horizon=1,target_end_date='2020-03-01',forecast=1000.,baseline=1000.,true_value=5000.))
        causal=uncertainty.calibration_plan(prior,'2020-02-08')
        source=uncertainty.calibration_plan(prior,'2020-02-08',policy='figure2_retrospective')
        self.assertEqual(causal[('forecast',1)]['count'],6)
        self.assertEqual(source[('forecast',1)]['count'],7)
        self.assertNotEqual(causal[('forecast',1)]['size'],source[('forecast',1)]['size'])
        with self.assertRaises(ValueError): uncertainty.calibration_plan(prior,'2020-02-08',policy='typo')

    def test_paper_dates_mean_first_target_not_origin(self):
        plan=schedule()
        self.assertEqual(len(plan),24)
        self.assertEqual(plan.iloc[0].forecast_origin,'2020-03-14')
        self.assertEqual(plan.iloc[0].forecast_date,'2020-03-16')
        self.assertEqual(plan.iloc[0].first_target_end_date,'2020-03-21')
        self.assertEqual(plan.iloc[0].last_target_end_date,'2020-04-11')
        self.assertEqual(set(plan.series),{'California','Florida','New York','Texas'})

    def test_figure_embedding_omits_bridge_and_override_restores_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'curves.csv'
            path.write_text('curve_id,step,value\n'+''.join(f'a,{i},{v}\n' for i,v in enumerate([1,3,6,10,15,21],1)))
            x,y,_,_=figure2_embeddings(path,k=2,h=2)
            np.testing.assert_array_equal(x,[[2,3]])
            np.testing.assert_array_equal(y,[[5,6]])
        initial=uncertainty.INITIAL_SIZES
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            with figure2_initialization():
                self.assertEqual(uncertainty.INITIAL_SIZES,(20,10,6,6))
                raise RuntimeError('interrupted')
        self.assertEqual(uncertainty.INITIAL_SIZES,initial)

    def test_example_filter_rejects_date_shift_and_missing_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prediction=root/'prediction';prediction.mkdir()
            rows=[dict(series='California',forecast_origin='2020-03-14',horizon=h,
                       target_end_date=str(pd.Timestamp('2020-03-14').date()+pd.Timedelta(days=7*h)),forecast=1,baseline=1) for h in range(1,5)]
            rows[0]['target_end_date']='2020-03-28'
            with (prediction/'predictions.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Target date'):
                select_examples(prediction,root/'selected',{'2020-03-14'})
            with (prediction/'predictions.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows[:-1])
            with self.assertRaisesRegex(ValueError,'Missing'):
                select_examples(prediction,root/'selected',{'2020-03-14'})


if __name__=='__main__': unittest.main()
