"""No-bridge forecasts and explicit retrospective UQ, including future-label sensitivity."""
import csv
import gzip
from datetime import date,timedelta
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
sys.path.insert(0,str(ROOT/'experiments/reproduction'))
from embedding_cache import load_cached_library
from model import forecast,load_library
from run_backtest import parser,run_experiment
from uncertainty import calibration_plan
from run_full import parser as full_parser


class FullProtocolTests(unittest.TestCase):
    def test_full_defaults_to_main_r_uq_and_retrospective_requires_opt_in(self):
        args=full_parser().parse_args([])
        self.assertEqual(args.uq_policy,'matured')
        self.assertEqual(args.bridge_mode,'omit')  # Existing explicit user choice.
        self.assertEqual(full_parser().parse_args(['--uq-policy','figure2_retrospective']).uq_policy,
                         'figure2_retrospective')

    def test_omit_bridge_requires_additional_future_level_and_keeps_four_horizons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'curves.csv'
            source.write_text('curve_id,step,value\n'+''.join(f'a,{i},{v}\n' for i,v in enumerate([10,12,15,19,24,30,37,45],1)))
            kept=load_library(source,2,4)
            x,y,ids,steps=load_cached_library(source,root/'omit',2,4,bridge_mode='omit')
            self.assertEqual(len(kept[0]),2)
            self.assertEqual(x.shape,(1,2))
            np.testing.assert_array_equal(y,[[5,6,7,8]])
            self.assertEqual(ids[0],'a');self.assertEqual(steps[0],1)
            np.testing.assert_array_equal(forecast([95,97,100],x,y,1,smoothing='none')[0],[105,111,118,126])
            np.testing.assert_array_equal(forecast([95,97,100],kept[0],kept[1],1,smoothing='none')[0],[104,109,115,122])
            with self.assertRaisesRegex(ValueError,'Unknown bridge'):
                load_cached_library(source,root/'wrong',2,4,bridge_mode='invalid')

    def test_retrospective_counts_future_outcomes_but_never_current_issued_forecasts(self):
        origin=date(2020,8,29)
        prior=[]
        for i in range(6):
            issued=date(2020,7,18)+timedelta(weeks=i)
            for h in range(1,5):
                prior.append(dict(horizon=h,target_end_date=(issued+timedelta(weeks=h)).isoformat(),
                                  forecast=1000.,baseline=1000.,true_value=980.+20*(i%3)))
        matured=calibration_plan(prior,str(origin),policy='matured')
        retro=calibration_plan(prior,str(origin),policy='figure2_retrospective')
        self.assertEqual(matured[('forecast',4)]['count'],3)
        self.assertEqual(matured[('forecast',4)]['future_count'],0)
        self.assertEqual(retro[('forecast',4)]['count'],6)
        self.assertEqual(retro[('forecast',4)]['future_count'],3)
        changed=[dict(r,true_value=-1) if r['horizon']==4 and r['target_end_date']>str(origin) else r for r in prior]
        self.assertEqual(calibration_plan(changed,str(origin),policy='figure2_retrospective')[('forecast',4)]['excluded_negative'],3)
        self.assertEqual(calibration_plan(changed,str(origin),policy='matured')[('forecast',4)]['excluded_negative'],0)

    def test_future_labels_change_only_retrospective_intervals_and_leave_points_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'curves.csv'
            source.write_text('curve_id,step,value\n'+''.join(f'a,{i},{v}\n' for i,v in enumerate([10,12,15,19,24,30,37,45],1)))
            for policy in ['matured','figure2_retrospective']:
                for shock in [False,True]:
                    truth=root/f'truth_{shock}.csv'
                    truth.write_text('target_end_date,value\n'+''.join(
                        f'{date(2020,1,25)+timedelta(weeks=i)},{(1000+i*10)*(4 if shock and i>31 else 1)}\n' for i in range(40)))
                    output=root/f'{policy}_{shock}'
                    args=parser().parse_args(['--truth',str(truth),'--synthetic',str(source),
                        '--k','2','--closest','1','--smoothing','none','--bridge-mode','omit',
                        '--uq-policy',policy,'--calibration-truth','final','--probabilistic','--interval-draws','200',
                        '--warmup-start-date','2020-06-01','--start-date','2020-08-29','--max-origins','1','--output-dir',str(output)])
                    args.compress_matches=policy=='figure2_retrospective' and shock
                    run_experiment(args)
            self.assertEqual((root/'matured_False/quantiles.csv').read_bytes(),(root/'matured_True/quantiles.csv').read_bytes())
            self.assertNotEqual((root/'figure2_retrospective_False/quantiles.csv').read_bytes(),(root/'figure2_retrospective_True/quantiles.csv').read_bytes())
            reference=(root/'matured_False/predictions.csv').read_bytes()
            for policy in ['matured','figure2_retrospective']:
                for shock in [False,True]:
                    self.assertEqual((root/f'{policy}_{shock}/predictions.csv').read_bytes(),reference)
            with (root/'figure2_retrospective_False/calibration.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertTrue(any(int(r['future_calibration_n'])>0 for r in rows))
            self.assertEqual({r['calibration_policy'] for r in rows},{'figure2_retrospective'})
            with gzip.open(root/'figure2_retrospective_True/matches.csv.gz','rb') as f:
                self.assertEqual(f.read(),(root/'figure2_retrospective_False/matches.csv').read_bytes())


if __name__=='__main__':unittest.main()
