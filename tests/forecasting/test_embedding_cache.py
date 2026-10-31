"""Ensure disk-backed full runs retain exact predictions and match provenance."""
import csv
from datetime import date,timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
sys.path.insert(0,str(ROOT/'experiments/reproduction'))
sys.path.insert(0,str(ROOT/'experiments/data'))
from embedding_cache import load_cached_library
from model import load_library
from run_backtest import parser,run_experiment
from prepare_weekly_archive import common_schedule
from run_full import compare_runs


class FullRunTests(unittest.TestCase):
    def test_disk_embeddings_keep_every_curve_and_window_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'curves.csv'
            with source.open('w') as f:
                w=csv.writer(f);w.writerow(['curve_id','step','value'])
                for name,values in [('flat',[2]*12),('a',[3,5,3,6,10,20,15,0,0,0,0,0,0,0,0,0,0,0]),('s-2',list(range(25)))]:
                    w.writerows((name,i,v) for i,v in enumerate(values,1))
            reference=load_library(source,5,4)
            cached=load_cached_library(source,root/'cache',5,4)
            for a,b in zip(reference[:2],cached[:2]): np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(reference[2],cached[2][np.arange(len(cached[0]))])
            np.testing.assert_array_equal(reference[3],cached[3])
            truth=root/'truth.csv'
            truth.write_text('target_end_date,value\n'+''.join(f'{date(2020,1,4)+timedelta(weeks=i)},{100+i*2}\n' for i in range(20)))
            for label,library in [('memory',reference),('disk',cached)]:
                args=parser().parse_args(['--truth',str(truth),'--synthetic',str(source),'--closest','3',
                    '--start-date','2020-03-07','--max-origins','2','--output-dir',str(root/label)])
                run_experiment(args,embeddings=library)
            for name in ['predictions.csv','matches.csv','forecasts.csv']:
                self.assertEqual((root/'memory'/name).read_bytes(),(root/'disk'/name).read_bytes())
            with (root/'disk/matches.csv').open() as f: self.assertEqual(len(list(csv.DictReader(f))),6)

    def test_full_schedule_requires_all_four_future_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i in range(50):
                path=root/f'{i}.csv'
                length=149 if i==0 else 145
                path.write_text('target_end_date,value\n'+''.join(f'{date(2020,1,25)+timedelta(weeks=j)},1\n' for j in range(length)))
                rows.append(dict(series=f's{i}',truth=path.name))
            with (root/'manifest.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=['series','truth']);w.writeheader();w.writerows(rows)
            entries,origins=common_schedule(root/'manifest.csv')
            self.assertEqual(len(origins),112)
            self.assertEqual(origins[0],'2020-08-15')
            self.assertEqual(origins[-1],'2022-10-01')
            self.assertTrue(all(date.fromisoformat(b)-date.fromisoformat(a)==timedelta(weeks=1) for a,b in zip(origins,origins[1:])))

    def test_bo_paper_comparison_rejects_different_protocol_or_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=dict(method={'seed':13},origins=['2020-08-15'],truth_sha256={'A':'x'},source_sha256={'model':'y'},packages={'numpy':'z'})
            for name in ['paper','bo']:
                path=root/name; (path/'05_evaluation').mkdir(parents=True)
                (path/'run_config.json').write_text(json.dumps(config))
                (path/'05_evaluation/scores.csv').write_text('series,forecast_origin,target_end_date,horizon,model,true_value,abs_error,wis\nA,2020-08-15,2020-08-22,1,smoa,100,5,3\n')
            compare_runs(root/'paper',root/'bo',root/'comparison')
            self.assertEqual(json.loads((root/'comparison/metadata.json').read_text())['targets'],1)
            different=dict(config,method={'seed':99})
            (root/'bo/run_config.json').write_text(json.dumps(different))
            with self.assertRaisesRegex(ValueError,'protocol differs'):
                compare_runs(root/'paper',root/'bo',root/'bad')


if __name__=='__main__': unittest.main()
