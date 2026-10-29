"""End-to-end validation of stage order and applied Bayesian selection."""
import csv
from datetime import date,timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PipelineTests(unittest.TestCase):
    def test_state_manifest_prediction_reports_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'pipeline';manifest=root/'manifest.csv'
            manifest.write_text('series,truth,as_of_truth\nA,A.csv,\nB,B.csv,\n')
            for series,scale in [('A',100),('B',200)]:
                (root/f'{series}.csv').write_text('target_end_date,value\n'+''.join(
                    f'{date(2020,1,4)+timedelta(weeks=i)},{1000+scale*i}\n' for i in range(28)))
            command=[sys.executable,str(Path(__file__).with_name('run_pipeline.py')),
                '--manifest',str(manifest),'--synthetic-curves-per-family','1','--workers','1',
                '--initial-points','2','--iterations','1','--bo-num-curves-range','1','2',
                '--bo-k-range','2','3','--bo-closest-range','1','2','--bo-max-origins-per-curve','2',
                '--start-date','2020-02-29','--max-origins','2','--output-dir',str(out)]
            completed=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            benchmark=json.loads((out/'04_prediction/summary.json').read_text())
            evaluation=json.loads((out/'05_evaluation/summary.json').read_text())
            self.assertEqual(evaluation['series'],2)
            self.assertEqual(evaluation['forecasts_per_model'],16)
            self.assertAlmostEqual(evaluation['models']['smoa']['mae'],benchmark['pooled']['mae'])
            self.assertAlmostEqual(evaluation['models']['smoa']['wis'],benchmark['pooled']['wis'])
            for subdir in ['01','02']:
                self.assertTrue((out/f'05_evaluation/{subdir}/forecast_vs_actual.png').is_file())
            resumed=subprocess.run([*command,'--resume'],capture_output=True,text=True)
            self.assertEqual(resumed.returncode,0,resumed.stderr)
            self.assertIn('Benchmark already complete and verified',resumed.stdout)

    def test_all_five_stages_and_selected_parameters_are_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);truth=root/'truth.csv';out=root/'pipeline'
            truth.write_text('target_end_date,value\n'+''.join(
                f'{date(2020,1,4)+timedelta(weeks=i)},{1000+100*i}\n' for i in range(28)))
            completed=subprocess.run([sys.executable,str(Path(__file__).with_name('run_pipeline.py')),
                '--truth',str(truth),'--initial-points','2','--iterations','1',
                '--synthetic-curves-per-family','1','--workers','1',
                '--bo-num-curves-range','1','2','--bo-k-range','2','3','--bo-closest-range','1','2',
                '--bo-max-origins-per-curve','2',
                '--start-date','2020-02-29','--max-origins','2','--output-dir',str(out)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            manifest=json.loads((out/'pipeline.json').read_text())
            self.assertEqual([s['stage'] for s in manifest['stages']],[1,2,3,4,5])
            self.assertEqual(manifest['status'],'complete')
            self.assertFalse(manifest['full_search_budget'])
            self.assertFalse(manifest['publication_reproduction_complete'])
            params=json.loads((out/'02_optimization/best_params.json').read_text())
            config=json.loads((out/'04_prediction/metadata.json').read_text())['config']
            self.assertEqual(config['k'],params['k']);self.assertEqual(config['closest'],params['closest'])
            self.assertEqual(Path(config['synthetic']).resolve(),(out/'02_optimization/best_library/curves.csv').resolve())
            with (out/'04_prediction/predictions.csv').open() as f:
                reader=csv.DictReader(f);rows=list(reader)
                self.assertNotIn('true_value',reader.fieldnames)
            self.assertEqual(len(rows),8)
            result=json.loads((out/'05_evaluation/summary.json').read_text())
            legacy=json.loads((out/'04_prediction/summary.json').read_text())
            self.assertAlmostEqual(result['models']['smoa']['wis'],legacy['overall']['wis'])
            self.assertEqual(len(result['coverage_by_level']['smoa']),11)
            self.assertEqual(result['models']['smoa']['mae'],legacy['overall']['mae'])
            report=(out/'05_evaluation/report.md').read_text()
            self.assertIn(f"{result['models']['smoa']['wis']:,.2f}",report)
            self.assertIn('MAE',completed.stdout)
            self.assertIn('WIS',completed.stdout)
            for figure in ['forecast_vs_actual','metrics_by_horizon']:
                self.assertTrue((out/f'05_evaluation/{figure}.png').read_bytes().startswith(b'\x89PNG'))
                self.assertIn('<svg',(out/f'05_evaluation/{figure}.svg').read_text())
            resumed=subprocess.run([*completed.args,'--resume'],capture_output=True,text=True)
            self.assertEqual(resumed.returncode,0,resumed.stderr)
            self.assertIn('BO already complete and verified',resumed.stdout)


if __name__=='__main__': unittest.main()
