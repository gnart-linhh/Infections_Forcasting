"""Exercise notebook cells with issued fixture forecasts, including plots and scores."""
from contextlib import redirect_stdout
import csv
from datetime import date, timedelta
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/forecasting'))
from run_backtest import parser, run_experiment
from runtime import atomic_json, file_hash


def execute_cells(filename, overrides):
    namespace = {}
    notebook = nbformat.read(ROOT / 'experiments/forecasting' / filename, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        exec(compile(cell.source, filename, 'exec'), namespace)
        if 'parameters' in cell.metadata.get('tags', []):
            namespace.update(overrides)
    return namespace


class NotebookTests(unittest.TestCase):
    def test_forecasting_fan_and_recomputed_evaluation_use_issued_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp, patch('IPython.display.display') as display, redirect_stdout(io.StringIO()):
            root = Path(tmp); run = root / 'fixture'
            truth = root / 'California.csv'
            truth.write_text('target_end_date,value\n' + ''.join(
                f'{date(2020,1,4)+timedelta(weeks=i)},{1000+100*i}\n' for i in range(32)))
            manifest = root / 'manifest.csv'
            manifest.write_text('series,truth,as_of_truth\nCalifornia,California.csv,\n')
            selection = run / '02_optimization'
            library = selection / 'best_library/curves.csv'
            library.parent.mkdir(parents=True)
            library.write_text('curve_id,step,value\n' + ''.join(f'a,{i},{100*i}\n' for i in range(1,41)))
            atomic_json(selection / 'best_params.json', dict(num_curves=1,k=2,closest=1))
            atomic_json(selection / 'config.json', dict(profile='custom',initial_points=2,iterations=1,full_search_budget=False))
            atomic_json(selection / 'checkpoint.json', dict(history=[dict(trial=i) for i in range(3)]))
            atomic_json(selection / 'metadata.json', dict(status='complete',artifacts_sha256={
                name:file_hash(selection/name) for name in ['best_params.json','best_library/curves.csv']}))
            prediction = run / '04_prediction'
            args = parser().parse_args(['--truth',str(truth),'--series','California','--selection-dir',str(selection),
                '--smoothing','none','--probabilistic','--start-date','2020-02-29','--max-origins','2',
                '--negative-truth','clip','--output-dir',str(prediction/'01')])
            expected = run_experiment(args)
            (prediction/'series_metrics.csv').write_text('series,output_subdir\nCalifornia,01\n')
            atomic_json(prediction/'metadata.json',dict(status='complete'))

            forecast_ns = execute_cells('forecasting.ipynb',dict(RUN_DIR=run,MANIFEST=manifest,
                SELECTED_SERIES='California',FORECAST_MODE='pipeline',NOTEBOOK_OUTPUT=root/'unused'))
            self.assertTrue(forecast_ns['BO_READY'])
            self.assertEqual(len(forecast_ns['issued']),8)
            self.assertEqual(len(forecast_ns['forecast']),4)
            self.assertTrue((forecast_ns['history']['target_end_date']<=forecast_ns['origin']).all())
            self.assertTrue((forecast_ns['forecast']['forecast_origin']==forecast_ns['origin']).all())

            evaluation_ns = execute_cells('evaluation.ipynb',dict(RUN_DIR=run,MANIFEST=manifest,
                PREDICTION_DIR=prediction,SELECTED_SERIES='California',EVALUATION_MODE='recompute',
                NOTEBOOK_EVALUATION_DIR=root/'evaluation',REFERENCE_SCORES=None))
            scores = evaluation_ns['overall_table']
            self.assertAlmostEqual(scores.loc['smoa','mae'],expected['overall']['mae'])
            self.assertAlmostEqual(scores.loc['smoa','wis'],expected['overall']['wis'])
            self.assertEqual(evaluation_ns['summary']['forecasts_per_model'],8)
            self.assertEqual(len(evaluation_ns['calibration_table']),22)
            images = [arg for call in display.call_args_list for arg in call.args
                      if getattr(arg,'format',None)=='png']
            self.assertGreaterEqual(len(images),5)
            self.assertTrue(all(image.data.startswith(b'\x89PNG') for image in images))
            self.assertTrue((root/'evaluation/report.md').is_file())


if __name__ == '__main__':
    unittest.main()
