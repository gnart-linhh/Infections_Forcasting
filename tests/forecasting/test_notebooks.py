"""Exercise notebook cells with issued fixture forecasts, including plots and scores."""
from contextlib import redirect_stdout
import argparse
import ast
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
from notebooks import execute_notebook


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
            self.assertEqual(len(evaluation_ns['calibration_table']),11)
            images = [arg for call in display.call_args_list for arg in call.args
                      if getattr(arg,'format',None)=='png']
            self.assertGreaterEqual(len(images),5)
            self.assertTrue(all(image.data.startswith(b'\x89PNG') for image in images))
            self.assertTrue((root/'evaluation/report.md').is_file())


class NotebookRunnerTests(unittest.TestCase):
    def test_experiment_flags_dispatch_without_running_forecasts(self):
        scripts = [('comparison', 'experiment_1'), ('library_characteristics', 'experiment_3'),
                   ('analogue_quality', 'experiment_5'), ('probabilistic', 'experiment_6')]
        for folder, stem in scripts:
            with self.subTest(script=stem):
                path = ROOT/'experiments'/folder/(stem+'.py')
                tree = ast.parse(path.read_text())
                entry = next(n for n in tree.body if isinstance(n, ast.If)
                             and isinstance(n.test, ast.Compare)
                             and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')
                def no_forecast(*args):
                    self.fail('Notebook execution must not start a forecasting run')
                namespace = dict(__name__='__main__', __file__=str(path), __doc__='', ROOT=ROOT,
                    Path=Path, argparse=argparse, SEEDS=[0,1,2],
                    LIBRARIES=['rollercoaster','wiggle','seasonal','mixed'],
                    run=no_forecast, worker=no_forecast, smoa_worker=no_forecast)
                with patch.object(sys, 'argv', [str(path), '--execute-notebook']), patch('notebooks.execute_notebook') as execute:
                    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(path), 'exec'), namespace)
                    execute.assert_called_once_with(path.with_suffix('.ipynb'), working_dir=ROOT, timeout=600)

    def test_uses_current_python_and_saves_completed_cells(self):
        with tempfile.TemporaryDirectory() as tmp, patch('jupyter_client.KernelManager') as manager, patch('nbclient.NotebookClient') as client, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'example.ipynb'
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print(42)')]), path)

            def execute():
                book = client.call_args.args[0]
                book.cells[0].execution_count = 1
                book.cells[0].outputs = [nbformat.v4.new_output('stream', name='stdout', text='42\n')]

            client.return_value.execute.side_effect = execute
            self.assertEqual(execute_notebook(path, working_dir=ROOT, timeout=60), path.resolve())
            self.assertEqual(manager.return_value.kernel_spec.argv[0], sys.executable)
            kwargs = client.call_args.kwargs
            self.assertEqual(kwargs['resources']['metadata']['path'], str(ROOT))
            self.assertEqual(kwargs['timeout'], 60)
            self.assertIn('--HistoryManager.hist_file=:memory:', kwargs['extra_arguments'])
            saved = nbformat.read(path, as_version=4)
            self.assertEqual(saved.cells[0].outputs[0].text, '42\n')

    def test_failed_execution_preserves_existing_notebook(self):
        with tempfile.TemporaryDirectory() as tmp, patch('jupyter_client.KernelManager'), patch('nbclient.NotebookClient') as client:
            path = Path(tmp)/'example.ipynb'
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise RuntimeError()')]), path)
            original = path.read_bytes()
            client.return_value.execute.side_effect = RuntimeError('cell failed')
            with self.assertRaisesRegex(RuntimeError, 'cell failed'):
                execute_notebook(path)
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_path_or_timeout_does_not_start_kernel(self):
        with tempfile.TemporaryDirectory() as tmp, patch('jupyter_client.KernelManager') as manager:
            path = Path(tmp)/'example.ipynb'
            nbformat.write(nbformat.v4.new_notebook(), path)
            with self.assertRaises(ValueError):
                execute_notebook(path, timeout=0)
            with self.assertRaises(ValueError):
                execute_notebook(path.with_suffix('.py'))
            manager.assert_not_called()


if __name__ == '__main__':
    unittest.main()
