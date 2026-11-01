import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/forecasting"))
sys.path.insert(0, str(ROOT / "experiments/data"))

import csv
from datetime import date,timedelta
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from model import load_library
from run_backtest import parser,run_experiment
from uncertainty import fit_size,calibration_plan,LONG_QUANTILES,interval_quantiles,coverage_rows
from prediction import predict_origin
from evaluate import evaluate_outputs,evaluate_benchmark
from compare_models import compare
from selection import apply_selection


class ReproductionAuditTests(unittest.TestCase):
    def test_zero_nb_outcomes_and_negative_exclusion(self):
        self.assertTrue(np.isfinite(fit_size([50,1000],[0,1000])))
        prior=[dict(horizon=j,target_end_date='2020-01-01',forecast=1000.,baseline=1000.,true_value=1100.)
               for _ in range(6) for j in range(1,5)]
        prior[0]['true_value']=-100
        plan=calibration_plan(prior,'2020-02-01')
        self.assertEqual(plan[('forecast',1)]['count'],5)
        self.assertEqual(plan[('forecast',1)]['excluded_negative'],1)

    def test_full_source_quantiles_and_11_intervals(self):
        qs=interval_quantiles(100,10,np.random.default_rng(13),probabilities=LONG_QUANTILES)
        self.assertEqual(len(qs),23)
        self.assertEqual(qs[11],100)
        rows=coverage_rows(qs,100,LONG_QUANTILES)
        self.assertEqual(len(rows),11)
        self.assertAlmostEqual(rows[0]['level'],.98)
        self.assertAlmostEqual(rows[-1]['level'],.10)

    def test_prediction_api_has_no_future_truth_inputs(self):
        import inspect
        parameters=set(inspect.signature(predict_origin).parameters)
        self.assertFalse(parameters & {'actual','truth','future','true_value'})
        plan=calibration_plan([], '2020-01-25',2)
        one=predict_origin(np.array([100,101,102]),np.array([[1,1]]),np.array([[1,1]]),date(2020,1,25),plan,
                           k_closest=1,smoothing='none')
        two=predict_origin(np.array([100,101,102]),np.array([[1,1]]),np.array([[1,1]]),date(2020,1,25),plan,
                           k_closest=1,smoothing='none')
        np.testing.assert_array_equal(one[0][0]['quantiles'],two[0][0]['quantiles'])

    def test_raw_scoring_keeps_zero_and_negative_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            truth=root/'truth.csv'
            values=[0,0,0,0,0,0,-3,0,0,0]
            days=[date(2020,1,4)+timedelta(weeks=i) for i in range(len(values))]
            truth.write_text('target_end_date,value\n'+''.join(f'{d},{v}\n' for d,v in zip(days,values)))
            library=root/'curves.csv';library.write_text('curve_id,step,value\n'+''.join(f'a,{i+1},{i+1}\n' for i in range(12)))
            args=parser().parse_args(['--synthetic',str(library),'--truth',str(truth),'--k','2','--h','2',
                '--closest','1','--smoothing','none','--probabilistic','--start-date',str(days[4]),
                '--max-origins','1','--negative-truth','clip','--output-dir',str(root/'prediction')])
            result=run_experiment(args)
            with (root/'prediction/forecasts.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual([float(r['true_value']) for r in rows],[0,-3])
            self.assertTrue(all('baseline_abs_error' not in r for r in rows))
            evaluation=evaluate_outputs(root/'prediction',truth,root/'evaluation',create_plots=False)
            self.assertEqual(evaluation['models']['smoa']['abs_error'],result['overall']['mae'])
            self.assertEqual(evaluation['models']['smoa']['wis'],result['overall']['wis'])
            self.assertEqual(len(evaluation['coverage_by_level']['smoa']),11)
            with (root/'prediction/predictions.csv').open() as f:
                self.assertNotIn('true_value',csv.DictReader(f).fieldnames)

    def test_report_handles_point_only_forecasts_without_inventing_wis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred=root/'prediction';pred.mkdir()
            truth=root/'truth.csv'
            truth.write_text('target_end_date,value\n2020-01-04,100\n2020-01-11,0\n2020-01-18,-3\n2020-01-25,200\n')
            (pred/'predictions.csv').write_text(
                'series,forecast_origin,target_end_date,horizon,forecast,baseline\n'
                'A,2020-01-04,2020-01-11,1,10,100\n'
                'A,2020-01-04,2020-01-18,2,20,100\n'
                'A,2020-01-11,2020-01-18,1,30,0\n'
                'A,2020-01-11,2020-01-25,2,40,0\n')
            result=evaluate_outputs(pred,truth,root/'evaluation')
            self.assertEqual(result['models']['smoa']['mae'],56.5)
            self.assertEqual(set(result['models']),{'smoa'})
            self.assertEqual(result['by_horizon']['1']['smoa']['mae'],21.5)
            self.assertEqual(result['by_horizon']['2']['smoa']['mae'],91.5)
            self.assertNotIn('wis',result['models']['smoa'])
            report=(root/'evaluation/report.md').read_text()
            self.assertIn('| sMOA | 4 | 56.50 | N/A |',report)
            self.assertIn('forecast_vs_actual.png',result['artifacts'])

    def test_comparison_is_paired_not_global_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);header='series,forecast_origin,target_end_date,horizon,model,abs_error,wis\n'
            own=root/'own.csv';ref=root/'ref.csv'
            own.write_text(header+'A,2020-01-01,2020-01-08,1,smoa,2,3\nA,2020-01-08,2020-01-15,1,smoa,1000,2000\n')
            ref.write_text(header+'A,2020-01-01,2020-01-08,1,external,4,6\nA,2020-01-15,2020-01-22,1,external,999,999\n')
            result=compare(own,ref,root/'out')
            self.assertTrue(all(r['paired_targets']==1 and r['ratio']==.5 for r in result))
            self.assertEqual(result[0]['unmatched_reference_targets'],1)

    def test_state_evaluation_pools_mae_and_wis_by_forecast_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);benchmark=root/'prediction';benchmark.mkdir()
            manifest=root/'manifest.csv';manifest.write_text('series,truth,as_of_truth\nA,A.csv,\nB,B.csv,\n')
            (benchmark/'series_metrics.csv').write_text('series,output_subdir\nA,01\nB,02\n')
            for series,subdir,points in [('A','01',[1,3]),('B','02',[10])]:
                (root/f'{series}.csv').write_text('target_end_date,value\n2020-01-04,0\n2020-01-11,0\n2020-01-18,0\n')
                folder=benchmark/subdir;folder.mkdir()
                issued=[];quantiles=[]
                for i,point in enumerate(points):
                    row=dict(series=series,forecast_origin=str(date(2020,1,4)+timedelta(weeks=i)),
                        target_end_date=str(date(2020,1,11)+timedelta(weeks=i)),horizon=1)
                    issued.append(dict(**row,forecast=point,baseline=0))
                    for model,value in [('smoa',point),('persistence',0)]:
                        quantiles.extend(dict(**row,model=model,quantile=q,value=value) for q in LONG_QUANTILES)
                for name,rows in [('predictions.csv',issued),('quantiles.csv',quantiles)]:
                    with (folder/name).open('w',newline='') as handle:
                        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            result=evaluate_benchmark(benchmark,manifest,root/'evaluation',create_plots=False)
            self.assertEqual(result['series'],2)
            self.assertEqual(result['forecasts_per_model'],3)
            self.assertAlmostEqual(result['models']['smoa']['mae'],14/3)
            self.assertAlmostEqual(result['models']['smoa']['wis'],14/3)
            self.assertEqual(result['by_horizon']['1']['smoa']['n'],3)
            with (root/'evaluation/scores.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),3)
            self.assertTrue((root/'evaluation/02/report.md').is_file())

    def test_selection_rejects_modified_library(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'best_library').mkdir()
            (root/'best_params.json').write_text('{"num_curves":10,"k":3,"closest":20}')
            (root/'best_library/curves.csv').write_text('curve_id,step,value\na,1,2\n')
            (root/'config.json').write_text('{"window_filter":"nonconstant_v2"}')
            hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['best_params.json','best_library/curves.csv']}
            (root/'metadata.json').write_text(json.dumps(dict(status='complete',artifacts_sha256=hashes)))
            args=parser().parse_args(['--truth','irrelevant.csv','--selection-dir',str(root)])
            apply_selection(args);self.assertEqual(args.k,3);self.assertEqual(args.closest,20)
            (root/'config.json').write_text('{"smoothing":"spline","window_filter":"nonconstant_v2"}')
            with self.assertRaisesRegex(ValueError,'BO used spline'): apply_selection(args)
            args.smoothing='spline'
            apply_selection(args)
            (root/'config.json').write_text('{"smoothing":"spline"}')
            with self.assertRaisesRegex(ValueError,'legacy window filter'): apply_selection(args)
            (root/'config.json').write_text('{"smoothing":"spline","window_filter":"nonconstant_v2"}')
            (root/'best_library/curves.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Modified selection'): apply_selection(args)


class InputAdapterTests(unittest.TestCase):
    def test_source_asof_labels_and_raw_values_preserved(self):
        from prepare_asof import convert
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'truth_manifest.csv';source=root/'source.csv'
            manifest.write_text('series,truth,as_of_truth\nCalifornia,California.csv,\n')
            source.write_text('location_name,as_of,target_end_date,value\nCalifornia,2020-01-11,2020-01-04,-5\nCalifornia,2020-01-11,2020-01-18,100\n')
            meta=convert(source,manifest,root/'out')
            self.assertEqual(meta['rows_per_series']['California'],1)
            self.assertEqual(meta['future_targets_removed'],1)
            with (root/'out/01_asof.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(float(rows[0]['value']),-5)
            self.assertEqual(rows[0]['as_of'],'2020-01-11')

    def test_reference_dates_match_source_target_horizon_join(self):
        from prepare_reference_scores import convert
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'scores.csv';locations=root/'locations.csv'
            locations.write_text('STATE,NAME\n6,California\n')
            source.write_text('location,horizon,target_end_date,forecast_date,model,abs_error,wis\n06,1,2020-08-22,2020-08-17,ensemble,10,5\nUS,1,2020-08-22,2020-08-17,ensemble,1,1\n')
            convert(source,locations,root/'out.csv')
            with (root/'out.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['series'],'California')
            self.assertEqual(rows[0]['forecast_origin'],'2020-08-15')
            self.assertEqual(rows[0]['submission_date'],'2020-08-17')


class RawSnapshotTests(unittest.TestCase):
    def test_raw_outcomes_stay_separate_from_floored_history(self):
        from snapshots import SnapshotTruth
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'snapshot.csv'
            path.write_text('as_of,target_end_date,value\n2020-01-11,2020-01-04,0\n2020-01-11,2020-01-11,-3\n')
            snapshot=SnapshotTruth(path,'clip')
            dates=[date(2020,1,4),date(2020,1,11)]
            self.assertEqual(snapshot.history(date(2020,1,11),dates).tolist(),[1,1])
            self.assertEqual(snapshot.raw_values(date(2020,1,11),dates).tolist(),[0,-3])


if __name__=='__main__': unittest.main()
