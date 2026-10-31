"""Check origin-only features, normalization, sMOA diagnostics and embedding parity."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'forecasting'))
from conditions import features, errors, summarize
from library import embeddings
from model import load_library, forecast
from neighbors import ExactNeighbors
from prepare_panel import calendar


class ConditionsTests(unittest.TestCase):
    def test_origin_features_cannot_see_future(self):
        history=np.array([10,11,12,13,15,15,30,30],dtype=float)
        data=np.r_[history,[5000,0,99999]]
        before=features(data[:8])
        data[8:]=[-100000,10,3]
        self.assertEqual(before,features(data[:8]))
        self.assertEqual(before['phase'],'rising')
        self.assertEqual(before['normalization'],22.5)

    def test_group_boundaries_and_revisions(self):
        for count,group in [(999,'low'),(1000,'medium'),(9999,'medium'),(10000,'high')]:
            self.assertEqual(features([count]*8)['case_level'],group)
        self.assertEqual(features([20]*26)['history_stage'],'early')
        self.assertEqual(features([20]*27)['history_stage'],'later')
        f=features([-20,0,1,1,1,1,1,1])
        self.assertEqual(f['negative_history_weeks'],1)
        self.assertEqual(f['floored_history_weeks'],2)
        self.assertEqual(f['phase'],'stable')
        self.assertEqual(features([20]*4+[5]*4)['phase'],'stable')  # Last four are flat.
        self.assertEqual(features([20]*6+[5]*2)['phase'],'declining')

    def test_signed_error_extreme_uses_origin_scale(self):
        frame=pd.DataFrame(dict(prediction=[10,31,30],actual=[20,10,10],normalization=[10]*3,
                                series=['a','b','c'],forecast_origin=['2020-04-04']*3,model=['x']*3))
        result=errors(frame)
        self.assertEqual(result.extreme.tolist(),[False,True,True])
        self.assertEqual(result.error.tolist(),[-10,21,20])
        self.assertEqual(result.normalized_absolute_error.tolist(),[1,2.1,2])
        self.assertEqual(result.extreme_2.tolist(),[False,True,False])
        summary=summarize(result,[]).iloc[0]
        self.assertAlmostEqual(summary.MAE,17)
        self.assertAlmostEqual(summary.bias,31/3)
        self.assertAlmostEqual(summary.RMSE,np.sqrt((100+441+400)/3))
        self.assertEqual(summary.states,3)
        pd.testing.assert_frame_equal(summarize(result,[]),summarize(result,[]))

    def test_exact_embedding_and_forecast_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'curves.csv'
            with path.open('w') as handle:
                writer=csv.writer(handle)
                writer.writerow(['curve_id','step','value'])
                for curve,values in enumerate([[0]*16,list(range(20)),[10,2,4,3,8,0,1,40,3,20,30,10]],1):
                    writer.writerows((curve,i,v) for i,v in enumerate(values,1))
            expected_x,expected_y,_,_=load_library(path,5,4)
            x,y=embeddings(path,Path(tmp)/'cache')
            np.testing.assert_array_equal(x,expected_x)
            np.testing.assert_array_equal(y,expected_y)
            first=forecast([2,3,4,10,8,9,11,15],expected_x,expected_y,closest=7)
            actual=forecast([2,3,4,10,8,9,11,15],x,y,closest=7,neighbor_index=ExactNeighbors(x))
            for a,b in zip(first,actual): np.testing.assert_array_equal(a,b)
            x2,y2=embeddings(path,Path(tmp)/'cache')
            np.testing.assert_array_equal(x2,x)
            with (Path(tmp)/'cache/x.dat').open('r+b') as h:
                h.write(b'corrupt!')
            with self.assertRaisesRegex(ValueError,'checksum'):
                embeddings(path,Path(tmp)/'cache')

    def test_calendar_is_fixed_and_all_targets_available(self):
        origins=pd.to_datetime(calendar())
        self.assertEqual(len(origins),50)
        self.assertEqual(origins.min(),pd.Timestamp('2020-04-04'))
        self.assertTrue((origins.dayofweek==5).all())
        self.assertLessEqual(origins.max()+pd.Timedelta(weeks=4),pd.Timestamp('2022-10-29'))

    def test_omit_bridge_matches_full_embedding_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'curves.csv'
            with path.open('w') as handle:
                writer=csv.writer(handle);writer.writerow(['curve_id','step','value'])
                writer.writerows((1,i+1,float(i*i+1)) for i in range(20))
            x,y=embeddings(path,Path(tmp)/'cache',horizon=5)
            expected_x,expected_y,_,_=load_library(path,5,5)
            np.testing.assert_array_equal(x,expected_x)
            np.testing.assert_array_equal(y[:,1:],expected_y[:,1:])
            np.testing.assert_array_equal(y[0,1:],[13,15,17,19])
            history=[1,2,5,10,17,26]
            point,_,_=forecast(history,x,y[:,1:],closest=1,smoothing='none')
            np.testing.assert_array_equal(point,26+np.cumsum([13,15,17,19]))

    def test_quantile_examples_use_case_mean_not_worst_horizon(self):
        from report import case_quantiles
        data=pd.DataFrame([dict(series=f'S{i:02}',forecast_origin='2020-04-04',horizon=h,
                                normalized_absolute_error=i+h/10) for i in range(10) for h in range(1,5)])
        result=case_quantiles(data.sample(frac=1,random_state=2))
        self.assertEqual(result.series.tolist(),['S01','S04','S08','S09'])
        np.testing.assert_allclose(result.normalized_absolute_error,[1.25,4.25,8.25,9.25])

    def test_report_exports_observed_metrics_and_figures(self):
        from conditions import RULES
        from report import report
        rows,case_rows,histories=[],[],[]
        for i,(state,level) in enumerate([('A',20),('B',2000),('C',20000)]):
            origin='2020-04-04'
            history=np.arange(1,12)*level
            f=features(history)
            case_rows.append(dict(series=state,forecast_origin=origin,status='ok',**f))
            for d,value in zip(pd.date_range('2020-01-25',origin,freq='7D'),history):
                histories.append(dict(series=state,forecast_origin=origin,target_end_date=d.date().isoformat(),value=value))
            for h in range(1,5):
                rows.append(dict(series=state,forecast_origin=origin,model='Original_sMOA_Python',horizon=h,
                                 target_end_date=(pd.Timestamp(origin)+pd.Timedelta(weeks=h)).date().isoformat(),
                                 prediction=float(history[-1]*(1+h)),actual=float(history[-1]),**f))
        data=errors(pd.DataFrame(rows))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            report(data,pd.DataFrame(case_rows),pd.DataFrame(histories),out,
                   dict(panel_complete=True,library_attempted=55161,library_accepted=54081,rules=RULES))
            self.assertTrue((out/'report.md').exists())
            self.assertEqual(len(list((out/'figures').glob('*.png'))),8)
            summary=pd.read_csv(out/'overall.csv')
            self.assertEqual(summary.n.tolist(),[12])
            self.assertEqual(summary.model.tolist(),['Original_sMOA_Python'])
            self.assertEqual(RULES['models'],['Original_sMOA_Python'])
            for path in out.glob('*.csv'):
                table=pd.read_csv(path)
                if 'model' in table:
                    self.assertEqual(set(table.model),{'Original_sMOA_Python'})
            counts=pd.read_csv(out/'group_counts_phase.csv')
            self.assertEqual(counts.loc[counts.phase=='stable','cases'].iloc[0],0)
            with self.assertRaisesRegex(ValueError,'only sMOA'):
                report(data.assign(model='unexpected_model'),pd.DataFrame(case_rows),pd.DataFrame(histories),out,
                       dict(panel_complete=True,library_attempted=55161,library_accepted=54081,rules=RULES))


if __name__=='__main__': unittest.main()
