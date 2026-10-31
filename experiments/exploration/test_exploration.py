"""Checks for EDA rules, unit separation and matched revision cohorts."""
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from synthetic_eda import curve_metrics, scan_library
from real_world_eda import load_truth, weekly_changes, periods, revision_tables, revision_summary, compare_vintages


class ExplorationTests(unittest.TestCase):
    def test_peak_detection_scale_and_sensitivity(self):
        x=np.array([0,0,3,8,10,8,3,0,0,0,3,8,10,8,3,0,0],dtype=float)
        a=curve_metrics(x); b=curve_metrics(x*1000)
        self.assertEqual(a['epidemic_peaks'],2)
        self.assertEqual(a['epidemic_peaks'],b['epidemic_peaks'])
        self.assertAlmostEqual(a['mean_absolute_change_over_mean'],b['mean_absolute_change_over_mean'])
        self.assertGreaterEqual(a['peaks_prominence_05'],a['peaks_prominence_20'])
        self.assertEqual(curve_metrics(np.ones(10))['epidemic_peaks'],0)
        self.assertEqual(curve_metrics(np.zeros(10))['cv'],0)
        self.assertEqual(curve_metrics(np.arange(10.))['epidemic_peaks'],0)

    def test_stream_counts_all_curves_and_checks_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'curves.csv'
            fields=['curve_id','curve_type','scale','time_cadence','step','value']
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for c in range(5):
                    for step in range(1,16):
                        writer.writerow(dict(curve_id=str(c),curve_type='sir_rollercoaster',scale='counts',time_cadence='weekly',step=step,value=step))
            curves,groups,examples=scan_library(path)
            self.assertEqual(len(curves),5)
            self.assertEqual(groups.curves.sum(),5)
            self.assertEqual(examples.curve_id.nunique(),3)
            self.assertTrue(curves.embedding_windows_keep.eq(6).all())
            self.assertTrue(curves.embedding_windows_omit.eq(5).all())
            path.write_text(path.read_text().replace(',15,15',',16,15',1))
            with self.assertRaisesRegex(ValueError,'steps'):scan_library(path)

    def frame(self,values):
        return pd.DataFrame(dict(series='Example',region='West',population_2019=100000,
            target_end_date=pd.date_range('2020-01-04',periods=len(values),freq='7D'),value=values,cases_per_100k=values))

    def test_changes_zero_negative_and_gap(self):
        frame=self.frame([0,2000,4000,-10,10000])
        frame.loc[4,'target_end_date']+=pd.Timedelta(weeks=1)
        result=weekly_changes(frame)
        self.assertTrue(result.loc[1,'from_zero'])
        self.assertTrue(np.isnan(result.loc[1,'relative_change']))
        self.assertFalse(result.loc[1,'sharp_change'])
        self.assertTrue(result.loc[2,'sharp_change'])
        self.assertFalse(result.loc[3,'sharp_change'])
        self.assertEqual(result.loc[3,'direction'],'negative_revision')
        self.assertTrue(np.isnan(result.loc[4,'change']))

    def test_common_calendar_and_mixed_census_region_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for state,values in [('Alabama',[0,20,30]),('Alaska',[5,10])]:
                self.frame(values)[['target_end_date','value']].to_csv(root/f'{state}.csv',index=False)
            pd.DataFrame([dict(series=s,truth=f'{s}.csv') for s in ['Alabama','Alaska']]).to_csv(root/'manifest.csv',index=False)
            pd.DataFrame(dict(NAME=['Alabama','Alaska','Unused'],REGION=['3','4','X'],POPESTIMATE2019=[100000,200000,1])).to_csv(root/'population.csv',index=False)
            raw,common,census,_=load_truth(root/'manifest.csv',root/'population.csv')
            self.assertEqual(len(raw),5)
            self.assertEqual(len(common),4)
            self.assertEqual(set(census.region),{'South','West'})
            self.assertEqual(common.loc[common.series.eq('Alaska'),'cases_per_100k'].tolist(),[2.5,5])

    def test_low_high_runs_include_boundaries_exclude_negatives(self):
        result=periods(self.frame([0,100,20,-1,10000,11000,12000]))
        counts=result.loc[result.unit.eq('value')]
        self.assertEqual(counts.loc[counts.level.eq('low'),'weeks'].tolist(),[3])
        self.assertEqual(counts.loc[counts.level.eq('high'),'weeks'].tolist(),[3])
        self.assertTrue(counts.sustained.all())

    def test_balanced_lag_uses_identical_targets(self):
        rows=[]
        for target,lags,delta in [('2020-01-04',[0,1,2],10),('2020-01-11',[0],1000)]:
            for lag in lags:rows.append(dict(series='Example',target_end_date=pd.Timestamp(target),lag_weeks=lag,
                asof_value=100,final_value=100+delta,revision=delta))
        pairs=pd.DataFrame(rows)
        latest,all_lags,_,balanced,lag=revision_tables(pairs,max_lag=2)
        self.assertEqual(len(latest),2)
        self.assertEqual(len(balanced),3)
        self.assertTrue(lag.pairs.eq(1).all())
        self.assertTrue(lag.absolute_revision_mean.eq(10).all())
        self.assertEqual(all_lags.iloc[0].absolute_revision_mean,505)

    def test_revision_zero_denominator_and_sign(self):
        frame=pd.DataFrame(dict(series=['Example']*2,target_end_date=pd.to_datetime(['2020-01-04']*2),
            final_value=[0,50],asof_value=[0,100],revision=[0,-50]))
        result=revision_summary(frame)
        self.assertAlmostEqual(result['symmetric_relative_revision_mean'],1/3)
        self.assertEqual(result['signed_revision_mean'],-25)

    def test_vintage_join_rejects_future_and_missing_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame([dict(series='Example',as_of_truth='asof.csv')]).to_csv(root/'manifest.csv',index=False)
            truth=self.frame([10])
            path=root/'asof.csv'
            pd.DataFrame([dict(as_of='2020-01-04',target_end_date='2020-01-04',value=8)]).to_csv(path,index=False)
            self.assertEqual(compare_vintages(root,truth).revision.iloc[0],2)
            pd.DataFrame([dict(as_of='2020-01-04',target_end_date='2020-01-11',value=8)]).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Invalid'):compare_vintages(root,truth)
            pd.DataFrame([dict(as_of='2020-01-11',target_end_date='2020-01-11',value=8)]).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Missing'):compare_vintages(root,truth)


if __name__=='__main__':unittest.main()
