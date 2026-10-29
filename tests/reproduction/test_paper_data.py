from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/data'))
from fetch_paper_data import weekly_states


class HistoricalTruthTests(unittest.TestCase):
    def test_county_aggregation_sunday_saturday_and_incomplete_week(self):
        dates=pd.date_range('2020-01-22','2020-02-05')
        # Two counties, each +1/day; one reporting correction (negative daily).
        values=list(range(1,len(dates)+1));values[9]=values[8]-2
        frame=pd.DataFrame([dict(Province_State='California',**dict(zip(dates.strftime('%-m/%-d/%y'),values))),
                            dict(Province_State='California',**dict(zip(dates.strftime('%-m/%-d/%y'),range(1,len(dates)+1))))])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'jhu.csv';frame.to_csv(path,index=False)
            result=weekly_states(path,{'California':'06'})
        self.assertEqual(result.target_end_date.dt.strftime('%Y-%m-%d').tolist(),['2020-01-25','2020-02-01'])
        self.assertEqual(result.value.tolist(),[8.,14.])


if __name__=='__main__': unittest.main()
