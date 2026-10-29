import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/forecasting"))
sys.path.insert(0, str(ROOT / "experiments/data"))

import tempfile
import unittest
from pathlib import Path
import numpy as np
from model import load_library, forecast
from run_backtest import load_truth
from neighbors import ExactNeighbors


class ModelTests(unittest.TestCase):
    def test_windows_do_not_cross_curve_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'curves.csv'
            path.write_text('curve_id,step,value\na,1,1\na,2,3\na,3,6\na,4,10\nb,1,100\nb,2,110\nb,3,130\nb,4,160\n')
            x, y, ids, starts = load_library(path, 2, 1)
            np.testing.assert_array_equal(x, [[2,3], [10,20]])
            np.testing.assert_array_equal(y, [[4], [30]])
            self.assertEqual(ids.tolist(), ['a','b'])
            self.assertEqual(starts.tolist(), [1,1])

    def test_l1_selection_and_reconstruction(self):
        x = np.array([[2,3], [10,20], [2,4]])
        y = np.array([[4,5], [30,40], [6,7]])
        pred, ids, distances = forecast([1,3,6], x, y, 2, 'none')
        np.testing.assert_array_equal(ids, [0,2])
        np.testing.assert_array_equal(distances, [0,1])
        np.testing.assert_array_equal(pred, [11,17])

    def test_truth_rejects_missing_weeks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'truth.csv'
            path.write_text('target_end_date,value\n2020-01-01,1\n2020-01-15,2\n')
            with self.assertRaises(ValueError):
                load_truth(path)

    def test_tree_neighbors_equal_stable_brute_force_including_boundary_ties(self):
        rng=np.random.default_rng(13)
        x=rng.normal(size=(12000,5));x[:100]=0
        queries=np.vstack([np.zeros(5),rng.normal(size=(4,5))])
        index=ExactNeighbors(x,workers=1)
        self.assertIsNotNone(index.tree)
        ids,distances=index.query(queries,12)
        for query,actual,actual_distances in zip(queries,ids,distances):
            brute=np.abs(x-query).sum(axis=1)
            expected=np.argsort(brute,kind='stable')[:12]
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(actual_distances,brute[expected])


if __name__ == '__main__':
    unittest.main()
