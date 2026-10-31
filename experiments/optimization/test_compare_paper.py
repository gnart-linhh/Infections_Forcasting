from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import compare_paper as comparison
from objective import generate_bank
from generator import SyntheticCurve


def fake_curve(family, rng):
    values=100+np.cumsum(rng.uniform(-2,3,20))
    return SyntheticCurve(values,family,'weekly','counts',True)


class PaperComparisonTests(unittest.TestCase):
    def test_bo_cli_runs_comparison_after_selection_and_verifies_on_resume(self):
        import run_optimization
        with tempfile.TemporaryDirectory() as tmp, patch('objective.gen_curve',fake_curve), \
             patch.object(comparison,'PAPER_PARAMETERS',dict(num_curves=2,k=2,closest=1)), \
             patch.dict(comparison.COMPARISON_PROTOCOL,dict(test_attempts_per_family=2,test_curve_cap=6,bootstrap_draws=200)), \
             redirect_stdout(io.StringIO()) as output:
            args=['run_optimization.py','--workers','1','--num-curves-range','1','2','--k-range','2','3',
                  '--closest-range','1','2','--initial-points','2','--iterations','1','--compare-paper',
                  '--output-dir',tmp]
            with patch.object(sys,'argv',args): run_optimization.main()
            out=Path(tmp)
            selected=json.loads((out/'best_params.json').read_text())
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['trials'],3)
            self.assertEqual(summary['paper_comparison']['winner_params'],selected)
            self.assertEqual(json.loads((out/'metadata.json').read_text())['status'],'complete')
            with patch.object(sys,'argv',args+['--resume']): run_optimization.main()
            self.assertIn('BO already complete and verified',output.getvalue())

    def test_paper_prefix_uses_attempt_indices_with_failures(self):
        with patch('objective.gen_curve',fake_curve):
            bank,audit=generate_bank(3,13,0,4)
        kept=[]
        for curve,row in zip(bank,audit):
            row['accepted']=row['index']!=0
            if row['accepted']: kept.append(curve)
        paper,pa=comparison.paper_bank(kept,audit,3,2,13,4,1)
        self.assertEqual(len(paper),3)
        self.assertEqual(len(pa),6)
        self.assertTrue(all(row['index']<2 for row in pa))
        self.assertEqual(sum(row['accepted'] for row in pa),len(paper))

    def test_paper_extension_reproduces_selected_shared_curves(self):
        with patch('objective.gen_curve',fake_curve):
            bank,audit=generate_bank(1,13,0,4)
            paper,pa=comparison.paper_bank(bank,audit,1,2,13,4,1)
        self.assertEqual(len(paper),6)
        for family in range(3): np.testing.assert_array_equal(bank[family].ts,paper[2*family].ts)
        corrupted=[dict(row,seed_key=[99,0,4,0,0]) for row in audit]
        with self.assertRaisesRegex(ValueError,'seed namespace'):
            comparison.paper_bank(bank,corrupted,1,2,13,4,1)

    def test_paired_targets_reject_missing_duplicate_or_changed_labels(self):
        rows=[dict(curve_id=0,family='seasonal',known_steps=6,horizon=h,
                   actual=100.,forecast=110.,abs_error=10.,baseline=90.,baseline_abs_error=10.) for h in [1,2]]
        for other in [rows[:1],rows+[rows[0]],[dict(r,actual=200.) for r in rows]]:
            with self.assertRaises(ValueError): comparison.paired_scores(rows,other)

    def test_bootstrap_resamples_curves_and_weights_forecast_counts(self):
        frame=pd.DataFrame(dict(curve_id=[0,0,1],abs_error_winner=[2.,2.,1.],
            abs_error_paper=[1.,1.,4.],baseline_abs_error_winner=[3.,3.,3.],delta_abs_error=[1.,1.,-3.]))
        result=comparison.summarize_pairs(frame,13)
        self.assertAlmostEqual(result['delta_mae'],-1/3)
        np.testing.assert_array_equal(result['delta_mae_bootstrap_95'],[-3,1])

    def test_fresh_gam_test_exports_identical_targets_and_no_selection_feedback(self):
        protocol=dict(comparison.COMPARISON_PROTOCOL,test_attempts_per_family=2,test_curve_cap=6,bootstrap_draws=200)
        paper=dict(num_curves=2,k=2,closest=1)
        with tempfile.TemporaryDirectory() as tmp, patch('objective.gen_curve',fake_curve), \
             patch.object(comparison,'COMPARISON_PROTOCOL',protocol),patch.object(comparison,'PAPER_PARAMETERS',paper):
            bank,audit=generate_bank(3,13,0,4)
            out=Path(tmp)
            result=comparison.run_comparison(out,dict(num_curves=3,k=3,closest=2),bank,audit,13,4)
            winner=pd.read_csv(out/'winner_forecasts.csv');baseline=pd.read_csv(out/'paper_forecasts.csv')
            pd.testing.assert_frame_equal(winner[comparison.KEYS],baseline[comparison.KEYS])
            self.assertEqual(winner.known_steps.min(),4)
            self.assertEqual(len(winner),result['scored_forecasts'])
            self.assertAlmostEqual(result['winner_mae'],winner.abs_error.mean())
            self.assertAlmostEqual(result['paper_mae'],baseline.abs_error.mean())
            self.assertTrue((out/'report.md').exists())
            self.assertTrue((out/'mae_comparison.png').read_bytes().startswith(b'\x89PNG'))
            import json
            self.assertFalse(json.loads((out/'metadata.json').read_text())['test_used_for_selection'])
            from bank import read_audit
            test_audit=read_audit(out/'test_audit.jsonl.gz')
            self.assertEqual({(row['role'],row['trial']) for row in test_audit},{(2,1)})
        with self.assertRaisesRegex(ValueError,'requires GAM'):
            comparison.run_comparison(Path('/unused'),{},[],[],13,4,smoothing='spline')


if __name__=='__main__': unittest.main()
