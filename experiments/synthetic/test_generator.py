import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import numpy as np
from generator import monthly_aggregate,trim_tails,beta_observations,gen_curve,sir,GENERATOR_VERSION


class GeneratorSourceTests(unittest.TestCase):
    def test_month_bins_match_r_one_based_indices(self):
        np.testing.assert_array_equal(monthly_aggregate(np.arange(1,9,dtype=float)),[6,22,8])

    def test_tail_removal_matches_inclusive_r_slices(self):
        ts=np.r_[np.zeros(25),np.ones(11),np.zeros(64)]
        result=trim_tails(ts)
        np.testing.assert_array_equal(result,np.r_[np.zeros(9),np.ones(11),np.zeros(9)])

    def test_beta_endpoint_distributions_have_no_epsilon_noise(self):
        obs=beta_observations(np.array([0.,1.,.5]),100,np.random.default_rng(13))
        self.assertEqual(obs[0],0);self.assertEqual(obs[1],1)
        self.assertTrue(0<obs[2]<1)

    def test_seasonal_lengths_use_27_step_tail_and_repeat_full_wave(self):
        weekly={52*n-79 for n in range(5,16)}
        monthly={n//4+1 for n in weekly}
        seen=set()
        # All 500 values remain available for 52-step overlapping repetitions.
        with patch('generator.solve_wave',return_value=np.full(500,.02)):
            for seed in range(12):
                result=gen_curve('seasonal',np.random.default_rng(seed))
                self.assertIsNotNone(result)
                seen.add(result.time_cadence)
                self.assertIn(len(result.ts),weekly if result.time_cadence=='weekly' else monthly)
        self.assertEqual(seen,{'weekly','monthly'})

    def test_sir_preserves_mass_and_incidence_definition(self):
        sol=sir(.4,.1,.99,.01,0,np.arange(1,51))
        np.testing.assert_allclose(sol['S']+sol['I']+sol['R'],1,atol=1e-7)
        np.testing.assert_array_equal(sol['incidence'][1:],-np.diff(sol['S']))

    def test_curves_are_reproducible_and_floor_is_final(self):
        for family in ['sir_rollercoaster','sir_rollercoaster_wiggle','seasonal']:
            a=gen_curve(family,np.random.default_rng(17));b=gen_curve(family,np.random.default_rng(17))
            if a is None:
                self.assertIsNone(b)
            else:
                np.testing.assert_array_equal(a.ts,b.ts)
                self.assertTrue(np.all(np.isfinite(a.ts)) and np.all(a.ts>=1e-10))
        self.assertEqual(GENERATOR_VERSION,'source-structure-v2')

    def test_parallel_generation_and_partial_csv_resume_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);first=root/'serial';second=root/'parallel'
            command=[sys.executable,str(Path(__file__).with_name('generate_python.py')),'--num-curves','2']
            subprocess.run([*command,'--workers','1','--output-dir',str(first)],check=True,capture_output=True,text=True)
            subprocess.run([*command,'--workers','2','--output-dir',str(second)],check=True,capture_output=True,text=True)
            expected=(first/'curves.csv').read_bytes()
            self.assertEqual(expected,(second/'curves.csv').read_bytes())
            metadata=json.loads((first/'metadata.json').read_text())
            self.assertEqual(metadata['attempted_total'],6)
            lines=expected.splitlines(keepends=True)
            first_curve_end=len(lines[0])+sum(len(line) for line in lines[1:] if line.startswith(b'1,'))
            (first/'curves.csv').rename(first/'curves.csv.partial')
            (first/'metadata.json').unlink()
            (first/'checkpoint.json').write_text(json.dumps(dict(config=metadata['config'],next_index=1,
                csv_bytes=first_curve_end,accepted={'sir_rollercoaster':1,'sir_rollercoaster_wiggle':0,'seasonal':0},
                failed=[],total_attempts=6,status='running')))
            subprocess.run([*command,'--workers','1','--output-dir',str(first),'--resume'],check=True,capture_output=True,text=True)
            self.assertEqual(expected,(first/'curves.csv').read_bytes())
            self.assertEqual(json.loads((first/'metadata.json').read_text())['accepted_total'],metadata['accepted_total'])


if __name__=='__main__': unittest.main()
