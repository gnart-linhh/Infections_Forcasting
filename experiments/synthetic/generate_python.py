"""Portable, approximate Python synthetic-data experiment (not R-equivalent)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import scipy
from generator import gen_curve

FAMILIES = ('sir_rollercoaster', 'sir_rollercoaster_wiggle', 'seasonal')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-curves', type=int, default=10, help='Attempts per family')
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/synthetic_python')
    args = parser.parse_args()
    if args.num_curves < 1 or args.seed_offset < 0:
        parser.error('num-curves must be positive and seed-offset nonnegative')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; select a new directory')
    failures, counts = [], dict.fromkeys(FAMILIES, 0)
    with (args.output_dir / 'curves.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['curve_id', 'seed', 'curve_type', 'time_cadence', 'scale', 'seasonal', 'step', 'value'])
        for family_idx, family in enumerate(FAMILIES):
            for j in range(args.num_curves):
                curve_id = family_idx * args.num_curves + j + 1
                seed = args.seed_offset + curve_id
                curve = gen_curve(family, np.random.default_rng(seed))
                if curve is None or not len(curve.ts) or not np.all(np.isfinite(curve.ts)) or np.any(curve.ts <= 0):
                    failures.append({'curve_id': curve_id, 'seed': seed, 'curve_type': family})
                    continue
                counts[family] += 1
                writer.writerows((curve_id, seed, family, curve.time_cadence, curve.scale, curve.seasonal, step, float(value))
                                 for step, value in enumerate(curve.ts, 1))
            print(f'{family}: {counts[family]}/{args.num_curves}', flush=True)
    metadata = {'engine': 'python-approximate', 'num_curves_per_family': args.num_curves,
                'seed_offset': args.seed_offset, 'accepted': counts, 'failed': failures,
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                
                'helper_sha256': hashlib.sha256(Path(__file__).with_name('generator.py').read_bytes()).hexdigest(),
                'source': 'https://github.com/lanl/precog/tree/main/smoa'}
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if not sum(counts.values()):
        raise RuntimeError('No valid curves generated')


if __name__ == '__main__':
    main()
