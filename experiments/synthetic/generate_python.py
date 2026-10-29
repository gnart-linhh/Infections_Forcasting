"""Generate the source-sized synthetic library in Python, with resumable CSV export."""
from __future__ import annotations
import argparse
import csv
import json
import os
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import numpy as np
import scipy
from generator import GENERATOR_VERSION
from bank import FAMILIES, PUBLISHED_CURVES_PER_FAMILY, generate_ordered
from runtime import atomic_json, file_hash, acquire_run_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-curves', type=int, default=PUBLISHED_CURVES_PER_FAMILY, help='Generation attempts per family; source default 18,387')
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reproduction_full/01_synthetic')
    args = parser.parse_args()
    if args.num_curves < 1 or args.seed_offset < 0 or args.workers < 1:
        parser.error('num-curves/workers must be positive and seed-offset nonnegative')
    out = args.output_dir
    config = dict(num_curves_per_family=args.num_curves, seed_offset=args.seed_offset,
                  generator_version=GENERATOR_VERSION,
                  source_sha256={name: file_hash(Path(__file__).with_name(name))
                                 for name in ['generator.py', 'bank.py', 'generate_python.py']})
    checkpoint_path = out / 'checkpoint.json'
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error('Output directory must be empty; use --resume to continue this run')
    out.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(out / 'run.lock')
    if args.resume and (out / 'metadata.json').exists():
        metadata = json.loads((out / 'metadata.json').read_text())
        if metadata.get('config') != config or metadata.get('curves_sha256') != file_hash(out / 'curves.csv'):
            parser.error('Completed library/config/checksum differs; select a new output directory')
        print('Synthetic library already complete and verified.', flush=True)
        return
    partial = out / 'curves.csv.partial'
    counts = dict.fromkeys(FAMILIES, 0); failures = []; start = 0
    total = len(FAMILIES) * args.num_curves
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if not args.resume or checkpoint['config'] != config:
            parser.error('Checkpoint config/source differs')
        counts, failures, start = checkpoint['accepted'], checkpoint['failed'], checkpoint['next_index']
        if not partial.exists() and start==total and (out/'curves.csv').stat().st_size==checkpoint['csv_bytes']:
            (out/'curves.csv').replace(partial)
        # Bytes written after the last atomic checkpoint are replayed, not duplicated.
        with partial.open('r+b') as handle:
            if handle.seek(0, 2) < checkpoint['csv_bytes']:
                parser.error('Partial CSV shorter than checkpoint')
            handle.truncate(checkpoint['csv_bytes'])
    elif args.resume and any(p.name!='run.lock' for p in out.iterdir()):
        parser.error('Cannot resume without a synthetic checkpoint')
    tasks = [(i // args.num_curves, i % args.num_curves, [args.seed_offset + i + 1])
             for i in range(start, total)]
    with partial.open('a' if start else 'w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        if not start:
            writer.writerow(['curve_id', 'seed', 'curve_type', 'time_cadence', 'scale', 'seasonal', 'step', 'value'])
        def checkpoint(next_index):
            handle.flush()
            os.fsync(handle.fileno())
            atomic_json(checkpoint_path, dict(config=config, next_index=next_index, csv_bytes=handle.tell(),
                        accepted=counts, failed=failures, total_attempts=total, status='running'))
        checkpoint(start)
        for completed, (task, curve) in enumerate(zip(tasks, generate_ordered(tasks, args.workers)), start + 1):
            family_index, _, key = task; family = FAMILIES[family_index]
            if curve is None:
                failures.append(dict(curve_id=completed, seed=key[0], curve_type=family))
            else:
                counts[family] += 1
                writer.writerows((completed, key[0], family, curve.time_cadence, curve.scale, curve.seasonal, step, float(value))
                                 for step, value in enumerate(curve.ts, 1))
            if completed % 500 == 0 or completed == total:
                checkpoint(completed)
                print(f'Synthetic attempts {completed:,}/{total:,}; accepted {sum(counts.values()):,}', flush=True)
    partial.replace(out / 'curves.csv')
    metadata = dict(status='complete', engine='python-source-structure', config=config,
                    generator_version=GENERATOR_VERSION, num_curves_per_family=args.num_curves,
                    attempted_total=total, accepted_total=sum(counts.values()), accepted=counts, failed=failures,
                    workers=args.workers, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    curves_sha256=file_hash(out / 'curves.csv'), source='https://github.com/lanl/precog/tree/main/smoa')
    atomic_json(out / 'metadata.json', metadata)
    checkpoint_path.unlink()
    print(json.dumps(dict(attempted=total, accepted=counts, failed=len(failures)), indent=2), flush=True)


if __name__ == '__main__':
    main()
