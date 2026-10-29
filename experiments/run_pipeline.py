"""Full synthetic -> Bayesian selection -> UQ -> prediction -> evaluation."""
from __future__ import annotations
import argparse
import csv
from datetime import date
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from runtime import acquire_run_lock, atomic_json, file_hash


def run_script(relative, arguments):
    subprocess.run([sys.executable, '-u', str(ROOT / relative), *map(str, arguments)], check=True, cwd=ROOT)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    data = p.add_mutually_exclusive_group()
    data.add_argument('--truth', type=Path, help='Optional single-series run instead of the state manifest')
    data.add_argument('--manifest', type=Path, help='Default: outputs/truth_states/manifest.csv (all 50 states)')
    p.add_argument('--as-of-truth', type=Path, help='Single-series historical archive')
    p.add_argument('--reference-scores', type=Path)
    p.add_argument('--series', default='California')
    p.add_argument('--profile', choices=['paper'], default='paper')
    p.add_argument('--protocol', choices=['source', 'causal'], default='source')
    p.add_argument('--validation-protocol', choices=['resampled', 'fixed'], default='resampled')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument('--synthetic-curves-per-family', type=int, default=18387)
    p.add_argument('--initial-points', type=int)
    p.add_argument('--iterations', type=int)
    p.add_argument('--bo-num-curves-range', type=int, nargs=2)
    p.add_argument('--bo-k-range', type=int, nargs=2)
    p.add_argument('--bo-closest-range', type=int, nargs=2)
    p.add_argument('--bo-max-origins-per-curve', type=int)
    p.add_argument('--start-date', type=date.fromisoformat, default=date(2020, 8, 15))
    p.add_argument('--end-date', type=date.fromisoformat)
    p.add_argument('--warmup-start-date', type=date.fromisoformat)
    p.add_argument('--max-origins', type=int, default=0, help='0: all eligible origins')
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reproduction_full')
    return p


def execute(a, out, entries):
    existing = json.loads((out / 'pipeline.json').read_text()) if (out / 'pipeline.json').exists() else {}
    stages = existing.get('stages', [])
    def checkpoint(stage, **details):
        stages[:] = [s for s in stages if s['stage'] != stage]
        stages.append(dict(stage=stage, **details)); stages.sort(key=lambda s: s['stage'])
        atomic_json(out / 'pipeline.json', dict(status='running', stages=stages))
    synthetic = out / '01_synthetic'
    args = ['--num-curves', a.synthetic_curves_per_family, '--workers', a.workers, '--output-dir', synthetic]
    if a.resume and synthetic.exists(): args.append('--resume')
    run_script('experiments/synthetic/generate_python.py', args)
    checkpoint(1, name='Generate synthetic data', artifact='01_synthetic/curves.csv',
        attempts_per_family=a.synthetic_curves_per_family, total_attempts=3*a.synthetic_curves_per_family,
        role='Full reference library for source-scale generation/EDA. BO independently resamples each candidate; prediction uses the selected training bank.')
    bo = ['--profile', 'paper', '--seed', a.seed, '--workers', a.workers,
          '--validation-protocol', a.validation_protocol, '--output-dir', out / '02_optimization']
    overrides = {'initial-points': a.initial_points, 'iterations': a.iterations,
                 'num-curves-range': a.bo_num_curves_range, 'k-range': a.bo_k_range,
                 'closest-range': a.bo_closest_range, 'max-origins-per-curve': a.bo_max_origins_per_curve}
    for name, value in overrides.items():
        if value is not None: bo += [f'--{name}', *(value if isinstance(value, list) else [value])]
    if a.resume and (out / '02_optimization').exists(): bo.append('--resume')
    run_script('experiments/optimization/run_optimization.py', bo)
    params = json.loads((out / '02_optimization/best_params.json').read_text())
    bo_config = json.loads((out / '02_optimization/config.json').read_text())
    checkpoint(2, name='Bayesian hyperparameter selection', artifact='02_optimization/best_params.json',
        selected_params=params, full_search_budget=bo_config['full_search_budget'], selected_library='02_optimization/best_library/curves.csv')
    sys.path.insert(0, str(ROOT / 'experiments/forecasting'))
    from uncertainty import INITIAL_SIZES, QUANTILES, LONG_QUANTILES
    policy = dict(initial_nb_sizes=list(INITIAL_SIZES), initial_mean='point+1', mle_mean='point',
        mle_inverse_size_bounds=[1e-8, 1], initial_origins=6, standard_quantiles=QUANTILES.tolist(),
        coverage_quantiles=LONG_QUANTILES.tolist(), calibration_truth='final' if a.protocol=='source' else 'history',
        execution='At each origin fit UQ on prior matured outcomes, then issue point forecast and predictive quantiles',
        negative_calibration='Excluded from NB likelihood and explicitly counted; retained for scoring')
    atomic_json(out / '03_uq/policy.json', policy)
    checkpoint(3, name='Uncertainty quantification', artifact='03_uq/policy.json', fitted_parameters='04_prediction/*/calibration.csv')
    prediction_dir = out / '04_prediction'
    prediction = ['--selection-dir', out / '02_optimization', '--start-date', a.start_date,
        '--warmup-start-date', a.warmup_start_date or (date(1900,1,1) if a.protocol=='source' else date(2020,6,6)),
        '--max-origins', a.max_origins, '--probabilistic', '--seed', a.seed, '--score-truth', 'raw',
        '--coverage-grid', 'full', '--negative-truth', 'clip', '--calibration-truth', policy['calibration_truth'],
        '--prehistory-mode', 'frozen' if a.protocol=='source' else 'rolling', '--output-dir', prediction_dir]
    if a.end_date: prediction += ['--end-date', a.end_date]
    if a.truth:
        prediction += ['--truth', a.truth.resolve(), '--series', a.series]
        if a.as_of_truth: prediction += ['--as-of-truth', a.as_of_truth.resolve()]
        if not (a.resume and (prediction_dir/'metadata.json').exists() and (prediction_dir/'summary.json').exists()):
            if a.resume and prediction_dir.exists(): shutil.rmtree(prediction_dir)
            run_script('experiments/forecasting/run_backtest.py', prediction)
    else:
        prediction += ['--manifest', a.manifest.resolve(), '--history-mode', 'as_of' if entries[0].get('as_of_truth') else 'final']
        if a.resume and prediction_dir.exists(): prediction.append('--resume')
        run_script('experiments/forecasting/run_benchmark.py', prediction)
    checkpoint(4, name='Prediction', artifact='04_prediction', series=len(entries),
        note='Issued payloads have no actual values or scores; parameters/library are the observed BO winner.')
    from evaluate import evaluate_outputs, evaluate_benchmark, format_metrics
    if a.truth:
        summary = evaluate_outputs(prediction_dir, a.truth.resolve(), out / '05_evaluation')
    else:
        summary = evaluate_benchmark(prediction_dir, a.manifest.resolve(), out / '05_evaluation')
    if a.reference_scores:
        from compare_models import compare
        compare(out/'05_evaluation/scores.csv', a.reference_scores.resolve(), out/'05_evaluation/reference_comparison')
    checkpoint(5, name='Evaluation', artifact='05_evaluation/report.md')
    manifest = dict(status='complete', publication_reproduction_complete=False, protocol=a.protocol, stages=stages,
        full_search_budget=bo_config['full_search_budget'], selected_params=params, evaluation=summary,
        truth_mode='as_of' if entries[0].get('as_of_truth') else 'final',
        remaining_differences=['Python RNG, SciPy LSODA/GP and thin-plate GCV optimizer may differ numerically from R',
            'Source duplicate first-origin prehistory record is intentionally not replayed',
            'Historical as-of archive and external reference scores must be supplied for their respective comparisons'])
    atomic_json(out / 'pipeline.json', manifest)
    print('\n' + format_metrics(summary), flush=True)
    print(f"Report: {out/'05_evaluation/report.md'}", flush=True)


def main():
    p = parser(); a = p.parse_args(); out = a.output_dir.resolve()
    if a.workers < 1 or a.synthetic_curves_per_family < 1 or a.max_origins < 0: p.error('Invalid workers/curve/origin counts')
    if not a.truth:
        if a.as_of_truth: p.error('--as-of-truth needs --truth; state archives belong in the manifest')
        a.manifest = a.manifest or ROOT / 'outputs/truth_states/manifest.csv'
        with a.manifest.open() as handle: entries = list(csv.DictReader(handle))
        if not entries or len({r['series'] for r in entries}) != len(entries): p.error('Manifest must contain unique nonempty series')
        for entry in entries:
            for key in ['truth', 'as_of_truth']:
                entry[key] = (a.manifest.parent/entry[key]).resolve() if entry.get(key) else None
        if any(bool(r['as_of_truth'])!=bool(entries[0]['as_of_truth']) for r in entries): p.error('Archive paths must be supplied for all series or none')
    else:
        entries = [dict(series=a.series, truth=a.truth.resolve(), as_of_truth=a.as_of_truth.resolve() if a.as_of_truth else None)]
    paths = [r[key] for r in entries for key in ['truth', 'as_of_truth'] if r[key]]
    if a.reference_scores: paths.append(a.reference_scores)
    for path in paths:
        if not path.is_file(): p.error(f'Input missing: {path}')
    config = {key: str(value.resolve()) if isinstance(value, Path) else value.isoformat() if isinstance(value, date) else value
              for key, value in vars(a).items() if key not in ['resume', 'workers']}
    config['input_sha256'] = {str(path.resolve()): file_hash(path) for path in paths}
    if a.manifest: config['input_sha256'][str(a.manifest.resolve())] = file_hash(a.manifest)
    source_paths = [Path(__file__), ROOT/'experiments/runtime.py']
    source_paths += [path for folder in ['synthetic', 'optimization', 'forecasting']
                     for path in (ROOT/'experiments'/folder).glob('*.py') if not path.name.startswith('test_')]
    config['source_sha256'] = {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths}
    if out.exists() and any(out.iterdir()) and not a.resume: p.error('Output directory must be empty; use --resume to continue')
    out.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(out / 'run.lock')
    if (out/'run_config.json').exists():
        if json.loads((out/'run_config.json').read_text()) != config: p.error('Resume config/input/source differs; use a new directory')
    else:
        atomic_json(out/'run_config.json', config)
    if not (out/'pipeline.json').exists():
        atomic_json(out/'pipeline.json',dict(status='running',stages=[]))
    try:
        execute(a, out, entries)
    except BaseException as exc:
        state = json.loads((out/'pipeline.json').read_text()) if (out/'pipeline.json').exists() else dict(stages=[])
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        atomic_json(out/'pipeline.json', state)
        raise


if __name__ == '__main__':
    main()
