"""Full source-budget Bayesian selection: 99 LHS + 5 EI, all validation origins."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import scipy
from bayes import NAMES, minimize_objective
from objective import ROOT, SyntheticObjective, generate_bank, embeddings, evaluate, save_library
from bank import write_audit
from runtime import atomic_json, file_hash, acquire_run_lock
from compare_paper import COMPARISON_PROTOCOL, run_comparison

PAPER = dict(bounds=[[10000, 30000], [2, 7], [1000, 6000]], initial_points=99, iterations=5, max_origins=0)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['paper'], default='paper')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--initial-points', type=int)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--max-origins-per-curve', type=int, help='0 evaluates every upstream eligible origin')
    parser.add_argument('--test-curves-per-family', type=int, help='Default: floor(selected num_curves/1000), capped at 50 curves total')
    parser.add_argument('--num-curves-range', type=int, nargs=2)
    parser.add_argument('--k-range', type=int, nargs=2)
    parser.add_argument('--closest-range', type=int, nargs=2)
    parser.add_argument('--embedding-mode', choices=['upstream', 'pipeline'], default='upstream')
    parser.add_argument('--validation-protocol', choices=['resampled', 'fixed'], default='resampled')
    parser.add_argument('--validation-curves-per-family', type=int, default=3, help='Only used for fixed validation')
    parser.add_argument('--smoothing', choices=['gam', 'spline', 'none'], default='gam')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--compare-paper', action='store_true',
                        help='After selection, compare winner and published params on one fresh paired GAM test')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reproduction_full/02_optimization')
    args = parser.parse_args()
    bounds = [args.num_curves_range or PAPER['bounds'][0], args.k_range or PAPER['bounds'][1],
              args.closest_range or PAPER['bounds'][2]]
    initial = args.initial_points if args.initial_points is not None else PAPER['initial_points']
    iterations = args.iterations if args.iterations is not None else PAPER['iterations']
    max_origins = args.max_origins_per_curve if args.max_origins_per_curve is not None else 0
    if args.seed < 0 or max_origins < 0 or args.workers < 1 or args.validation_curves_per_family < 1 or (args.test_curves_per_family is not None and args.test_curves_per_family < 1):
        parser.error('Invalid seed, workers, origins or test/validation counts')
    if any(lo < 1 or hi <= lo for lo, hi in bounds) or initial < 2 or iterations < 0:
        parser.error('Positive nonconstant bounds, >=2 initial points, >=0 iterations required')
    if args.compare_paper and (args.smoothing != 'gam' or args.embedding_mode != 'upstream'):
        parser.error('--compare-paper requires --smoothing gam and --embedding-mode upstream')
    full_budget = bounds == PAPER['bounds'] and initial == 99 and iterations == 5 and max_origins == 0
    source_paths = [Path(__file__), Path(__file__).with_name('bayes.py'), Path(__file__).with_name('objective.py'),
                    ROOT / 'experiments/runtime.py', ROOT / 'experiments/synthetic/generator.py',
                    ROOT / 'experiments/synthetic/bank.py', ROOT / 'experiments/forecasting/neighbors.py',
                    ROOT / 'experiments/forecasting/smoothing.py', Path(__file__).with_name('compare_paper.py')]
    config = dict(profile='paper' if full_budget else 'custom', full_search_budget=full_budget,
        bounds=dict(zip(NAMES, bounds)), initial_points=initial, iterations=iterations,
        max_origins_per_curve=max_origins, test_curves_per_family=args.test_curves_per_family,
        embedding_mode=args.embedding_mode, smoothing=args.smoothing, seed=args.seed, h=4,
        validation_protocol=args.validation_protocol, validation_curves_per_family=args.validation_curves_per_family,
        min_history_for_fixed_validation=bounds[1][1]+1 if args.validation_protocol=='fixed' else None,
        objective='mean absolute error pooled over synthetic validation forecasts',
        train_validation_protocol='Independent resampling per trial; floor(num_curves/1000) validation attempts/family; shuffle then cap at 50'
            if args.validation_protocol=='resampled' else 'Fixed validation and nested common-seed training banks; fixed origin grid',
        heldout_protocol='Independent role 2 generated only after selecting the best observed validation score',
        paper_comparison=dict(COMPARISON_PROTOCOL) if args.compare_paper else None,
        source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in source_paths})
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error('Output directory must be empty; use --resume to continue')
    out.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(out / 'run.lock')
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text()) != config:
            parser.error('Cannot resume: configuration/source checksums differ')
    elif args.resume and any(p.name!='run.lock' for p in out.iterdir()):
        parser.error('Cannot resume without config.json')
    else:
        atomic_json(out / 'config.json', config)
    if (out / 'metadata.json').exists() and json.loads((out / 'metadata.json').read_text()).get('status') == 'complete':
        metadata = json.loads((out / 'metadata.json').read_text())
        for name, expected in metadata['artifacts_sha256'].items():
            if file_hash(out / name) != expected: parser.error(f'Completed artifact modified: {name}')
        print('BO already complete and verified.', flush=True)
        run_lock.close()
        return
    if not (out / 'run_info.json').exists():
        atomic_json(out / 'run_info.json', dict(started_utc=datetime.now(timezone.utc).isoformat(), started_epoch=time.time()))
    atomic_json(out / 'metadata.json', dict(status='running', full_search_budget=full_budget, workers=args.workers))
    state = json.loads((out / 'checkpoint.json').read_text()) if (out / 'checkpoint.json').exists() else None
    if state:
        write_csv(out / 'trials.csv', state['history'])
        atomic_json(out / 'trials.json', state['history'])
    objective = SyntheticObjective(args.seed, max_origins, args.smoothing, args.embedding_mode,
        args.validation_protocol, args.validation_curves_per_family, config['min_history_for_fixed_validation'],
        workers=args.workers, work_dir=out)
    def log_trial(row, history):
        write_csv(out / 'trials.csv', history)
        atomic_json(out / 'trials.json', history)
        atomic_json(out / 'progress.json', dict(status='running', phase='trial_complete',
            completed_trials=len(history), total_trials=initial+iterations, last_trial=row))
        print(f"[{len(history)}/{initial+iterations}] {row['phase']} num_curves={row['num_curves']} "
              f"k={row['k']} closest={row['closest']} MAE={row['validation_mae']:.4f}", flush=True)
    print(f"BO: {initial} LHS + {iterations} EI; bounds={bounds}; validation origins={'all' if max_origins==0 else max_origins}; workers={args.workers}", flush=True)
    try:
        best, history = minimize_objective(objective, bounds, initial, iterations, args.seed, log_trial,
            resume_state=state, on_checkpoint=lambda checkpoint: atomic_json(out / 'checkpoint.json', checkpoint))
    except BaseException as exc:
        atomic_json(out / 'metadata.json', dict(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc)))
        raise
    atomic_json(out / 'best_params.json', best)
    best_trial = min(history, key=lambda row: row['validation_mae'])
    objective.progress(best_trial['trial'], 'export_best_library')
    library_path = out / 'best_library/curves.csv'
    save_library(objective.best_bank, library_path.with_suffix('.partial'), objective.best_audit)
    library_path.with_suffix('.partial').replace(library_path)
    atomic_json(out / 'best_library/metadata.json', dict(engine='python-source-structure',
        num_curves_per_family=best['num_curves'], attempted_curves=3*best['num_curves'],
        accepted_curves=len(objective.best_bank), seed=args.seed, seed_namespace='[root,role=0,trial,family,index]',
        best_trial=best_trial['trial'], generation_trial=best_trial['trial'] if args.validation_protocol=='resampled' else 0))
    test_n = args.test_curves_per_family or max(1, best['num_curves']//1000)
    test, test_audit = generate_bank(test_n, args.seed, 2, 0, workers=args.workers)
    write_audit(out / 'audit/heldout.jsonl.gz', test_audit)
    test_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 4, 0]))
    test = [test[i] for i in test_rng.permutation(len(test))][:50]
    x, y = embeddings(objective.best_bank, best['k'], mode=args.embedding_mode)
    test_mae, records = evaluate(test, x, y, best['k'], best['closest'], max_origins, args.smoothing,
        min_history=config['min_history_for_fixed_validation'], workers=args.workers)
    write_csv(out / 'heldout_forecasts.csv', records)
    summary = dict(best_params=best, best_trial=best_trial['trial'], best_validation_mae=best_trial['validation_mae'],
        best_initial_lhs_mae=min(row['validation_mae'] for row in history if row['phase']=='lhs'),
        full_search_budget=full_budget, trials=len(history), ei_trials=sum(row['phase']=='ei' for row in history),
        heldout_test=dict(curves=len(test), scored_forecasts=len(records), mae=test_mae,
            persistence_mae=float(np.mean([r['baseline_abs_error'] for r in records])),
            by_family={family: dict(mae=float(np.mean([r['abs_error'] for r in records if r['family']==family])),
                scored_forecasts=sum(r['family']==family for r in records)) for family in sorted({r['family'] for r in records})}))
    atomic_json(out / 'summary.json', summary)
    audit_summaries = [json.loads(path.read_text()) for path in sorted((out / 'audit').glob('trial_*_summary.json'))]
    audit_summaries.append(dict(role=2, attempted=len(test_audit), accepted=sum(r['accepted'] for r in test_audit), audit='audit/heldout.jsonl.gz'))
    atomic_json(out / 'generation_audit.json', audit_summaries)
    del x, y
    if args.compare_paper:
        print('BO selection complete. Running fresh paired synthetic test versus published parameters...', flush=True)
        try:
            comparison = run_comparison(out/'paper_comparison', best, objective.best_bank, objective.best_audit,
                args.seed, best_trial['trial'] if args.validation_protocol=='resampled' else 0,
                workers=args.workers, mode=args.embedding_mode, smoothing=args.smoothing,
                progress=lambda phase,**details: objective.progress(best_trial['trial'], 'paper_comparison_'+phase, **details))
        except BaseException as exc:
            atomic_json(out/'metadata.json', dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                phase='paper_comparison', error=str(exc), completed_trials=len(history)))
            atomic_json(out/'progress.json', dict(status='failed',phase='paper_comparison',
                error=str(exc),completed_trials=len(history),total_trials=initial+iterations))
            raise
        summary['paper_comparison'] = comparison
        atomic_json(out/'summary.json', summary)
    metadata = dict(status='complete', full_search_budget=full_budget, workers=args.workers,
        artifacts_sha256={name: file_hash(out / name) for name in ['best_params.json', 'best_library/curves.csv']+
                          (['paper_comparison/summary.json','paper_comparison/metadata.json'] if args.compare_paper else [])},
        elapsed_seconds=time.time()-json.loads((out/'run_info.json').read_text())['started_epoch'],
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, source_sha256=config['source_sha256'],
        upstream=dict(repository='https://github.com/lanl/precog/tree/main/smoa', revision='aba1948d228e5380d72907048236601874b53cac'),
        limitations=['Python RNG, SciPy LSODA/GAM optimizer/GP differ numerically from R',
            'Resampled validation changes the holdout samples across trials' if args.validation_protocol=='resampled' else 'Fixed validation differs from upstream',
            'Best observed configuration is retained; finite-budget BO does not prove a global optimum',
            'Initial LHS objective evaluations run sequentially; curve generation uses independent processes'])
    atomic_json(out / 'metadata.json', metadata)
    atomic_json(out / 'progress.json', dict(status='complete', completed_trials=len(history), total_trials=initial+iterations))
    # Only transient caches are removed; selected bank, detailed audit and RNG checkpoint remain.
    shutil.rmtree(out / 'working', ignore_errors=True)
    print(json.dumps(summary, indent=2), flush=True)
    run_lock.close()


if __name__ == '__main__':
    main()
