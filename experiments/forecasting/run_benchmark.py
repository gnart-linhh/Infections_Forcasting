"""Paired multi-series benchmark using shared synthetic embeddings."""
from __future__ import annotations
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from model import load_library
from run_backtest import parser, run_experiment
from selection import apply_selection
from neighbors import ExactNeighbors
from runtime import acquire_run_lock,atomic_json,file_hash


def aggregate(rows):
    """Pool sMOA metrics in proportion to the number of scored origins."""
    if not rows or len({r['series'] for r in rows}) != len(rows):
        raise ValueError('Need one unique summary per series')
    weights = np.asarray([r['origins'] for r in rows])
    metrics = rows[0]['overall']
    pooled = {m: float(np.average([r['overall'][m] for r in rows], weights=weights)) for m in metrics}
    return dict(series=len(rows), scored_origins=int(weights.sum()), pooled=pooled)


def main():
    p = parser(truth_required=False)
    p.description = __doc__
    p.add_argument('--manifest', type=Path, required=True, help='CSV: series,truth,as_of_truth; relative paths resolve beside manifest')
    p.add_argument('--history-mode', choices=['final', 'as_of'], default='final')
    p.add_argument('--resume',action='store_true',help='Continue a run with the same inputs, parameters and code')
    p.add_argument('--embedding-cache',type=Path,help='Exact full embeddings on disk, including curve/window audit lookup')
    a = p.parse_args()
    if a.truth or a.as_of_truth:
        p.error('Supply truth paths through manifest, not --truth or --as-of-truth')
    with a.manifest.open() as handle:
        entries = list(csv.DictReader(handle))
    if not entries or len({r['series'] for r in entries}) != len(entries):
        p.error('Manifest must contain unique, nonempty series')
    for entry in entries:
        if not entry['series'].strip():
            p.error('Series names must be nonempty')
        for field in ['truth', 'as_of_truth']:
            value = entry.get(field, '')
            entry[field] = (a.manifest.parent/value).resolve() if value else None
        if not entry['truth'] or not entry['truth'].is_file():
            p.error(f'Missing truth: {entry["series"]}')
        if a.history_mode == 'as_of' and (not entry['as_of_truth'] or not entry['as_of_truth'].is_file()):
            p.error(f'Missing as-of truth: {entry["series"]}')
        if a.history_mode == 'final' and entry['as_of_truth']:
            p.error('Manifest contains as-of paths; select --history-mode as_of explicitly')
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.resume:
        p.error('Output directory must be empty; use --resume to continue')
    apply_selection(a)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    run_lock=acquire_run_lock(a.output_dir/'run.lock')
    config={k:str(v) if isinstance(v,Path) or hasattr(v,'isoformat') else v for k,v in vars(a).items() if k!='resume'}
    config['input_hashes']={str(path):file_hash(path) for path in [a.synthetic,a.manifest]+[
        entry[field] for entry in entries for field in ['truth','as_of_truth'] if entry[field]]}
    source_names=['run_benchmark.py','run_backtest.py','model.py','smoothing.py','neighbors.py','prediction.py','snapshots.py','uncertainty.py']
    if a.embedding_cache: source_names.append('embedding_cache.py')
    config['source_hashes']={name:file_hash(Path(__file__).with_name(name)) for name in source_names}
    if (a.output_dir/'run_config.json').exists():
        if json.loads((a.output_dir/'run_config.json').read_text())!=config: p.error('Resume config/input/source differs')
    else: atomic_json(a.output_dir/'run_config.json',config)
    if (a.output_dir/'metadata.json').exists() and (a.output_dir/'summary.json').exists():
        print('Benchmark already complete and verified.',flush=True)
        return
    if a.embedding_cache:
        from embedding_cache import load_cached_library
        embeddings=load_cached_library(a.synthetic,a.embedding_cache,a.k,a.h,a.bridge_mode)
    else:
        embeddings = load_library(a.synthetic,a.k,a.h+int(a.bridge_mode=='omit'))
        if a.bridge_mode=='omit':
            x,y,ids,steps=embeddings
            embeddings=(x,y[:,1:],ids,steps)
    neighbor_index=ExactNeighbors(embeddings[0])
    summaries, state_rows, horizon_rows = [], [], []
    output = a.output_dir
    for i, entry in enumerate(entries, 1):
        config = type(a)(**vars(a))
        config.truth = entry['truth']; config.as_of_truth = entry['as_of_truth']
        config.series = entry['series']; config.output_dir = output/f'{i:02d}'
        if a.resume and (config.output_dir/'metadata.json').exists() and (config.output_dir/'summary.json').exists():
            summary=json.loads((config.output_dir/'summary.json').read_text())
        else:
            if a.resume and config.output_dir.exists(): shutil.rmtree(config.output_dir)
            summary = run_experiment(config, embeddings=embeddings,neighbor_index=neighbor_index)
        summaries.append(summary)
        metrics = summary['overall']
        row = dict(series=summary['series'], origins=summary['origins'], warmup_origins=summary['warmup_origins'],
                   history_mode=summary['history_mode'], scored_start=summary['scored_start'],
                   scored_end=summary['scored_end'], output_subdir=f'{i:02d}', **metrics)
        state_rows.append(row)
        horizon_rows.extend(dict(series=summary['series'], horizon=int(h), **stats)
                            for h, stats in summary['by_horizon'].items())
        # Save progress after each completed series; failures never become a partial success.
        with (output/'series_metrics.csv').open('w', newline='') as handle:
            w = csv.DictWriter(handle, fieldnames=list(state_rows[0])); w.writeheader(); w.writerows(state_rows)
        print(f'[{i}/{len(entries)}] {summary["series"]}: {summary["origins"]} scored origins', flush=True)
    with (output/'horizon_metrics.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(horizon_rows[0])); w.writeheader(); w.writerows(horizon_rows)
    report = dict(history_mode=a.history_mode, **aggregate(summaries),
        scored_start=min(r['scored_start'] for r in summaries), scored_end=max(r['scored_end'] for r in summaries),
        common_scored_range=len({(r['scored_start'],r['scored_end'],r['origins']) for r in summaries}) == 1,
        scored_forecasts=sum(r['origins']*a.h for r in summaries),
        warmup_origins=sum(r['warmup_origins'] for r in summaries),
        by_horizon={str(h): aggregate([dict(series=r['series'], origins=r['origins'],
                         overall=r['by_horizon'][str(h)]) for r in summaries]) for h in range(1,a.h+1)})
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    metadata = dict(config={k:str(v) if isinstance(v,Path) or hasattr(v,'isoformat') else v for k,v in vars(a).items()},
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ['run_benchmark.py','run_backtest.py','model.py','smoothing.py','snapshots.py','uncertainty.py']},
        status='complete', notes=['Per-series outputs include input checksums and environment versions',
                                  'Same synthetic library and hyperparameters across series; no tuning on scores',
                                  'Only sMOA is generated and evaluated'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
