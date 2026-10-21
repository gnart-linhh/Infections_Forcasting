"""Paired multi-series benchmark using shared synthetic embeddings."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from model import load_library
from run_backtest import parser, run_experiment


def aggregate(rows):
    """Report micro averages and equally weighted series ratios separately."""
    if not rows or len({r['series'] for r in rows}) != len(rows):
        raise ValueError('Need one unique summary per series')
    weights = np.asarray([r['origins'] for r in rows])
    metrics = rows[0]['overall']
    pooled = {m: float(np.average([r['overall'][m] for r in rows], weights=weights)) for m in metrics}
    macro = {}
    for metric in ['mae', 'wis']:
        if metric not in metrics:
            continue
        eligible = [r for r in rows if r['overall']['baseline_'+metric] > 0]
        ratios = [r['overall'][metric]/r['overall']['baseline_'+metric] for r in eligible]
        macro[metric] = dict(mean_series_ratio=float(np.mean(ratios)) if ratios else None,
            median_series_ratio=float(np.median(ratios)) if ratios else None,
            series_better_than_baseline=sum(r < 1 for r in ratios),
            eligible_series=len(eligible), excluded_zero_baseline=len(rows)-len(eligible))
    return dict(series=len(rows), scored_origins=int(weights.sum()), pooled=pooled, relative=macro)


def main():
    p = parser(truth_required=False)
    p.description = __doc__
    p.add_argument('--manifest', type=Path, required=True, help='CSV: series,truth,as_of_truth; relative paths resolve beside manifest')
    p.add_argument('--history-mode', choices=['final', 'as_of'], default='final')
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
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Output directory must be empty')
    embeddings = load_library(a.synthetic, a.k, a.h)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, state_rows, horizon_rows = [], [], []
    output = a.output_dir
    for i, entry in enumerate(entries, 1):
        config = type(a)(**vars(a))
        config.truth = entry['truth']; config.as_of_truth = entry['as_of_truth']
        config.series = entry['series']; config.output_dir = output/f'{i:02d}'
        summary = run_experiment(config, embeddings=embeddings)
        summaries.append(summary)
        metrics = summary['overall']
        row = dict(series=summary['series'], origins=summary['origins'], warmup_origins=summary['warmup_origins'],
                   history_mode=summary['history_mode'], scored_start=summary['scored_start'],
                   scored_end=summary['scored_end'], output_subdir=f'{i:02d}', **metrics)
        for metric in ['mae', 'wis']:
            if metric in metrics:
                denominator = metrics['baseline_'+metric]
                row[metric+'_ratio'] = metrics[metric]/denominator if denominator > 0 else ''
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
                       for name in ['run_benchmark.py','run_backtest.py','model.py','snapshots.py','uncertainty.py']},
        status='complete', notes=['Per-series outputs include input checksums and environment versions',
                                  'Same synthetic library and hyperparameters across series; no tuning on scores',
                                  'Relative ratios use persistence+NB, not COVIDHub ensemble'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
