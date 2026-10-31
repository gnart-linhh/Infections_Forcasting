"""Rolling-origin sMOA backtest with optional online NB intervals and WIS."""
from __future__ import annotations
import argparse
import csv
import gzip
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
from model import load_library, forecast
from uncertainty import QUANTILES, scores, calibration_plan, coverage_rows
from prediction import predict_origin
from selection import apply_selection
from snapshots import SnapshotTruth, calibration_rows
from neighbors import ExactNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from runtime import file_hash


def load_truth(path, negative_policy="reject", floor_values=True):
    with path.open() as handle:
        rows = [(date.fromisoformat(r['target_end_date']), float(r['value'])) for r in csv.DictReader(handle)]
    rows.sort()
    if not rows or not all(np.isfinite(v) for _, v in rows):
        raise ValueError('Truth must have finite values')
    if negative_policy not in ['reject', 'clip']:
        raise ValueError('Unknown negative truth policy')
    if negative_policy == 'reject' and any(v < 0 for _,v in rows):
        raise ValueError('Negative weekly revisions: use --negative-truth clip to floor at 1 explicitly')
    if any(b[0] - a[0] != timedelta(days=7) for a, b in zip(rows, rows[1:])):
        raise ValueError('Truth must contain one series with unique, consecutive weekly dates')
    return [d for d, _ in rows], (np.maximum([v for _, v in rows], 1) if floor_values else np.asarray([v for _,v in rows],float))


def parser(truth_required=True):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir',type=Path,help='Completed BO run; validates and applies its library/k/closest')
    p.add_argument('--synthetic', type=Path, default=ROOT/'outputs/reproduction_full/01_synthetic/curves.csv')
    p.add_argument('--truth', type=Path, required=truth_required, help='CSV: target_end_date,value; one weekly series')
    p.add_argument('--as-of-truth', type=Path, help='Single-series CSV: as_of,target_end_date,value')
    p.add_argument('--warmup-start-date', type=date.fromisoformat, help='Earlier origins used for calibration only')
    p.add_argument('--score-truth', choices=['raw','floored'], default='raw', help='Raw matches source; floored reproduces older project scores')
    p.add_argument('--calibration-truth', choices=['history','final'], default='history')
    p.add_argument('--uq-policy',choices=['matured','figure2_retrospective'],default='matured',
                   help='Retrospective uses final outcomes of earlier issued forecasts, including unmatured targets')
    p.add_argument('--bridge-mode',choices=['keep','omit'],default='keep',
                   help='Omit uses diff(future levels), requiring one additional future level per synthetic window')
    p.add_argument('--prehistory-mode', choices=['rolling','frozen'], default='rolling')
    p.add_argument('--frozen-as-of-date', type=date.fromisoformat, default=date(2020,8,15))
    p.add_argument('--coverage-grid', choices=['full','standard'], default='full')
    p.add_argument('--series', default='California')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--h', type=int, default=4)
    p.add_argument('--closest', type=int, default=4422)
    p.add_argument('--negative-truth', choices=['reject','clip'], default='reject')
    p.add_argument('--end-date', type=date.fromisoformat, help='Last scored origin, inclusive')
    p.add_argument('--start-date', type=date.fromisoformat, default=date(2020, 8, 15))
    p.add_argument('--max-origins', type=int, default=0, help='0 means all eligible origins')
    p.add_argument('--smoothing', choices=['gam', 'spline', 'none'], default='gam')
    p.add_argument('--probabilistic', action='store_true', help='Online NB intervals and WIS, horizons up to 4')
    p.add_argument('--interval-draws', type=int, default=5000)
    p.add_argument('--compress-matches',action='store_true',help='Lossless gzip for the full neighbor audit')
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/forecasting')
    return p


def run_experiment(a, embeddings=None, neighbor_index=None):
    p = parser()
    apply_selection(a)
    if min(a.k, a.h, a.closest) < 1 or a.max_origins < 0:
        p.error('k, h, closest must be positive; max-origins must be nonnegative')
    if a.probabilistic and (a.h > 4 or a.interval_draws < 1 or a.seed < 0):
        p.error('Probabilistic mode requires h <= 4, positive draws and nonnegative seed')
    if a.warmup_start_date and a.warmup_start_date > a.start_date:
        p.error('warmup-start-date must be <= start-date')
    if a.uq_policy=='figure2_retrospective' and a.calibration_truth!='final':
        p.error('Retrospective UQ requires --calibration-truth final so unmatured outcomes are available')
    if embeddings is None:
        x,y,curve_ids,steps=load_library(a.synthetic,a.k,a.h+int(a.bridge_mode=='omit'))
        if a.bridge_mode=='omit': y=y[:,1:]
    else:
        x,y,curve_ids,steps=embeddings
    if x.shape[1]!=a.k or y.shape[1]!=a.h:
        p.error('Embedding dimensions differ from the requested k/h')
    neighbor_index=neighbor_index if neighbor_index is not None else ExactNeighbors(x)
    if a.end_date and a.end_date < a.start_date:
        p.error('end-date must be >= start-date')
    dates, truth = load_truth(a.truth, a.negative_truth, floor_values=a.score_truth=="floored")
    eligible = [i for i in range(max(a.k, 4 if a.smoothing in ['gam','spline'] else a.k), len(truth)-a.h) if dates[i] >= a.start_date and (not a.end_date or dates[i] <= a.end_date)]
    origins = eligible[:a.max_origins] if a.max_origins else eligible
    if not origins:
        p.error('No eligible forecast origins')
    snapshots = SnapshotTruth(a.as_of_truth,a.negative_truth) if a.as_of_truth else None
    warmup = [i for i in range(max(a.k, 4 if a.smoothing in ['gam','spline'] else a.k), origins[0])
              if a.warmup_start_date and a.warmup_start_date <= dates[i] < a.start_date]
    frozen_history=None;frozen_index=None
    if snapshots:
        for i in origins:
            snapshots.history(dates[i], dates[:i+1])
        if a.prehistory_mode=='frozen' and warmup:
            if a.frozen_as_of_date not in dates: p.error('Frozen cutoff not in truth calendar')
            frozen_index=dates.index(a.frozen_as_of_date)
            frozen_history=snapshots.history(a.frozen_as_of_date,dates[:frozen_index+1])
            for i in warmup:
                if i>frozen_index: snapshots.history(dates[i],dates[:i+1])
        else:
            for i in warmup: snapshots.history(dates[i],dates[:i+1])
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Output directory must be empty')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records, quantile_rows, issued = [], [], []
    coverage_table, calibration_table, predictions = [], [], []
    match_fields=['forecast_origin','rank','curve_id','window_start','distance']
    match_path=a.output_dir/('matches.csv.gz' if a.compress_matches else 'matches.csv')
    def match_handle(mode):
        return (gzip.open(match_path,mode+'t',newline='',compresslevel=1) if a.compress_matches else
                match_path.open(mode,newline=''))
    with match_handle('w') as handle:
        csv.DictWriter(handle,fieldnames=match_fields).writeheader()
    for i in warmup + origins:
        history = (frozen_history[:i+1] if frozen_history is not None and i in warmup and i<=frozen_index else
                   snapshots.history(dates[i], dates[:i+1]) if snapshots else np.maximum(truth[:i+1],1))
        vintage=(a.frozen_as_of_date if frozen_history is not None and i in warmup and i<=frozen_index else dates[i])
        observed=snapshots.raw_values(vintage,dates[:i+1]) if snapshots else truth[:i+1]
        prior = (issued if a.calibration_truth=='final' else calibration_rows(issued, dates[i], dates[:i+1], observed))
        plan = calibration_plan(prior,dates[i].isoformat(),a.h,policy=a.uq_policy) if a.probabilistic and i in origins else None
        output,indices,distances,probabilities = predict_origin(history,x,y,dates[i],plan,a.closest,a.smoothing,
            a.seed,a.interval_draws,a.coverage_grid=='full',clip_smoothed=not(i in warmup and a.prehistory_mode=='frozen'),
            neighbor_index=neighbor_index)
        current=[]
        for j in range(1,a.h+1):
            per_horizon=[r for r in output if r['horizon']==j]
            row=dict(series=a.series,forecast_origin=dates[i].isoformat(),target_end_date=dates[i+j].isoformat(),
                horizon=j,forecast=per_horizon[0]['point'],true_value=float(truth[i+j]))
            # Evaluation begins only after the prediction API has returned.
            row.update(abs_error=abs(row['forecast']-row['true_value']))
            if i in origins:
                predictions.append(dict(**{key:row[key] for key in ['series','forecast_origin','target_end_date','horizon','forecast']},
                    forecast_date=(dates[i]+timedelta(days=2)).isoformat()))
            if plan is not None:
                for result in per_horizon:
                    prefix=''
                    standard=np.array([result['quantiles'][np.flatnonzero(np.isclose(probabilities,q))[0]] for q in QUANTILES])
                    row.update({prefix+key:value for key,value in scores(standard,row['true_value']).items()})
                    row.update({prefix+'nb_size':result['size'],prefix+'dispersion_method':result['method'],
                                prefix+'calibration_n':result['count'],prefix+'negative_calibration_excluded':result['excluded_negative']})
                    model='smoa'
                    calibration_table.append(dict(series=a.series,forecast_origin=dates[i].isoformat(),horizon=j,model=model,
                        nb_size=result['size'],method=result['method'],calibration_n=result['count'],excluded_negative=result['excluded_negative'],
                        calibration_policy=result['calibration_policy'],future_calibration_n=result['future_count']))
                    quantile_rows.extend(dict(series=a.series,forecast_origin=dates[i].isoformat(),target_end_date=dates[i+j].isoformat(),
                        horizon=j,model=model,quantile=float(q),value=float(v)) for q,v in zip(probabilities,result['quantiles']))
                    coverage_table.extend(dict(series=a.series,forecast_origin=dates[i].isoformat(),target_end_date=dates[i+j].isoformat(),
                        horizon=j,model=model,**interval) for interval in coverage_rows(result['quantiles'],row['true_value'],probabilities))
            current.append(row)
        issued.extend(current)
        if i not in origins:
            continue
        records.extend(current)
        with match_handle('a') as handle:
            writer=csv.DictWriter(handle,fieldnames=match_fields)
            writer.writerows(dict(forecast_origin=dates[i].isoformat(), rank=rank,
                curve_id=curve_ids[idx], window_start=int(steps[idx]), distance=float(distance))
                for rank,(idx,distance) in enumerate(zip(indices,distances),1))
    tables = [('forecasts.csv', records), ('predictions.csv',predictions)]
    if warmup:
        tables.append(('warmup_forecasts.csv', [r for r in issued if r['forecast_origin'] < a.start_date.isoformat()]))
    if a.probabilistic:
        tables.extend([('quantiles.csv',quantile_rows),('coverage.csv',coverage_table),('calibration.csv',calibration_table)])
    for name, rows in tables:
        with (a.output_dir/name).open('w', newline='') as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def stats(rows):
        result = dict(mae=float(np.mean([r['abs_error'] for r in rows])))
        if a.probabilistic:
            for prefix in ['']:
                for metric in ['wis', 'coverage_50', 'coverage_80', 'coverage_95', 'width_95', 'sharpness', 'overprediction', 'underprediction']:
                    result[prefix+metric] = float(np.mean([r[prefix+metric] for r in rows]))
        return result
    summary = dict(series=a.series, origins=len(origins), warmup_origins=len(warmup),
                   history_mode='as_of' if snapshots else 'final', embedding_rows=len(x),
                   scored_start=dates[origins[0]].isoformat(), scored_end=dates[origins[-1]].isoformat(),
                   closest_used=min(a.closest, len(x)), overall=stats(records),
                   by_horizon={str(j): stats([r for r in records if r['horizon']==j]) for j in range(1, a.h+1)})
    metadata = dict(config={k: str(v) if isinstance(v, (Path, date)) else v for k,v in vars(a).items()},
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        sha256={str(path): file_hash(path) for path in
                [a.synthetic, a.truth, Path(__file__), Path(__file__).with_name('model.py'),
                 Path(__file__).with_name('uncertainty.py'), Path(__file__).with_name('snapshots.py'),
                 Path(__file__).with_name('prediction.py'),Path(__file__).with_name('smoothing.py')]
                + ([a.as_of_truth] if a.as_of_truth else [])},
        limitations=[('Exact as-of snapshots for history; calibration/scoring conventions explicitly recorded in config' if snapshots
                      else 'Final truth backtest, not historical as-of data'),
                     ('Python thin plate regression spline/GCV; mgcv optimizer and floating point may differ' if a.smoothing=='gam' else 'Legacy SciPy spline or no smoothing, differs from source GAM'),
                     'Frozen prehistory must be selected explicitly; duplicate first-origin row in upstream is not replayed',
                     'Monte Carlo quantiles use NumPy RNG, not R RNG',
                     ('Retrospective UQ uses final outcomes of earlier-issued forecasts even before targets mature; not real-time causal'
                      if a.uq_policy=='figure2_retrospective' else 'UQ excludes targets after the current origin'),
                     f'Embedding bridge mode: {a.bridge_mode}'])
    summary['score_truth']=a.score_truth
    summary['calibration_truth']=a.calibration_truth
    summary['coverage_grid']=a.coverage_grid
    summary['bridge_mode']=a.bridge_mode
    summary['uq_policy']=a.uq_policy
    for name, obj in [('summary.json', summary), ('metadata.json', metadata)]:
        (a.output_dir/name).write_text(json.dumps(obj, indent=2)+'\n')
    return summary


def main():
    print(json.dumps(run_experiment(parser().parse_args()), indent=2))


if __name__ == '__main__':
    main()
