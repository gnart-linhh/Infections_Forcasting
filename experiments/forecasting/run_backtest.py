"""Rolling-origin sMOA backtest with optional online NB intervals and WIS."""
from __future__ import annotations
import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from model import load_library, forecast
from uncertainty import QUANTILES, dispersion, interval_quantiles, scores

ROOT = Path(__file__).resolve().parents[2]


def load_truth(path):
    with path.open() as handle:
        rows = [(date.fromisoformat(r['target_end_date']), float(r['value'])) for r in csv.DictReader(handle)]
    rows.sort()
    if not rows or not all(np.isfinite(v) and v >= 0 for _, v in rows):
        raise ValueError('Truth must have finite nonnegative values')
    if any(b[0] - a[0] != timedelta(days=7) for a, b in zip(rows, rows[1:])):
        raise ValueError('Truth must contain one series with unique, consecutive weekly dates')
    return [d for d, _ in rows], np.maximum([v for _, v in rows], 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', type=Path, default=ROOT/'outputs/synthetic_demo/curves.csv')
    p.add_argument('--truth', type=Path, required=True, help='CSV: target_end_date,value; one weekly series')
    p.add_argument('--series', default='California')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--h', type=int, default=4)
    p.add_argument('--closest', type=int, default=50)
    p.add_argument('--start-date', type=date.fromisoformat, default=date(2020, 8, 15))
    p.add_argument('--max-origins', type=int, default=8, help='0 means all eligible origins')
    p.add_argument('--smoothing', choices=['spline', 'none'], default='spline')
    p.add_argument('--probabilistic', action='store_true', help='Online NB intervals and WIS, horizons up to 4')
    p.add_argument('--interval-draws', type=int, default=5000)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/backtest_demo')
    a = p.parse_args()
    if min(a.k, a.h, a.closest) < 1 or a.max_origins < 0:
        p.error('k, h, closest must be positive; max-origins must be nonnegative')
    if a.probabilistic and (a.h > 4 or a.interval_draws < 1 or a.seed < 0):
        p.error('Probabilistic mode requires h <= 4, positive draws and nonnegative seed')
    x, y, curve_ids, steps = load_library(a.synthetic, a.k, a.h)
    dates, truth = load_truth(a.truth)
    origins = [i for i in range(max(a.k, 4 if a.smoothing == 'spline' else a.k), len(truth)-a.h) if dates[i] >= a.start_date]
    if a.max_origins:
        origins = origins[:a.max_origins]
    if not origins:
        p.error('No eligible forecast origins')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Output directory must be empty')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records, matches, quantile_rows = [], [], []
    for i in origins:
        pred, indices, distances = forecast(truth[:i+1], x, y, a.closest, a.smoothing)
        current = []
        for j, value in enumerate(pred, 1):
            actual = truth[i+j]
            row = dict(series=a.series, forecast_origin=dates[i].isoformat(),
                target_end_date=dates[i+j].isoformat(), horizon=j, forecast=float(value),
                true_value=float(actual), baseline=float(truth[i]),
                abs_error=float(abs(value-actual)), baseline_abs_error=float(abs(truth[i]-actual)))
            if a.probabilistic:
                for model_idx, field in enumerate(['forecast', 'baseline']):
                    size, method, calibration_n = dispersion(records, dates[i].isoformat(), j, field, a.h)
                    rng = np.random.default_rng(np.random.SeedSequence([a.seed, dates[i].toordinal(), j, model_idx]))
                    qs = interval_quantiles(row[field], size, rng, a.interval_draws, method == 'initial')
                    prefix = '' if field == 'forecast' else 'baseline_'
                    row.update({prefix+key: val for key, val in scores(qs, actual).items()})
                    row.update({prefix+'nb_size': size, prefix+'dispersion_method': method, prefix+'calibration_n': calibration_n})
                    quantile_rows.extend(dict(series=a.series, forecast_origin=dates[i].isoformat(),
                        target_end_date=dates[i+j].isoformat(), horizon=j, model='smoa' if model_idx == 0 else 'persistence',
                        quantile=float(q), value=float(v)) for q, v in zip(QUANTILES, qs))
            current.append(row)
        records.extend(current)
        matches.extend(dict(forecast_origin=dates[i].isoformat(), rank=rank,
            curve_id=curve_ids[idx], window_start=int(steps[idx]), distance=float(distance))
            for rank, (idx, distance) in enumerate(zip(indices, distances), 1))
    tables = [('forecasts.csv', records), ('matches.csv', matches)]
    if a.probabilistic:
        tables.append(('quantiles.csv', quantile_rows))
    for name, rows in tables:
        with (a.output_dir/name).open('w', newline='') as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def stats(rows):
        result = dict(mae=float(np.mean([r['abs_error'] for r in rows])),
                      baseline_mae=float(np.mean([r['baseline_abs_error'] for r in rows])))
        if a.probabilistic:
            for prefix in ['', 'baseline_']:
                for metric in ['wis', 'coverage_50', 'coverage_80', 'coverage_95', 'width_95', 'sharpness', 'overprediction', 'underprediction']:
                    result[prefix+metric] = float(np.mean([r[prefix+metric] for r in rows]))
        return result
    summary = dict(series=a.series, origins=len(origins), embedding_rows=len(x),
                   closest_used=min(a.closest, len(x)), overall=stats(records),
                   by_horizon={str(j): stats([r for r in records if r['horizon']==j]) for j in range(1, a.h+1)})
    metadata = dict(config={k: str(v) if isinstance(v, (Path, date)) else v for k,v in vars(a).items()},
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                [a.synthetic, a.truth, Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).with_name('uncertainty.py')]},
        limitations=['Final truth backtest, not historical as-of data',
                     'SciPy smoothing spline replaces the R mgcv GAM',
                     'Calibration starts at the selected backtest start, without upstream early-pandemic forecast prehistory',
                     'Monte Carlo quantiles use NumPy RNG, not R RNG',
                     'Persistence NB intervals are an added diagnostic baseline'])
    for name, obj in [('summary.json', summary), ('metadata.json', metadata)]:
        (a.output_dir/name).write_text(json.dumps(obj, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
