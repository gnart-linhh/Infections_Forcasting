"""Python point-forecast component of sMOA; no dependency on the precog checkout."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.interpolate import make_smoothing_spline


def load_library(path: Path, k: int, h: int):
    """Embed each curve separately; k differences require k+1 history levels."""
    if k < 1 or h < 1:
        raise ValueError('k and h must be positive')
    curves = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            curves.setdefault(row['curve_id'], []).append((int(row['step']), float(row['value'])))
    xs, ys, ids, steps = [], [], [], []
    for curve_id, observations in curves.items():
        observations.sort()
        positions, values = map(np.asarray, zip(*observations))
        if not np.array_equal(positions, np.arange(1, len(values) + 1)):
            raise ValueError(f'Nonconsecutive or duplicate steps: {curve_id}')
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f'Invalid curve values: {curve_id}')
        if len(values) < k + 1 + h:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(np.maximum(values, 1e-8), k + 1 + h)
        keep = np.var(windows[:, :k+1], axis=1, ddof=1) > 0
        starts = np.flatnonzero(keep) + 1
        diffs = np.diff(windows[keep], axis=1)
        xs.append(diffs[:, :k]); ys.append(diffs[:, k:])
        ids.extend([curve_id] * len(diffs)); steps.extend(starts)
    if not xs or not sum(map(len, xs)):
        raise ValueError('No valid embedding windows')
    return np.vstack(xs), np.vstack(ys), np.asarray(ids), np.asarray(steps)


def forecast(history, x, y, closest=50, smoothing='spline'):
    values = np.maximum(np.asarray(history, dtype=float), 1)
    k = x.shape[1]
    if len(values) < k+1 or not np.all(np.isfinite(values)):
        raise ValueError('History must be finite and contain at least k+1 values')
    if closest < 1:
        raise ValueError('closest must be positive')
    if smoothing == 'spline' and len(values) >= 5:
        t = np.arange(len(values), dtype=float)
        fitted = np.maximum(1, make_smoothing_spline(t, values)(t))
    elif smoothing == 'none':
        fitted = values
    else:
        raise ValueError('Unsupported smoothing or fewer than five history points')
    distances = np.abs(x - np.diff(fitted[-k-1:])).sum(axis=1)
    # Stable ordering gives reproducible handling of tied distances.
    indices = np.argsort(distances, kind='stable')[:min(closest, len(x))]
    prediction = np.maximum(1, values[-1] + np.cumsum(np.median(y[indices], axis=0)))
    return prediction, indices, distances[indices]
