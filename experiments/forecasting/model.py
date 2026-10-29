"""Python point-forecast component of sMOA; no dependency on the precog checkout."""
from __future__ import annotations
import csv
from itertools import groupby
from pathlib import Path
import numpy as np
from scipy.interpolate import make_smoothing_spline
from neighbors import ExactNeighbors
from smoothing import thin_plate_gcv


def load_library(path: Path, k: int, h: int):
    """Embed each curve separately; k differences require k+1 history levels."""
    if k < 1 or h < 1:
        raise ValueError('k and h must be positive')
    xs, ys, ids, steps = [], [], [], []
    def curves():
        seen=set()
        with path.open() as handle:
            for curve_id,rows in groupby(csv.DictReader(handle),key=lambda r:r['curve_id']):
                if curve_id in seen: raise ValueError('CSV rows must be grouped by curve_id')
                seen.add(curve_id)
                yield curve_id,[(int(r['step']),float(r['value'])) for r in rows]
    for curve_id, observations in curves():
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


def forecast(history, x, y, closest=4422, smoothing='gam', clip_smoothed=True, neighbor_index=None):
    values = np.maximum(np.asarray(history, dtype=float), 1)
    k = x.shape[1]
    if len(values) < k+1 or not np.all(np.isfinite(values)):
        raise ValueError('History must be finite and contain at least k+1 values')
    if closest < 1:
        raise ValueError('closest must be positive')
    if smoothing == 'gam' and len(values) >= 5:
        fitted = thin_plate_gcv(values)
        if clip_smoothed: fitted = np.maximum(1, fitted)
    elif smoothing == 'spline' and len(values) >= 5:
        t = np.arange(len(values), dtype=float)
        fitted = make_smoothing_spline(t, values)(t)
        if clip_smoothed: fitted = np.maximum(1, fitted)
    elif smoothing == 'none':
        fitted = values
    else:
        raise ValueError('Unsupported smoothing or fewer than five history points')
    index=neighbor_index if neighbor_index is not None else ExactNeighbors(x)
    indices,distances=index.query(np.diff(fitted[-k-1:]),closest)
    indices,distances=indices[0],distances[0]
    prediction = np.maximum(1, values[-1] + np.cumsum(np.median(y[indices], axis=0)))
    return prediction, indices, distances
