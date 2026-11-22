"""Direct sMOA modifications. No outcomes, model fitting, or disease metadata."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

# Preserve the original research modules' sibling-import interfaces.
FORECASTING = Path(__file__).resolve().parents[1]/'forecasting'
if str(FORECASTING) not in sys.path:
    sys.path.insert(0, str(FORECASTING))
from model import forecast as original_forecast
from neighbors import ExactNeighbors

MODELS = ('M0', 'M1', 'M2', 'M3')


def similarity_weights(distances, temperature: float | None):
    """tau = temperature * IQR(d); fall back to mean(d-min(d)) if IQR=0.

    None requests uniform weights. All-equal distances are also uniform. Work
    relative to max(distance) before computing spread, avoiding scale overflow.
    Subtraction of the minimum and long-double logits prevent unstable exp().
    """
    d = np.asarray(distances, float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Need finite nonnegative analogue distances')
    if temperature is not None and (not np.isfinite(temperature) or temperature <= 0):
        raise ValueError('Temperature coefficient must be finite and positive')
    if temperature is None or np.ptp(d) == 0:
        return np.full(len(d), 1/len(d))
    normalized = d/max(float(d.max()), np.finfo(float).tiny)
    offset = normalized-normalized.min()
    spread = float(np.quantile(normalized, .75)-np.quantile(normalized, .25))
    if spread == 0:
        spread = float(offset.mean())
    with np.errstate(over='ignore', under='ignore'):
        log_weight = -offset.astype(np.longdouble)/np.longdouble(spread)/np.longdouble(temperature)
        weights = np.exp(log_weight).astype(float)
    return weights/weights.sum()


def weighted_median(values, weights):
    """Columnwise midpoint weighted median, with stable ordering.

    At a half-mass boundary, average the two adjacent positive-mass values.
    Uniform weights call NumPy median exactly (including even sample sizes).
    Boundaries within four machine epsilons count as exactly half the mass.
    """
    a, w = np.asarray(values, float), np.asarray(weights, float)
    scalar = a.ndim == 1
    if scalar:
        a = a[:, None]
    if a.ndim != 2 or not len(a) or w.shape != (len(a),) or not np.isfinite(a).all() or not np.isfinite(w).all() or (w < 0).any() or not w.max() > 0:
        raise ValueError('Need finite values and nonnegative positive-total weights')
    if np.all(w == w[0]):
        with np.errstate(over='ignore'):
            result = np.median(a, axis=0)
        if not np.isfinite(result).all():
            ordered = np.sort(a, axis=0)
            result = ordered[(len(a)-1)//2]/2+ordered[len(a)//2]/2
    else:
        positive = w > 0
        a, w = a[positive], w[positive]
        w = w/w.max(); w = w/w.sum()
        result = []
        for column in a.T:
            order = np.argsort(column, kind='stable')
            z, mass = column[order], np.cumsum(w[order])
            index = min(int(np.searchsorted(mass, .5)), len(z)-1)
            if abs(mass[index]-.5) <= 4*np.finfo(float).eps and index+1 < len(z):
                result.append(z[index]/2+z[index+1]/2)
            else:
                result.append(z[index])
        result = np.asarray(result)
    return float(result[0]) if scalar else result


def regularize(point, anchor, dispersion, beta):
    """Convex shrinkage of the aggregation forecast toward its raw-scale anchor."""
    p, u = np.asarray(point, float), np.asarray(dispersion, float)
    if p.shape != u.shape or not np.isfinite(p).all() or not np.isfinite(u).all() or (p < 0).any() or (u < 0).any() or not np.isfinite(anchor) or anchor < 0 or not np.isfinite(beta) or beta < 0:
        raise ValueError('Invalid count-scale regularization inputs')
    if beta == 0:
        return p.copy(), np.ones_like(p)
    with np.errstate(divide='ignore'):
        # Preserve strict positivity even when the mathematical factor is below
        # float64's representable range; this is a numerical support safeguard.
        lam = np.maximum(expit(-np.log(beta)-np.log(u)), np.nextafter(0., 1.))
    # Convex form avoids overflow in p-anchor for very large valid counts.
    result = lam*p+(1-lam)*anchor
    result = np.where(u == 0, p, result)
    if not np.isfinite(result).all():
        raise ValueError('Regularized forecast exceeds representable count range')
    return np.maximum(0., result), lam


@dataclass(frozen=True)
class Match:
    history: np.ndarray
    baseline: np.ndarray
    indices: np.ndarray
    distances: np.ndarray
    future: np.ndarray
    dispersion: np.ndarray
    scale: float


class RobustSMOA:
    """One frozen reference index; cache only identical full supplied inputs.

    M0 delegates directly to the original forecast(). Every changed history
    recomputes GAM and matching. The cache belongs to this immutable reference
    and is bounded; neither precomputed real rankings nor labels are accepted.
    """
    def __init__(self, x, y, closest=4422, smoothing='gam', index=None, cache_size=128):
        self.x, self.y = np.asarray(x), np.asarray(y)
        if self.x.ndim != 2 or self.y.ndim != 2 or len(self.x) != len(self.y) or not len(self.x):
            raise ValueError('Need aligned nonempty reference embedding matrices')
        if not 1 <= closest <= len(self.x) or int(closest) != closest or cache_size < 0:
            raise ValueError('Neighbour count must not exceed reference windows')
        self.closest, self.smoothing = closest, smoothing
        self.index = index if index is not None else ExactNeighbors(self.x)
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.match_calls = 0

    def match(self, history):
        history = np.asarray(history, dtype='<f8')
        if history.ndim != 1 or len(history) < self.x.shape[1]+1 or not np.isfinite(history).all():
            raise ValueError('Need finite one-dimensional permitted history, n>=k+1')
        key = (len(history), history.tobytes())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        baseline, ids, distances = original_forecast(history, self.x, self.y, self.closest,
                                                     self.smoothing, neighbor_index=self.index)
        self.match_calls += 1
        future = np.asarray(self.y[ids], float)
        scale = max(float(np.maximum(history, 1).mean()), 1.)
        changes = np.cumsum(future, axis=1)
        spread = np.quantile(changes, .75, axis=0)-np.quantile(changes, .25, axis=0)
        dispersion = spread/scale
        if not np.isfinite(baseline).all() or not np.isfinite(dispersion).all():
            raise ValueError('Reference forecast/dispersion exceeds finite numeric range')
        matched = Match(history.copy(), baseline, ids, distances, future, dispersion, scale)
        if self.cache_size:
            self.cache[key] = matched
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return matched

    def variants(self, history, temperature, beta):
        matched = self.match(history)
        weights = similarity_weights(matched.distances, temperature)
        anchor = max(float(matched.history[-1]), 1.)
        weighted = np.maximum(1., anchor+np.cumsum(weighted_median(matched.future, weights)))
        regular, lam = regularize(matched.baseline, anchor, matched.dispersion, beta)
        combined, _ = regularize(weighted, anchor, matched.dispersion, beta)
        forecasts = dict(M0=matched.baseline.copy(), M1=weighted, M2=regular, M3=combined)
        if not np.isfinite(list(forecasts.values())).all():
            raise ValueError('Nonfinite forecast')
        diagnostics = dict(dispersion=matched.dispersion, shrinkage=lam,
                           minimum_distance=float(matched.distances.min()),
                           mean_distance=float(matched.distances.mean()),
                           effective_analogues=float(1/np.sum(weights**2)), scale=matched.scale)
        return forecasts, diagnostics
