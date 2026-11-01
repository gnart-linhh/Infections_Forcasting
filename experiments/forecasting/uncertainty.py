"""Negative-binomial UQ with explicit matured or retrospective calibration."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

QUANTILES = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
LONG_QUANTILES = np.r_[.01,.025,np.arange(1,20)/20,.975,.99]
INITIAL_SIZES = (20.0, 10.0, 6.0, 0.3)


def fit_size(points, actuals):
    """Match upstream MLE on inverse size in (0,1], including its <100 rule."""
    means = np.asarray(points, dtype=float).copy()
    observed = np.asarray(actuals, dtype=float)
    if not len(means) or means.shape != observed.shape:
        raise ValueError('Need paired, nonempty calibration data')
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(observed)) or np.any(means < 1) or np.any(observed < 0):
        raise ValueError('Calibration means must be >=1 and outcomes >=0, all finite')
    means[means < 100] = observed[means < 100]

    def objective(inverse_size):
        size = 1.0 / inverse_size
        return -float(nbinom.logpmf(np.rint(observed), size, size / (size + means)).sum())

    result = minimize_scalar(objective, bounds=(1e-8, 1.0), method='bounded', options={'xatol': 1e-10})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Dispersion optimization failed')
    # Include boundaries; the lower numerical bound approximates the Poisson limit.
    inverse = min([1e-8, float(result.x), 1.0], key=objective)
    return 1.0 / inverse


def dispersion(prior, origin, horizon, field='forecast', horizons=4, policy='matured'):
    if policy not in ['matured', 'figure2_retrospective']:
        raise ValueError('Unknown calibration policy')
    eligible = [r for r in prior if r['horizon'] == horizon and
                (policy=='figure2_retrospective' or r['target_end_date'] <= origin) and
                np.isfinite(r['true_value']) and r['true_value'] >= 0]
    if horizon < 1 or horizon > len(INITIAL_SIZES):
        raise ValueError('Probabilistic forecasts support horizons 1 to 4')
    # Upstream switches after six complete forecast origins (24 stored rows).
    # Eligibility then follows policy; the current origin is never in prior.
    if len(prior) < 6 * horizons or not eligible:
        return INITIAL_SIZES[horizon-1], 'initial', len(eligible)
    return fit_size([r[field] for r in eligible], [r['true_value'] for r in eligible]), 'mle', len(eligible)


def interval_quantiles(point, size, rng, draws=5000, initial=False, probabilities=QUANTILES):
    if draws < 1 or not np.isfinite(point) or point < 1 or not np.isfinite(size) or size <= 0:
        raise ValueError('Invalid negative-binomial parameters')
    mean = point + 1 if initial else point
    sims = rng.negative_binomial(size, size / (size + mean), size=draws)
    probabilities = np.asarray(probabilities,float)
    median = np.flatnonzero(np.isclose(probabilities,.5))
    if len(median)!=1: raise ValueError('Quantile grid must contain one median')
    quantiles = np.quantile(sims, probabilities)
    # Match upstream median recentering, including possibly negative lower bounds.
    return point + quantiles - quantiles[median[0]]


def scores(quantiles, actual):
    values = np.asarray(quantiles, dtype=float)
    if values.shape != QUANTILES.shape or not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0) or not np.isfinite(actual):
        raise ValueError('Need seven ordered finite quantiles and a finite actual')
    wis = float(2 * np.mean(np.maximum(QUANTILES * (actual-values), (1-QUANTILES) * (values-actual))))
    over = float(np.mean(np.r_[np.repeat(np.maximum(values[:3]-actual, 0), 2), max(values[3]-actual, 0)]))
    under = float(np.mean(np.r_[np.repeat(np.maximum(actual-values[4:], 0), 2), max(actual-values[3], 0)]))
    return dict(wis=wis, overprediction=over, underprediction=under, sharpness=wis-over-under,
                coverage_50=int(values[2] < actual <= values[4]),
                coverage_80=int(values[1] < actual <= values[5]),
                coverage_95=int(values[0] < actual <= values[6]),
                width_95=float(values[6]-values[0]))


def calibration_plan(prior,origin,horizons=4,policy='matured'):
    """Matured limits target dates, not data vintage or warmup look-ahead.

    Figure 2's retrospective convention must be explicit. Neither policy alone
    establishes a fully real-time evaluation.
    """
    result={}
    for field in ['forecast']:
        for horizon in range(1,horizons+1):
            size,method,count=dispersion(prior,origin,horizon,field,horizons,policy)
            excluded=sum(r['horizon']==horizon and (policy=='figure2_retrospective' or r['target_end_date']<=origin) and
                         np.isfinite(r['true_value']) and r['true_value']<0 for r in prior)
            future_count=sum(r['horizon']==horizon and policy=='figure2_retrospective' and r['target_end_date']>origin and
                             np.isfinite(r['true_value']) and r['true_value']>=0 for r in prior)
            result[(field,horizon)]=dict(size=size,method=method,count=count,excluded_negative=excluded,
                                        calibration_policy=policy,future_count=future_count)
    return result


def coverage_rows(values,actual,probabilities=LONG_QUANTILES):
    probabilities=np.asarray(probabilities)
    values=np.asarray(values)
    return [dict(level=round(float(1-2*probabilities[i]),10),lower_probability=float(probabilities[i]),upper_probability=float(probabilities[-i-1]),lower=float(values[i]),upper=float(values[-i-1]),
                 covered=int(values[i]<actual<=values[-i-1])) for i in range(len(probabilities)//2)]
