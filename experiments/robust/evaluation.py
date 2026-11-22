"""Central paired point and interval metrics; all sample exclusions are explicit."""
from __future__ import annotations

import numpy as np
import pandas as pd

PROBABILITIES = np.array([.025, .1, .25, .5, .75, .9, .975])
KEYS = ['case_id', 'horizon']


def point_metrics(actual, prediction):
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    if not len(actual) or actual.shape != prediction.shape or not np.isfinite([actual, prediction]).all():
        raise ValueError('Need nonempty paired finite values')
    error = prediction-actual
    return dict(n=len(actual), mae=float(np.mean(np.abs(error))),
                rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()))


def interval_metrics(actual, quantiles):
    actual, q = np.asarray(actual, float), np.asarray(quantiles, float)
    if q.shape != (len(actual), 7) or not len(actual) or not np.isfinite(q).all() or not np.isfinite(actual).all() or (np.diff(q, axis=1) < 0).any():
        raise ValueError('Need seven ordered finite quantiles per observation')
    residual = actual[:, None]-q
    # Seven-quantile pinball representation of WIS, median weight 1/2.
    wis = 2*np.maximum(PROBABILITIES*residual, (PROBABILITIES-1)*residual).mean(axis=1)
    result = dict(n=len(actual), wis=float(wis.mean()))
    for level, i in [(50, 2), (80, 1), (95, 0)]:
        lower, upper = q[:, i], q[:, -i-1]
        coverage = float(np.mean((lower <= actual) & (actual <= upper)))
        result.update({f'coverage_{level}': coverage, f'width_{level}': float(np.mean(upper-lower)),
                       f'coverage_error_{level}': coverage-level/100,
                       f'lower_miss_{level}': float(np.mean(actual < lower)),
                       f'upper_miss_{level}': float(np.mean(actual > upper))})
    return result


def paired_metrics(predictions, group_by=('dataset',), models=None, baseline_model='M0'):
    from experiments.robust.method import MODELS
    models=MODELS if models is None else tuple(models)
    if predictions.duplicated(KEYS+['model']).any():
        raise ValueError('Duplicate prediction keys')
    if predictions.groupby(KEYS).actual.nunique(dropna=False).gt(1).any():
        raise ValueError('Models have inconsistent observed outcomes')
    output = []
    groups = predictions.groupby(list(group_by), dropna=False) if group_by else [((), predictions)]
    for key, block in groups:
        key = key if isinstance(key, tuple) else (key,)
        pivot = block.pivot(index=KEYS, columns='model', values='prediction').reindex(columns=models)
        valid = np.isfinite(pivot).all(axis=1) & (pivot >= 0).all(axis=1)
        pivot = pivot.loc[valid]
        if pivot.empty:
            raise ValueError('No common finite four-model sample')
        truth = block.drop_duplicates(KEYS).set_index(KEYS).actual.reindex(pivot.index)
        baseline = point_metrics(truth, pivot[baseline_model])['mae']
        for model in models:
            m = point_metrics(truth, pivot[model])
            error = pivot[model].to_numpy()-truth.to_numpy()
            m.update(model=model, requested_n=len(valid), excluded_n=int((~valid).sum()),
                     relative_mae_improvement=1-m['mae']/baseline if baseline else np.nan,
                     p95_absolute_error=float(np.quantile(np.abs(error), .95)),
                     p99_absolute_error=float(np.quantile(np.abs(error), .99)),
                     severe_underprediction=float(np.mean((truth > 0) & (pivot[model] < .5*truth))),
                     severe_overprediction=float(np.mean(pivot[model] > 2*np.maximum(truth, 1))),
                     regions_or_curves=block.region.nunique(), origins=block[['region', 'forecast_origin']].drop_duplicates().shape[0])
            m.update(zip(group_by, key))
            output.append(m)
    return pd.DataFrame(output)


def cluster_comparisons(predictions, seed=42, draws=500,baseline_model='M0'):
    result = []
    for dataset, frame in predictions.groupby('dataset'):
        pivot = frame.pivot(index=KEYS+['actual', 'region'], columns='model', values='prediction').dropna()
        actual = pivot.index.get_level_values('actual').to_numpy()
        errors = pivot.sub(actual, axis=0).abs()
        sums = errors.groupby(level='region').sum()
        counts = errors.groupby(level='region').size().to_numpy()
        rng = np.random.default_rng(seed)
        samples = rng.integers(len(sums), size=(draws, len(sums)))
        for model in errors.columns:
            delta = (sums[model]-sums[baseline_model]).to_numpy()
            boot = delta[samples].sum(axis=1)/counts[samples].sum(axis=1)
            result.append(dict(dataset=dataset, model=model, n=len(errors), clusters=len(sums),
                               mae_delta_vs_original=float(delta.sum()/counts.sum()),
                               lower=float(np.quantile(boot, .025)), upper=float(np.quantile(boot, .975))))
    return pd.DataFrame(result)


def uncertainty_metrics(predictions, group_by=('dataset', 'model')):
    columns = ['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975']
    return pd.DataFrame([dict(zip(group_by, key), **interval_metrics(group.actual, group[columns]))
                         for key, group in predictions.groupby(list(group_by))])
