"""E6: stratified evaluation of the saved original sMOA negative-binomial UQ."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'infections-e6-matplotlib'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from runtime import atomic_json, file_hash, acquire_run_lock
from experiments.error_analysis.conditions import features, RULES as CONDITION_RULES
from experiments.forecasting.uncertainty import QUANTILES, LONG_QUANTILES, fit_size

WINDOW_FILTER = 'nonconstant_v2'
UQ_POLICY = 'matured'
VERSION = 2

KEYS = ['series', 'forecast_origin', 'target_end_date', 'horizon']
BOOTSTRAPS = 500
SEED = 20261102
HISTORY = ['27–52 weeks', '53–104 weeks', '105+ weeks']
CALIBRATION = ['24–49 pairs', '50–99 pairs', '100+ pairs']
RULES = dict(
    version=VERSION, window_filter=WINDOW_FILTER, uq_policy=UQ_POLICY,
    parameters=dict(N=18387, k=5, m=4422, bridge='omit', smoothing='gam'),
    cohort='50 states x 112 weekly origins x 4 horizons; unchanged paper_full forecasts',
    UQ='saved NB draws: 5000 per forecast, seed 13, median recentering; audit refits dispersion without replacing saved intervals',
    policy='matured: earlier-issued forecasts with target_end_date <= current origin; final revised outcomes',
    warmup='24 earlier weekly origins from frozen 2020-08-15 vintage; retrospective initialization',
    coverage='PRIMARY: lower <= actual <= upper; ALSO source convention lower < actual <= upper',
    WIS='original seven-quantile WIS = 2 * mean(pinball loss), probabilities .025,.1,.25,.5,.75,.9,.975',
    truth='final raw truth, including negative revisions; never floor scoring labels or interval bounds',
    strata=CONDITION_RULES,
    history='full available calendar history: 27–52, 53–104, 105+ weeks; early <=26 explicitly empty',
    calibration='unchanged bins: <24, 24–49, 50–99, 100+ eligible NB pairs; all scored forecasts already MLE',
    width='MPIW in cases; normalized width = width / max(mean(last 4 origin-available floored counts),1)',
    width_error='Spearman per horizon; raw and normalized; partial ranks adjusted for origin context',
    partial_context='log1p(origin level), asinh(growth), volatility, history_weeks, rising/declining indicators',
    bootstrap=f'{BOOTSTRAPS} whole-state resamples, seed {SEED}; all origins within state retained; descriptive',
    example='California, origin 2021-12-18, fixed before E6 inspection; illustrative, not representative',
    interpretation='observational: history, calibration_n, calendar and epidemic waves are confounded; no causal claim',
)


def history_bin(weeks):
    if weeks <= 26:
        return 'early <=26 weeks'
    return HISTORY[0] if weeks <= 52 else HISTORY[1] if weeks <= 104 else HISTORY[2]


def calibration_bin(count):
    if count < 24:
        return '<24 pairs'
    return CALIBRATION[0] if count < 50 else CALIBRATION[1] if count < 100 else CALIBRATION[2]


def quantile_matrix(frame, expected_keys):
    """Reject incomplete grids and duplicate targets instead of dropping rows in a join."""
    if not frame.model.eq('smoa').all() or frame.duplicated(KEYS + ['quantile']).any():
        raise ValueError('Require unique sMOA quantiles')
    grid = frame.pivot(index=KEYS, columns='quantile', values='value')
    expected = pd.MultiIndex.from_frame(expected_keys[KEYS])
    if len(grid) != len(expected) or set(grid.index) != set(expected):
        raise ValueError('Quantile target cohort mismatch')
    if grid.shape[1] != len(LONG_QUANTILES) or not np.allclose(grid.columns, LONG_QUANTILES, atol=1e-12, rtol=0):
        raise ValueError('Require the original complete 23-quantile grid')
    values = grid.reindex(expected).to_numpy(float)
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise ValueError('Nonfinite or crossing quantiles')
    return values


def interval_arrays(values, actual):
    lower, upper = values[:, :11], values[:, -1:-12:-1]
    actual = np.asarray(actual)[:, None]
    return dict(covered=(lower <= actual) & (actual <= upper),
                source_covered=(lower < actual) & (actual <= upper), width=upper-lower,
                lower_miss=actual < lower, upper_miss=actual > upper)


def wis_values(values, actual):
    positions = [int(np.flatnonzero(np.isclose(LONG_QUANTILES, q))[0]) for q in QUANTILES]
    q = values[:, positions]
    difference = np.asarray(actual)[:, None] - q
    return 2 * np.maximum(QUANTILES * difference, (QUANTILES - 1) * difference).mean(axis=1)


def origin_features(archive, origin, expected_dates):
    rows = archive.loc[archive.as_of.eq(origin)].sort_values('target_end_date')
    if rows.target_end_date.duplicated().any() or rows.target_end_date.tolist() != list(expected_dates):
        raise ValueError(f'Missing exact origin history: {origin}')
    result = features(rows.value.to_numpy(float))
    result['history_band'] = history_bin(result['history_weeks'])
    return result


def verify_calibration(forecasts, calibration, warmup):
    """Independently select matured pairs and refit every saved NB size for audit."""
    if calibration.duplicated(['series', 'forecast_origin', 'horizon']).any():
        raise ValueError('Duplicate calibration plan')
    if not calibration.model.eq('smoa').all() or not calibration.calibration_policy.eq(UQ_POLICY).all():
        raise ValueError('E6 requires the matured policy')
    if not calibration.method.eq('mle').all():
        raise ValueError('Full reference cohort is expected to be MLE after warmup')
    joined = forecasts.merge(calibration, on=['series', 'forecast_origin', 'horizon'], how='left',
                             validate='one_to_one', suffixes=('', '_plan'), indicator=True)
    if len(calibration)!=len(forecasts) or not joined._merge.eq('both').all():
        raise ValueError('Missing calibration plan')
    joined=joined.drop(columns='_merge')
    for left, right in [('nb_size', 'nb_size_plan'), ('calibration_n', 'calibration_n_plan'),
                        ('negative_calibration_excluded', 'excluded_negative')]:
        if not np.array_equal(joined[left], joined[right]):
            raise ValueError(f'Calibration disagreement: {left}')
    prior = pd.concat([warmup, forecasts], ignore_index=True)
    if prior.duplicated(KEYS).any():raise ValueError('Duplicate prior calibration pairs')
    rebuilt = pd.Series(index=joined.index,dtype=float)
    for (series,h), rows in joined.groupby(['series','horizon']):
        p = prior.loc[prior.series.eq(series) & prior.horizon.eq(h)]
        earlier = p.forecast_origin.to_numpy()[None, :] < rows.forecast_origin.to_numpy()[:, None]
        eligible = earlier & (p.target_end_date.to_numpy()[None,:] <= rows.forecast_origin.to_numpy()[:,None])
        valid = np.isfinite(p.true_value.to_numpy()) & p.true_value.ge(0).to_numpy()
        counts = (eligible & valid).sum(axis=1)
        future = (eligible & valid & (p.target_end_date.to_numpy()[None, :] > rows.forecast_origin.to_numpy()[:, None])).sum(axis=1)
        negatives = (eligible & p.true_value.lt(0).to_numpy()).sum(axis=1)
        if not (np.array_equal(counts, rows.calibration_n) and
                np.array_equal(future, rows.future_calibration_n) and
                np.array_equal(negatives, rows.excluded_negative)):
            raise ValueError('Calibration counts/future-label audit failed')
        for i,row in enumerate(rows.itertuples()):
            issued=prior.loc[prior.series.eq(series) & prior.forecast_origin.lt(row.forecast_origin)]
            selected=p.loc[eligible[i] & valid]
            if len(issued)<24 or selected.empty:raise ValueError('Expected MLE after warmup')
            size=fit_size(selected.forecast.to_numpy(),selected.true_value.to_numpy())
            if size!=row.nb_size:raise ValueError(f'Rebuilt NB size differs: {series} {row.forecast_origin} h{h}')
            rebuilt.at[row.Index]=size
    if not joined.future_calibration_n.eq(0).all():raise ValueError('Future targets entered matured MLE')
    joined['nb_size_recomputed']=rebuilt
    return joined


def load_reference(source, archive_dir, inputs):
    def read(path):
        inputs[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = file_hash(path)
        if path.suffix == '.json':
            return json.loads(path.read_text())
        return pd.read_csv(path, float_precision='round_trip')

    config = read(source/'run_config.json')
    if read(source/'pipeline.json').get('status')!='complete':raise ValueError('Full run is incomplete')
    if config['method'].get('window_filter')!=WINDOW_FILTER or config['method'].get('uq_policy')!=UQ_POLICY:
        raise ValueError('Wrong full-run version/filter/UQ policy')
    snapshot=read(source/'source_snapshot_sha256.json')
    for name in ['model.py','smoothing.py','neighbors.py','uncertainty.py']:
        logical='experiments/forecasting/'+name
        frozen=source/'source_snapshot'/logical
        read_hash=file_hash(frozen)
        if read_hash!=snapshot[logical] or file_hash(ROOT/logical)!=read_hash:
            raise ValueError(f'Full numerical source differs: {logical}')
        inputs[str(frozen)]=read_hash
        inputs[logical]=read_hash
    if file_hash(Path(config['library']))!=config['library_sha256']:raise ValueError('Full library modified')
    inputs[config['library']]=config['library_sha256']
    policy = read(source/'03_uq/policy.json')
    if config['parameters'] != dict(N=18387, k=5, m=4422, h=4) or policy['bridge_mode'] != 'omit' or policy['policy'] != UQ_POLICY:
        raise ValueError('Wrong full-run parameters or UQ policy')
    if policy['draws'] != 5000 or policy['seed'] != 13:
        raise ValueError('Unexpected NB generation configuration')
    manifest = read(archive_dir/'manifest.csv')
    origins = read(archive_dir/'origins.csv')
    states = []
    for folder in sorted((source/'04_prediction').glob('[0-9][0-9]')):
        f = read(folder/'forecasts.csv')
        if len(f) != 448 or f.duplicated(KEYS).any() or f.series.nunique() != 1:
            raise ValueError('Incomplete or duplicate state targets')
        name = f.series.iloc[0]
        meta = read(folder/'metadata.json')
        gen=meta
        if 'generation_metadata' in meta:
            generation = Path(meta['generation_metadata'])
            if file_hash(generation) != meta['generation_metadata_sha256']:
                raise ValueError('Generation metadata changed')
            gen = read(generation)
        if gen.get('window_filter')!=WINDOW_FILTER:raise ValueError('State generation filter mismatch')
        cfg = gen['config']
        if any(cfg[k] != v for k, v in dict(smoothing='gam', bridge_mode='omit', uq_policy=UQ_POLICY,
                calibration_truth='final', score_truth='raw', prehistory_mode='frozen', frozen_as_of_date='2020-08-15',
                k=5, closest=4422, interval_draws=5000, seed=13, history_mode='as_of').items()):
            raise ValueError('Generation protocol mismatch')
        if gen['sha256'][config['library']]!=config['library_sha256']:raise ValueError('State library differs')
        for path,sha in gen['sha256'].items():
            if '/source_snapshot/' in path:
                if file_hash(Path(path))!=sha:raise ValueError('Generation source snapshot modified')
                inputs[path]=sha
        entry = manifest.loc[manifest.series.eq(name)].iloc[0]
        archive_path = archive_dir/entry.as_of_truth
        archive = read(archive_path)
        truth_path = Path(entry.truth)
        truth = read(truth_path)
        for p in [archive_path, truth_path]:
            if file_hash(p) != gen['sha256'][str(p)]:
                raise ValueError('History/truth differs from original generation')
        truth_map = truth.set_index('target_end_date').value
        if not np.array_equal(f.true_value, f.target_end_date.map(truth_map)):
            raise ValueError('Scoring truth mismatch')
        if not (pd.to_datetime(f.target_end_date) == pd.to_datetime(f.forecast_origin)+pd.to_timedelta(7*f.horizon, unit='D')).all():
            raise ValueError('Target/horizon misalignment')
        original_keys = f[KEYS].reset_index(drop=True).copy()
        q = quantile_matrix(read(folder/'quantiles.csv'), f)
        arrays = interval_arrays(q, f.true_value)
        if not np.allclose(q[:, 11], f.forecast, rtol=0, atol=1e-9):
            raise ValueError('Median does not equal point forecast')
        if not np.allclose(wis_values(q, f.true_value), f.wis, atol=1e-7, rtol=1e-12):
            raise ValueError('Original seven-quantile WIS mismatch')
        if not np.array_equal(arrays['source_covered'][:, 1], f.coverage_95):
            raise ValueError('Upstream coverage convention mismatch')
        f = verify_calibration(f, read(folder/'calibration.csv'), read(folder/'warmup_forecasts.csv'))
        if not f.dispersion_method.eq(f.method).all():raise ValueError('Dispersion method differs from plan')
        contexts = []
        for origin in sorted(f.forecast_origin.unique()):
            dates = truth.loc[truth.target_end_date.le(origin), 'target_end_date'].tolist()
            contexts.append(dict(series=name, forecast_origin=origin, **origin_features(archive, origin, dates)))
        f = f.merge(pd.DataFrame(contexts), on=['series', 'forecast_origin'], validate='many_to_one', sort=False)
        # Merge order must preserve quantile alignment even if a join implementation changes.
        if not f[KEYS].reset_index(drop=True).equals(original_keys):
            raise AssertionError('Merge changed quantile alignment')
        for i, probability in enumerate(LONG_QUANTILES):
            f[f'q_{probability:.3f}'] = q[:, i]
        f['calibration_band'] = f.calibration_n.map(calibration_bin)
        f['absolute_error'] = (f.forecast-f.true_value).abs()
        f['normalized_error'] = f.absolute_error/f.normalization
        f['width_95'] = arrays['width'][:, 1]
        f['normalized_width_95'] = f.width_95/f.normalization
        f['covered_95_inclusive'] = arrays['covered'][:, 1]
        f['negative_actual'] = f.true_value.lt(0)
        states.append(f)
        print(f'Validated {len(states)}/50 states: {name}; 448 NB sizes refitted',flush=True)
    data = pd.concat(states, ignore_index=True)
    if len(data) != 22400 or data.series.nunique() != 50 or data.forecast_origin.nunique() != 112:
        raise ValueError('Require complete 50-state, 112-origin reference run')
    if data.duplicated(KEYS).any() or not data.groupby(['series', 'forecast_origin']).horizon.apply(lambda s: sorted(s)==[1,2,3,4]).all():
        raise ValueError('Incomplete horizon panel')
    data = data.merge(origins[['forecast_origin', 'data_cutoff', 'issuance_date']], on='forecast_origin', validate='many_to_one')
    upstream=read(source/'05_evaluation/summary.json')
    for key,value in [('wis',data.wis.mean()),('mae',data.absolute_error.mean()),('coverage_95',data.coverage_95.mean())]:
        if not np.isclose(value,upstream['models']['smoa'][key],rtol=1e-12,atol=1e-8):
            raise ValueError(f'Full evaluation mismatch: {key}')
    for h,g in data.groupby('horizon'):
        for key,value in [('wis',g.wis.mean()),('mae',g.absolute_error.mean())]:
            if not np.isclose(value,upstream['by_horizon'][str(h)]['smoa'][key],rtol=1e-12,atol=1e-8):
                raise ValueError(f'Full horizon evaluation mismatch: h{h} {key}')
    return data


def state_interval(frame, columns):
    grouped = frame.groupby('series', sort=True)
    sums = grouped[columns].sum().to_numpy(float)
    sizes = grouped.size().to_numpy(float)
    if len(sizes) < 2:
        return np.full((2, len(columns)), np.nan)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(sizes), size=(BOOTSTRAPS, len(sizes)))
    estimates = sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)[:, None]
    return np.quantile(estimates, [.025, .975], axis=0)


def strata(data):
    yield 'overall', 'all', data
    for dimension in ['horizon', 'history_band', 'history_stage', 'phase', 'case_level', 'calibration_band']:
        for name, group in data.groupby(dimension, sort=True):
            yield dimension, str(name), group
    for dimension in ['history_band', 'phase', 'case_level', 'calibration_band']:
        for (h, name), group in data.groupby(['horizon', dimension], sort=True):
            yield 'horizon × '+dimension, f'{h} | {name}', group


def summarize(data):
    metric_rows, interval_rows = [], []
    levels = np.round(1-2*LONG_QUANTILES[:11], 10)
    for dimension, name, g in strata(data):
        base = dict(dimension=dimension, group=name, n=len(g), states=g.series.nunique(), origins=g.forecast_origin.nunique())
        w = g.assign(normalized_WIS=g.wis/g.normalization)
        ci = state_interval(w, ['wis', 'absolute_error'])
        metric_rows.append(dict(**base, WIS=g.wis.mean(), WIS_lo=ci[0, 0], WIS_hi=ci[1, 0],
            MAE=g.absolute_error.mean(), MAE_lo=ci[0, 1], MAE_hi=ci[1, 1], normalized_WIS=w.normalized_WIS.mean(),
            nb_size_median=g.nb_size.median(), calibration_n_mean=g.calibration_n.mean(),
            future_calibration_n_mean=g.future_calibration_n.mean(), negative_actuals=int(g.negative_actual.sum())))
        values = g[[f'q_{q:.3f}' for q in LONG_QUANTILES]].to_numpy()
        arrays = interval_arrays(values, g.true_value)
        for i, level in enumerate(levels):
            work = pd.DataFrame(dict(series=g.series.to_numpy(), coverage=arrays['covered'][:, i].astype(float),
                MPIW=arrays['width'][:, i], normalized_MPIW=arrays['width'][:, i]/g.normalization.to_numpy()))
            ci = state_interval(work, ['coverage', 'MPIW', 'normalized_MPIW'])
            empirical = float(work.coverage.mean())
            interval_rows.append(dict(**base, level=level, coverage=empirical, coverage_lo=ci[0, 0], coverage_hi=ci[1, 0],
                calibration_gap=empirical-level, source_coverage=arrays['source_covered'][:, i].mean(),
                boundary_additions=int((arrays['covered'][:, i] & ~arrays['source_covered'][:, i]).sum()),
                lower_miss_rate=arrays['lower_miss'][:, i].mean(), upper_miss_rate=arrays['upper_miss'][:, i].mean(),
                MPIW=work.MPIW.mean(), MPIW_lo=ci[0, 1], MPIW_hi=ci[1, 1],
                normalized_MPIW=work.normalized_MPIW.mean(), normalized_MPIW_lo=ci[0, 2], normalized_MPIW_hi=ci[1, 2]))
    return pd.DataFrame(metric_rows), pd.DataFrame(interval_rows)


def rho(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(spearmanr(x, y).statistic)


def context_matrix(g):
    return np.column_stack([np.ones(len(g)), np.log1p(g.recent_cases), np.arcsinh(g.growth),
        g.volatility, g.history_weeks, g.phase.eq('rising').astype(float), g.phase.eq('declining').astype(float)])


def partial_rho(x, y, context):
    x, y = rankdata(x), rankdata(y)
    controls = np.column_stack([np.ones(len(x))]+[rankdata(context[:, j]) for j in range(1, context.shape[1])])
    x = x-controls@np.linalg.lstsq(controls, x, rcond=None)[0]
    y = y-controls@np.linalg.lstsq(controls, y, rcond=None)[0]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def width_correlations(data):
    rows = []
    for h, g in data.groupby('horizon'):
        for kind, width, error in [('raw', 'width_95', 'absolute_error'),
                                    ('normalized', 'normalized_width_95', 'normalized_error'),
                                    ('context_adjusted', 'normalized_width_95', 'normalized_error')]:
            def estimate(frame):
                if kind == 'context_adjusted':
                    return partial_rho(frame[width], frame[error], context_matrix(frame))
                return rho(frame[width], frame[error])
            value = estimate(g)
            clusters = [s for _, s in g.groupby('series', sort=True)]
            rng = np.random.default_rng(SEED)
            boot = [estimate(pd.concat([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))], ignore_index=True))
                    for _ in range(BOOTSTRAPS)]
            lo, hi = np.nanquantile(boot, [.025, .975])
            rows.append(dict(horizon=h, analysis=kind, n=len(g), rho=value, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def make_figures(out, data, metrics, calibration, correlations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    directory = out/'figures'; directory.mkdir(exist_ok=True)
    def save(fig, name):
        for ext in ['png', 'svg']:
            fig.savefig(directory/f'{name}.{ext}', dpi=155, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for h, g in calibration.loc[calibration.dimension.eq('horizon')].groupby('group'):
        g = g.sort_values('level')
        axes[0].plot(g.level, g.coverage, 'o-', ms=4, label=f'{h} week')
        axes[1].plot(g.level, g.normalized_MPIW, 'o-', ms=4, label=f'{h} week')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Nominal')
    axes[0].set(xlabel='Nominal interval coverage', ylabel='Empirical coverage (inclusive)', xlim=(0, 1), ylim=(0, 1), title='Calibration by horizon')
    axes[1].set(xlabel='Nominal interval coverage', ylabel='Mean width / origin case level', title='Sharpness by horizon')
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    save(fig, 'reliability_width')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    for ax, dim, labels in zip(axes, ['phase', 'case_level', 'history_band'],
                                [['declining', 'stable', 'rising'], ['low', 'medium', 'high'], HISTORY]):
        panel = calibration.loc[calibration.dimension.eq('horizon × '+dim) & calibration.level.eq(.95)].copy()
        panel[['h', 'label']] = panel.group.str.split(' | ', regex=False, expand=True)
        table = panel.pivot(index='label', columns='h', values='coverage').reindex(labels)
        image = ax.imshow(table.to_numpy()*100, cmap='RdYlGn', vmin=55, vmax=100, aspect='auto')
        ax.set(xticks=range(4), xticklabels=['1', '2', '3', '4'], yticks=range(len(labels)), yticklabels=labels,
               xlabel='Horizon (weeks)', title=dim.replace('_', ' '))
        for (i, j), value in np.ndenumerate(table.to_numpy()):
            n = panel.loc[panel.label.eq(labels[i]) & panel.h.eq(str(j+1)), 'n'].iloc[0]
            ax.text(j, i, f'{value:.1%}\nn={n:,}', ha='center', va='center', fontsize=8)
    fig.colorbar(image, ax=list(axes), label='Empirical coverage (%)', shrink=.8)
    fig.suptitle('95% intervals: stratified calibration (nominal = 95%)')
    save(fig, 'stratified_coverage95')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for h, g in data.groupby('horizon'):
        # All data enter estimates; fixed random subsample is used only to avoid overplotting.
        sample = g.sample(n=min(800, len(g)), random_state=SEED)
        axes[0].scatter(sample.normalized_width_95, sample.normalized_error, s=6, alpha=.22, label=f'{h} week')
        ordered = g.sort_values('normalized_width_95').copy()
        ordered['display_bin'] = np.minimum(np.arange(len(ordered))*5//len(ordered), 4)
        means = ordered.groupby('display_bin')[['normalized_width_95', 'normalized_error']].mean()
        axes[1].plot(means.normalized_width_95, means.normalized_error, 'o-', label=f'{h} week')
    for ax in axes:
        ax.set(xlabel='95% width / origin case level', ylabel='Absolute error / origin case level')
        ax.legend(fontsize=8)
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    axes[0].set(xscale='symlog', yscale='symlog')
    # Explicit readable ticks; auto symlog ticks can show only two odd boundary labels.
    for axis in [axes[0].xaxis, axes[0].yaxis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    axes[0].set_xticks([0, 1, 10]); axes[0].set_yticks([0, 1, 10])
    axes[1].set(xlim=(0, None), ylim=(0, None))
    axes[0].set_title('Width and realized error (display sample only)')
    axes[1].set_title('Five equal-count display bins per horizon')
    save(fig, 'width_error')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for h, g in data.groupby('horizon'):
        trajectory = g.groupby('forecast_origin').agg(n=('calibration_n', 'mean'), size=('nb_size', 'median'), width=('normalized_width_95', 'mean'))
        dates = pd.to_datetime(trajectory.index)
        axes[0].plot(dates, trajectory.n, label=f'{h} week')
        axes[1].plot(dates, trajectory['size'], label=f'{h} week')
        axes[2].plot(dates, trajectory.width, label=f'{h} week')
    for ax in axes:
        ax.tick_params(axis='x', rotation=35); ax.legend(fontsize=7); ax.set_xlabel('Forecast origin')
    axes[0].set(ylabel='Mean eligible calibration pairs', title='All origins already use MLE')
    axes[1].set(ylabel='Median NB size (log scale)', yscale='log', title='Smaller size: more relative dispersion')
    axes[2].set(ylabel='Mean width / origin case level', title='Width changes with size AND point forecast')
    fig.suptitle('Matured calibration over time; frozen warmup and final revisions')
    save(fig, 'dispersion_history')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    c = calibration.loc[calibration.dimension.eq('horizon') & calibration.level.eq(.95)].sort_values('group')
    m = metrics.loc[metrics.dimension.eq('horizon')].sort_values('group')
    x = np.arange(1, 5)
    axes[0].errorbar(x, c.coverage*100, yerr=np.stack([c.coverage-c.coverage_lo, c.coverage_hi-c.coverage])*100, fmt='o-', capsize=4)
    axes[0].axhline(95, color='k', ls='--'); axes[0].set(ylabel='Coverage (%)', title='95% coverage, state bootstrap CI')
    axes[1].bar(x, c.MPIW); axes[1].set(ylabel='Cases', title='Mean 95% interval width')
    axes[2].plot(x, m.WIS, 'o-', label='WIS'); axes[2].plot(x, m.MAE, 's-', label='MAE'); axes[2].legend()
    axes[2].set(ylabel='Cases', title='Distributional and point accuracy')
    for ax in axes: ax.set(xticks=x, xlabel='Horizon (weeks)')
    save(fig, 'horizon_metrics')

    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    origin = '2021-12-18'
    g = data.loc[data.series.eq('California') & data.forecast_origin.eq(origin)].sort_values('horizon')
    dates = pd.to_datetime(g.target_end_date)
    for lo, hi, alpha, label in [('q_0.025', 'q_0.975', .15, '95% PI'), ('q_0.100', 'q_0.900', .22, '80% PI'), ('q_0.250', 'q_0.750', .32, '50% PI')]:
        ax.fill_between(dates, g[lo], g[hi], color='tab:blue', alpha=alpha, label=label)
    ax.plot(dates, g.forecast, 'o-', color='tab:blue', label='sMOA median')
    ax.plot(dates, g.true_value, 'o-', color='black', label='Final observed')
    ax.set(ylabel='Weekly cases', xlabel='Target week ending', title=f'California, origin {origin}: fixed illustrative example\nRetrospective UQ; not a representative accuracy estimate')
    ax.legend(ncol=3, fontsize=8)
    save(fig, 'forecast_intervals')


def table(frame, columns):
    labels = '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    return labels+'\n'.join('| '+' | '.join(f'{v:,.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in row)+' |'
                           for row in frame[columns].itertuples(index=False, name=None))


def write_report(out, data, metrics, calibration, correlations):
    overall = calibration.loc[calibration.dimension.eq('overall') & calibration.level.eq(.95)].iloc[0]
    m = metrics.loc[metrics.dimension.eq('overall')].iloc[0]
    h = calibration.loc[calibration.dimension.eq('horizon') & calibration.level.eq(.95)].merge(
        metrics.loc[metrics.dimension.eq('horizon'), ['group', 'WIS', 'MAE']], on='group')
    groups = calibration.loc[calibration.dimension.isin(['phase', 'case_level', 'history_band', 'calibration_band']) & calibration.level.eq(.95)]
    future = int(data.future_calibration_n.gt(0).sum())
    def coverage_for(dimension, group):
        return float(calibration.loc[calibration.dimension.eq(dimension) & calibration.group.eq(group) & calibration.level.eq(.95), 'coverage'].iloc[0])
    first_h, last_h = coverage_for('horizon', '1'), coverage_for('horizon', '4')
    adjusted = correlations.loc[correlations.analysis.eq('context_adjusted')].sort_values('horizon')
    first_n, last_n = coverage_for('calibration_band', CALIBRATION[0]), coverage_for('calibration_band', CALIBRATION[-1])
    report = f'''# Experiment 6 — Probabilistic Forecast Evaluation

50 bang × 112 origins × 4 horizons = **{len(data):,} targets**; origins {data.forecast_origin.min()}–{data.forecast_origin.max()}. Bộ paper N=18.387/k=5/m=4.422, GAM, bỏ bridge. Dùng lại toàn bộ 515.200 quantiles đã lưu từ NB 5.000 draws/forecast, seed 13; không chạy lại inference hay thay UQ.

## Kết quả chính

- Coverage 95% inclusive: **{overall.coverage:.4%}**; nominal 95%, gap **{100*overall.calibration_gap:.3f} điểm phần trăm**; CI state-bootstrap [{overall.coverage_lo:.2%}, {overall.coverage_hi:.2%}].
- MPIW 95%: **{overall.MPIW:,.2f} ca**; normalized MPIW **{overall.normalized_MPIW:.4f}**.
- **WIS {m.WIS:,.2f}; MAE {m.MAE:,.2f}**. WIS dùng đúng bảy quantiles của pipeline, không đổi sang WIS 23 quantiles.
- Source coverage dùng L < y ≤ U: {overall.source_coverage:.4%}; inclusive thêm {int(overall.boundary_additions)} targets ở biên dưới. Tất cả quantiles và WIS đã kiểm tra khớp output gốc.

{table(h, ['group', 'n', 'coverage', 'MPIW', 'normalized_MPIW', 'WIS', 'MAE'])}

**Khoảng 95% chưa đạt coverage 95%** trên cohort này: cả CI overall cũng nằm dưới nominal. Coverage giảm từ {first_h:.2%} ở tuần 1 xuống {last_h:.2%} ở tuần 4 dù MPIW tăng. Truth vượt upper bound trong {overall.upper_miss_rate:.2%} targets, thấp hơn lower bound trong {overall.lower_miss_rate:.2%}; phần miss chủ yếu nằm phía trên interval.

![Horizon metrics](figures/horizon_metrics.png)
![Reliability and width](figures/reliability_width.png)

## Điều kiện dịch và dữ liệu lịch sử

Các strata được định nghĩa trong protocol trước khi tính các thống kê E6. Phase dùng (mean last2 +1)/(mean previous2 +1)−1: rising >10%, declining <−10%, stable còn lại. Case level dùng mean last4 (floor input tại 1): low <1.000, medium <10.000, high ≥10.000. Features lấy đúng vintage input của từng origin, không dùng final truth hay tương lai. Origins là nhãn thứ Bảy, cutoff Chủ nhật và issuance thứ Hai theo archive; không coi snapshot đã có vào thứ Bảy.

{table(groups, ['dimension', 'group', 'n', 'coverage', 'calibration_gap', 'MPIW', 'normalized_MPIW'])}

Coverage theo phase tại origin: declining **{coverage_for('phase', 'declining'):.2%}**, rising **{coverage_for('phase', 'rising'):.2%}**, stable **{coverage_for('phase', 'stable'):.2%}**. Case-count low có coverage **{coverage_for('case_level', 'low'):.2%}**; riêng low ở tuần 4 còn **{coverage_for('horizon × case_level', '4 | low'):.2%}**. Phase mô tả lịch sử tại origin, không phải phase đã biết ở target tương lai. Số ca thấp không bảo đảm rủi ro tương đối thấp.

Một số strata có overcoverage nhẹ trên mẫu: history 27–52 tuần ở horizon 1 đạt **{coverage_for('horizon × history_band', '1 | '+HISTORY[0]):.2%}**. Cần đọc CI và số mẫu trong CSV trước khi coi chênh lệch nhỏ so với 95% là bằng chứng chắc chắn.

History thực tế {data.history_weeks.min()}–{data.history_weeks.max()} tuần; nhóm early ≤26 tuần **rỗng**. Dùng các dải cố định 27–52, 53–104, ≥105 tuần, không tự cắt history hoặc gọi nhóm đầu là “early epidemic”. Khác biệt giữa các dải cũng chứa khác biệt về thời điểm dịch. Các bảng CSV còn phân tầng từng nhóm theo horizon và đủ 11 confidence levels (10%–98%). WIS theo cùng strata ở metrics.csv; coverage/gap, lower/upper misses, MPIW, CI và số mẫu ở calibration.csv. Gap >0 là overcoverage, <0 là undercoverage trên mẫu đánh giá, không phải kiểm định nhân quả.

![Stratified coverage](figures/stratified_coverage95.png)

## Khoảng rộng hơn có báo hiệu khó dự báo hơn?

{table(correlations, ['horizon', 'analysis', 'rho', 'lo', 'hi'])}

Sau chuẩn hóa, width vẫn đồng biến với error; sau điều chỉnh context, partial rho chỉ còn **{adjusted.rho.iloc[0]:.3f}–{adjusted.rho.iloc[-1]:.3f}** qua horizons 1–4. Width chứa một phần thông tin về độ khó, nhưng quan hệ raw mạnh hơn nhiều do khác biệt quy mô. Khoảng rộng không tự bảo đảm đủ coverage, và không dùng kết quả này như chứng nhận từng forecast đáng tin.

Raw width/error cùng tăng theo quy mô bang nên cần xem normalized và context-adjusted. Normalization chỉ dùng mean last4 của input tại origin. Partial Spearman lấy residual của ranks sau điều chỉnh origin level, growth, volatility, history length và phase; đây là chẩn đoán hồi cứu, không phải mô hình risk đã được xác nhận ngoài mẫu. Không gộp bốn horizons thành bốn quan sát độc lập của cùng một origin.

![Width and error](figures/width_error.png)

## Thêm dữ liệu dispersion làm khoảng thay đổi ra sao?

Toàn bộ **{len(data):,}/{len(data):,} fits đã dùng MLE** từ origin đầu tiên nhờ 24 warmup origins. Số eligible pairs tăng từ {data.calibration_n.min()} đến {data.calibration_n.max()}; không có scored initial-size phase để so sánh initial→MLE. Calibration counts, negative exclusions và future counts đều được dựng lại từ warmup/prior forecasts để kiểm tra.

**UQ policy `matured`**: {future:,} fits có target chưa đến hạn; {int(data.future_calibration_n.sum()):,} lượt sử dụng future targets. Từng calibration count, negative exclusion và NB size đã được dựng lại với forecast_origin < current origin và target_end_date ≤ current origin. Frozen warmup dùng snapshot 15/08/2020 và labels vẫn là final revisions: không được gọi toàn bộ backtest là causal hoặc real-time.

NB trước median-recentering có Var(Y)=μ+μ²/size. Size nhỏ làm relative dispersion lớn hơn ở cùng μ; width còn phụ thuộc point forecast và Monte Carlo draws. Hình và bảng calibration_band mô tả đồng biến qua thời gian, **không chứng minh thêm calibration data tự nó làm interval tốt hơn/hẹp hơn**. Cần controlled ablation cùng origins/means nếu muốn kết luận đó; E6 không thay UQ gốc để tạo một so sánh bị confound.

Thực tế coverage nhóm 24–49 pairs là **{first_n:.2%}**, nhóm ≥100 pairs là **{last_n:.2%}**. Vì đây cũng là hai giai đoạn lịch khác nhau, số pairs tăng không đi cùng cải thiện calibration một cách đơn điệu trong run này; không thể kết luận nguyên nhân là việc tăng pairs.

![Dispersion history](figures/dispersion_history.png)
![Fixed example](figures/forecast_intervals.png)

## Giới hạn và tái lập

Đây là evaluation của Python reproduction trên dữ liệu gốc đã chuẩn bị, không phải kiểm định numerical parity tuyệt đối với R/paper. Policy `matured` không đưa targets chưa đến hạn vào MLE; frozen warmup và final revisions vẫn là giới hạn hồi cứu. Giữ raw truth, kể cả {int(data.negative_actual.sum())} targets âm; interval bounds sau recentering cũng có thể âm và không bị clip. CI dùng {BOOTSTRAPS} lần bootstrap nguyên bang, seed {SEED}; giữ phụ thuộc theo thời gian trong bang nhưng không mô hình hóa hết đồng biến giữa bang, không hiệu chỉnh multiple comparisons. Bins hiếm có thể không ổn định; luôn đọc số bang/mẫu và CI.

Input checksums, source code và quy tắc nằm trong protocol.json; summary.json ghi kiểm chứng, software và output checksums. instances.csv.gz chứa từng target, 23 quantiles, origin features, NB size, calibration counts và policy. Notebook chỉ đọc kết quả, không sinh synthetic hoặc chạy lại BO.
'''
    (out/'report.md').write_text(report)


def run(args):
    start = time.perf_counter()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(out/'run.lock')
    if (out/'protocol.json').exists():raise ValueError('Use a fresh E6 output directory; do not overwrite or resume older results')
    inputs = {}
    # Freeze the specification and implementation before any E6 scoring.
    protocol = dict(version=VERSION,window_filter=WINDOW_FILTER,uq_policy=UQ_POLICY,
        reference=str(args.source.resolve()),rules=RULES, created_utc=datetime.now(timezone.utc).isoformat(),
        source={str(Path(__file__).relative_to(ROOT)):Path(__file__).read_text()}, input_sha256=inputs)
    atomic_json(out/'protocol.json', protocol)
    data = load_reference(args.source.resolve(), args.archive.resolve(), inputs)
    for path in [Path(__file__),ROOT/'experiments/runtime.py',ROOT/'experiments/error_analysis/conditions.py', ROOT/'experiments/forecasting/uncertainty.py']:
        inputs[str(path.relative_to(ROOT))] = file_hash(path)
    protocol['input_sha256'] = inputs
    atomic_json(out/'protocol.json', protocol)
    print('Verified 22400 original forecasts, all quantiles, WIS and calibration counts', flush=True)
    data.to_csv(out/'instances.csv.gz', index=False)
    metrics, calibration = summarize(data)
    metrics.to_csv(out/'metrics.csv', index=False)
    calibration.to_csv(out/'calibration.csv', index=False)
    print('Computing width/error correlations and state bootstrap', flush=True)
    correlations = width_correlations(data)
    correlations.to_csv(out/'width_correlations.csv', index=False)
    make_figures(out, data, metrics, calibration, correlations)
    write_report(out, data, metrics, calibration, correlations)
    if any(file_hash(ROOT/p if not Path(p).is_absolute() else Path(p)) != digest for p, digest in inputs.items()):
        raise RuntimeError('An input changed during analysis')
    overall = calibration.loc[calibration.dimension.eq('overall') & calibration.level.eq(.95)].iloc[0]
    atomic_json(out/'summary.json', dict(status='complete', forecasts=len(data), cases=len(data)//4, states=50, origins=112,
        version=VERSION,window_filter=WINDOW_FILTER,uq_policy=UQ_POLICY,reference=str(args.source.resolve()),
        protocol_sha256=file_hash(out/'protocol.json'), nb_sizes_verified=len(data),full_evaluation_matched=True,
        confidence_levels=11, quantiles_verified=len(data)*23, inputs_unchanged=True,
        WIS=float(data.wis.mean()), MAE=float(data.absolute_error.mean()), coverage95=float(overall.coverage),
        MPIW95=float(overall.MPIW), source_coverage95=float(overall.source_coverage),
        inclusive_boundary_additions95=int(overall.boundary_additions),future_target_uses=int(data.future_calibration_n.sum()),
        future_label_fits=int(data.future_calibration_n.gt(0).sum()), empty_groups=['history_stage: early <=26 weeks', 'dispersion_method: initial'],
        software={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','matplotlib']},
        wall_seconds=time.perf_counter()-start, code_sha256=file_hash(Path(__file__)),
        artifacts_sha256={str(p.relative_to(out)):file_hash(p) for p in [*out.glob('*.csv'), *out.glob('*.csv.gz'), out/'report.md', *out.joinpath('figures').glob('*')]}))
    print('Complete:', out/'report.md')


def validate_outputs(out):
    """Require current source, matured provenance and verified complete artifacts before display."""
    out=Path(out);summary=json.loads((out/'summary.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    if (summary.get('status')!='complete' or summary.get('version')!=VERSION
        or protocol.get('version')!=VERSION or summary.get('window_filter')!=WINDOW_FILTER
        or protocol.get('window_filter')!=WINDOW_FILTER or summary.get('uq_policy')!=UQ_POLICY
        or protocol.get('uq_policy')!=UQ_POLICY or summary.get('forecasts')!=22400
        or summary.get('nb_sizes_verified')!=22400 or summary.get('quantiles_verified')!=515200
        or summary.get('future_label_fits')!=0 or summary.get('future_target_uses')!=0
        or not summary.get('inputs_unchanged') or not summary.get('full_evaluation_matched')):
        raise ValueError('Incomplete or stale E6 result/policy/filter/verification')
    if summary['code_sha256']!=file_hash(Path(__file__)) or summary['protocol_sha256']!=file_hash(out/'protocol.json'):
        raise ValueError('E6 source/protocol changed')
    for name,sha in protocol['input_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if file_hash(path)!=sha:raise ValueError(f'E6 input/source changed: {name}')
    for name,sha in summary['artifacts_sha256'].items():
        if file_hash(out/name)!=sha:raise ValueError(f'E6 artifact changed: {name}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/paper_full_nonconstant_v2_matured')
    parser.add_argument('--archive', type=Path, default=ROOT/'outputs/paper_weekly_archive')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/experiment_6_nonconstant_v2_matured')
    parser.add_argument('--execute-notebook', action='store_true')
    args = parser.parse_args()
    if args.execute_notebook:
        from notebooks import execute_notebook
        execute_notebook(Path(__file__).with_suffix('.ipynb'), working_dir=ROOT, timeout=600)
    else: run(args)
