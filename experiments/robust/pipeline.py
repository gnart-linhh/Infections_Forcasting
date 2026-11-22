"""Cold-start forecasts, synthetic-only selection, and separated external scoring."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'infections-robust-matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd

from experiments.robust.data import (ROOT, resolve, array_hash, load_reference, load_real_cases,
                                     load_real_outcomes, attach_outcomes)
from experiments.robust.method import MODELS
from experiments.robust.tuning import generate_independent, synthetic_cases, select_parameters
from experiments.robust.evaluation import paired_metrics, cluster_comparisons, uncertainty_metrics
from experiments.runtime import atomic_json, file_hash, acquire_run_lock

EXPERIMENTS = ['validation', 'early', 'ablation', 'robustness', 'probabilistic', 'xai', 'all', 'library', 'policy']
QCOLS = ['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975']


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['history_lengths'] != [6, 8, 11] or cfg['forecast_horizons'] != [1, 2, 3, 4]:
        raise ValueError('Primary study requires histories 6/8/11 and horizons 1-4')
    if cfg['reference']['smoothing'] != 'gam' or cfg['reference']['closest'] != 4422:
        raise ValueError('Keep the original GAM and 4422-neighbour baseline unchanged')
    if not cfg['weighting']['temperatures'] or any(not np.isfinite(t) or t <= 0 for t in cfg['weighting']['temperatures']):
        raise ValueError('Temperature coefficients must be positive and finite')
    if 0 not in cfg['regularization']['betas'] or any(not np.isfinite(b) or b < 0 for b in cfg['regularization']['betas']):
        raise ValueError('Beta candidates must be nonnegative, finite, and include zero')
    if cfg['evaluation']['design'] not in ['early', 'fixed_origin']:
        raise ValueError('Choose one separate early or fixed-origin design')
    for role in ['validation', 'background', 'check']:
        if cfg['synthetic'][role+'_per_seed_family'] < 1:
            raise ValueError('Synthetic role sizes must be positive')
    if len(cfg['synthetic']['seeds']) < 2 or len(set(cfg['synthetic']['seeds'])) != len(cfg['synthetic']['seeds']):
        raise ValueError('Need multiple unique independent synthetic seeds')
    if cfg['xai']['horizon'] not in cfg['forecast_horizons'] or cfg['xai']['max_cases'] < 4:
        raise ValueError('XAI needs four representative cases and a valid horizon')
    return cfg


def forecast_cases(engine, cases, frozen, seed=42, draws=5000):
    """Outcome-free inference: only permitted histories, frozen reference/parameters.

    Initial NB sizes and the original simulated median-recentring are retained.
    No fit_size(), COVID errors, vintage refresh, or online update is used.
    """
    from experiments.forecasting.uncertainty import interval_quantiles, INITIAL_SIZES
    rows, matches = [], []
    for case in cases:
        values, diagnostic = engine.variants(case.history, frozen['temperature'], frozen['beta'])
        matched = engine.match(case.history)
        scale = diagnostic['scale']
        z = np.maximum(case.history, 1)
        growth = (z[-2:].mean()+1)/(z[-4:-2].mean()+1)-1
        phase = 'rising' if growth > .1 else 'declining' if growth < -.1 else 'stable'
        volatility = float(np.std(np.diff(z))/scale)
        key = int(array_hash(case.history)[:8], 16)
        for model, point in values.items():
            for h, value in enumerate(point, 1):
                target = (f'step_{case.reference_date+h}' if isinstance(case.reference_date, int)
                          else (pd.Timestamp(case.reference_date)+pd.Timedelta(weeks=h)).date().isoformat())
                q = interval_quantiles(float(value), INITIAL_SIZES[h-1],
                                       np.random.default_rng(np.random.SeedSequence([seed, key, h])),
                                       draws=draws, initial=True)
                rows.append(dict(case_id=case.case_id, dataset=case.dataset, region=case.region,
                                 forecast_origin=str(case.issuance_date), reference_date=str(case.reference_date),
                                 target_date=target, horizon=h, history_length=len(z), design=case.design,
                                 model=model, prediction=float(value), phase=phase, growth_rate=float(growth), volatility=volatility,
                                 high_volatility=volatility > .5, dispersion=float(diagnostic['dispersion'][h-1]),
                                 shrinkage=float(diagnostic['shrinkage'][h-1]),
                                 effective_analogues=diagnostic['effective_analogues'],
                                 minimum_distance=diagnostic['minimum_distance'], scale=scale,
                                 history_sha256=array_hash(case.history),
                                 uncertainty='original_nb_initial' if model == 'M0' else 'modified_point_nb_initial',
                                 **dict(zip(QCOLS, map(float, q)))))
        increments = engine.x[matched.indices[:8]]
        past = np.column_stack([np.zeros(len(increments)), np.cumsum(increments, axis=1)])
        past = past-past[:, -1:]+z[-1]
        paths = np.column_stack([past, np.maximum(1, z[-1]+np.cumsum(matched.future[:8], axis=1))])
        matches.append(dict(case_id=case.case_id, history=case.history.tolist(),
                            embedding_rows=matched.indices[:8].tolist(), distances=matched.distances[:8].tolist(),
                            analogue_paths=paths.tolist()))
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(['case_id', 'horizon', 'model']).any() or 'actual' in frame:
        raise ValueError('Invalid issued forecasts')
    if not frame.groupby(['case_id', 'horizon']).model.nunique().eq(4).all():
        raise ValueError('Four variants must cover identical targets')
    return frame, matches


def save_tables(predictions, out, config):
    paired_metrics(predictions).to_csv(out/'metrics.csv', index=False)
    for name, dimensions in [('horizon', ('dataset', 'horizon')),
                              ('history', ('dataset', 'history_length')),
                              ('history_horizon', ('dataset', 'history_length', 'horizon'))]:
        paired_metrics(predictions, dimensions).to_csv(out/f'metrics_by_{name}.csv', index=False)
    paired_metrics(predictions).assign(weighted=lambda d: d.model.isin(['M1', 'M3']),
                                       regularized=lambda d: d.model.isin(['M2', 'M3'])).to_csv(out/'ablation.csv', index=False)
    robust = []
    for dimension in ['phase', 'high_volatility', 'horizon']:
        table = paired_metrics(predictions, ('dataset', dimension)).rename(columns={dimension: 'stratum'})
        table['dimension'] = dimension
        robust.append(table)
    pd.concat(robust, ignore_index=True).to_csv(out/'robustness.csv', index=False)
    cluster_comparisons(predictions, config['seed'], config['evaluation']['bootstrap_draws']).to_csv(out/'paired_comparisons.csv', index=False)
    uncertainty_metrics(predictions).to_csv(out/'uncertainty_metrics.csv', index=False)
    uncertainty_metrics(predictions, ('dataset', 'model', 'history_length', 'horizon')).to_csv(out/'uncertainty_by_history_horizon.csv', index=False)
    audited = predictions.assign(invalid_prediction=~np.isfinite(predictions.prediction) | (predictions.prediction < 0))
    counts = audited.groupby(['dataset', 'model'], as_index=False).agg(
        forecasts=('prediction', 'size'), cases=('case_id', 'nunique'), regions_or_curves=('region', 'nunique'),
        invalid_predictions=('invalid_prediction', 'sum'))
    expected = predictions.groupby('dataset').case_id.nunique()*len(config['forecast_horizons'])
    counts['missing_predictions'] = counts.dataset.map(expected)-counts.forecasts
    counts['failures'] = counts.missing_predictions+counts.invalid_predictions
    counts.to_csv(out/'availability.csv', index=False)
    comparison_plot(predictions, out/'comparison.png')


def comparison_plot(predictions, path):
    import matplotlib.pyplot as plt
    table = paired_metrics(predictions, ('dataset', 'history_length'))
    datasets = list(table.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for model, block in table.loc[table.dataset.eq(dataset)].groupby('model'):
            ax.plot(block.history_length, block.mae, 'o-', label=model)
        ax.set(title=dataset, xlabel='Permitted weekly observations', ylabel='Pooled MAE', xticks=[6, 8, 11])
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def regenerate(out):
    out = Path(out)
    cfg = json.loads((out/'run_config.json').read_text())['config']
    save_tables(pd.read_csv(out/'predictions.csv.gz'), out, cfg)
    if (out/'explanation_metadata.json').exists():
        from experiments.robust.explain import render_explanations
        render_explanations(out)


def run(config, experiment='all', smoke=False, output_dir=None):
    if experiment == 'policy':
        if output_dir is not None:raise ValueError('Policy studies use their configured immutable root')
        from experiments.robust.policy_experiment import run_policy
        return run_policy(config,smoke)
    if experiment == 'library':
        if output_dir is not None:
            raise ValueError('Library studies use their immutable configured output root')
        from experiments.robust.library_study import run_library_study
        return run_library_study(config,smoke)
    if experiment not in EXPERIMENTS:
        raise ValueError('Unknown experiment')
    started = time.perf_counter()
    root = resolve(config['output_root'])
    engine, reference_hashes, provenance = load_reference(config)
    synthetic, synthetic_folder = generate_independent(config, reference_hashes, provenance, root, smoke)
    validation_cases, validation_outcomes = synthetic_cases(synthetic, 'validation', config['history_lengths'])
    selection_signature = dict(config={key: config[key] for key in ['seed', 'history_lengths', 'reference', 'weighting', 'regularization', 'synthetic', 'validation']},
                               reference=provenance, synthetic_identity=synthetic['identity'])
    source = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    with acquire_run_lock(synthetic_folder/'selection.lock'):
        selection_id = hashlib.sha256(json.dumps([selection_signature, source['method.py']], sort_keys=True).encode()).hexdigest()[:16]
        frozen_path = synthetic_folder/f'selected_{selection_id}.json'
        validation_path = synthetic_folder/f'validation_{selection_id}.csv'
        if frozen_path.exists():
            stored = json.loads(frozen_path.read_text())
            if stored['signature'] != selection_signature or stored['method_source'] != source['method.py']:
                raise ValueError('Frozen synthetic selection changed; use a new synthetic cache')
            frozen = stored['parameters']
            table = pd.read_csv(validation_path)
        else:
            frozen, table = select_parameters(engine, validation_cases, validation_outcomes, config, provenance)
            atomic_json(frozen_path, dict(parameters=frozen, signature=selection_signature, method_source=source['method.py']))
            table.to_csv(validation_path, index=False)
    identity = dict(config=config, smoke=smoke, source_sha256=source, reference=provenance,
                    external_input_sha256=None if experiment == 'validation' else file_hash(resolve(config['evaluation']['database'])),
                    synthetic_folder=str(synthetic_folder), packages={n: importlib.metadata.version(n)
                    for n in ['numpy', 'scipy', 'pandas', 'matplotlib', 'shap', 'scikit-learn']})
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = Path(output_dir) if output_dir else root/(('smoke_' if smoke else 'full_')+digest)
    out.mkdir(parents=True, exist_ok=True)
    with acquire_run_lock(out/'run.lock'):
        if (out/'run_config.json').exists() and json.loads((out/'run_config.json').read_text()) != identity:
            raise ValueError('Output identity differs; use a new directory')
        atomic_json(out/'run_config.json', identity)
        # The frozen parameters are persisted BEFORE even opening real-case data.
        atomic_json(out/'selected_config.json', frozen)
        table.to_csv(out/'validation_metrics.csv', index=False)
        atomic_json(out/'status.json', dict(status='synthetic_validation_complete', experiment=experiment))
        if experiment == 'validation':
            return out
        real_cases = load_real_cases(config, smoke)
        check_cases, check_outcomes = synthetic_cases(synthetic, 'check', config['history_lengths'])
        cases = real_cases+check_cases
        tick = time.perf_counter()
        issued, diagnostics = forecast_cases(engine, cases, frozen, config['seed'], config['uncertainty']['draws'])
        issued.to_csv(out/'issued_predictions.csv.gz', index=False)
        atomic_json(out/'matching_diagnostics.json', diagnostics)
        inference_seconds = time.perf_counter()-tick
        # Future external outcomes enter for the first time here, after issuance.
        external = load_real_outcomes(config, real_cases)
        outcomes = pd.concat([external, check_outcomes], ignore_index=True)
        predictions = attach_outcomes(issued, outcomes)
        predictions.to_csv(out/'predictions.csv.gz', index=False)
        save_tables(predictions, out, config)
        xai_seconds = None
        if experiment in ['xai', 'all']:
            from experiments.robust.explain import explain_experiment
            tick = time.perf_counter()
            explain_experiment(engine, real_cases, predictions, frozen, synthetic, config, out)
            xai_seconds = time.perf_counter()-tick
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system() == 'Darwin' else 1024)
        atomic_json(out/'runtime.json', dict(total_seconds=time.perf_counter()-started,
                    inference_and_uq_seconds=inference_seconds, xai_seconds=xai_seconds, peak_rss_mb=peak,
                    actual_match_calls=engine.match_calls, forecast_cases=len(cases)))
        atomic_json(out/'status.json', dict(status='complete', experiment=experiment, smoke=smoke,
                    real_cases=len(real_cases), synthetic_check_cases=len(check_cases),
                    forecasts_per_model=len(predictions)//4, target_disease_training=False))
        print('Complete:', out, flush=True)
    return out
