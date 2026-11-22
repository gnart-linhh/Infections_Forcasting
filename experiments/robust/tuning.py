"""Independent whole-curve synthetic validation; no target-disease inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from experiments.robust.data import ForecastCase, array_hash, ROOT
from experiments.runtime import atomic_json, file_hash, acquire_run_lock

if str(ROOT/'experiments/synthetic') not in sys.path:
    sys.path.insert(0, str(ROOT/'experiments/synthetic'))
from generator import gen_curve
from bank import FAMILIES, save_packed, load_packed

NAMESPACE = 3187361291  # distinct from all scalar reference-bank seed namespaces
ROLES = {'validation': 0, 'background': 1, 'check': 2}


def eligible(curve):
    if curve is None or curve.time_cadence != 'weekly' or curve.scale != 'counts':
        return None
    z = np.asarray(curve.ts, float)
    onset = np.flatnonzero(z >= 10)
    if not len(onset) or not np.isfinite(z).all() or (z < 0).any():
        return None
    start = int(onset[0])
    return start if len(z)-start >= 15 else None


def generate_independent(config, reference_hashes, reference_provenance, root, smoke=False,
                         role_counts=None, namespace=NAMESPACE):
    """Curve-level separation, independent SeedSequence roles, exact content audit."""
    roles = ROLES if role_counts is None else {name:i for i,name in enumerate(role_counts)}
    counts = ({name: config['synthetic'][name+'_per_seed_family'] for name in roles}
              if role_counts is None else dict(role_counts))
    if smoke:
        counts = {name: min(count, config['smoke']['curves_per_seed_family']) for name, count in counts.items()}
    identity = dict(seed=config['seed'], seeds=config['synthetic']['seeds'], counts=counts,
                    attempts=config['synthetic']['max_attempts_per_cell'], namespace=namespace,
                    generator=file_hash(ROOT/'experiments/synthetic/generator.py'),
                    implementation=file_hash(Path(__file__)), reference=reference_provenance['library_sha256'])
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(root)/('smoke_' if smoke else 'full_')/('synthetic_'+digest)
    folder.mkdir(parents=True, exist_ok=True)
    with acquire_run_lock(folder/'run.lock'):
        if (folder/'metadata.json').exists():
            meta = json.loads((folder/'metadata.json').read_text())
            if meta['identity'] != identity:
                raise ValueError('Synthetic cache configuration mismatch')
            for role in roles:
                if file_hash(folder/(role+'.npz')) != meta['artifacts'][role]:
                    raise ValueError('Modified synthetic cache')
                meta[role] = load_packed(folder/(role+'.npz'))
            verify_disjoint(meta, reference_hashes, roles, namespace)
            return meta, folder

        audit, metadata, used = [], {}, set(reference_hashes)
        for role, role_number in roles.items():
            curves, records = [], []
            for replicate in config['synthetic']['seeds']:
                for family_index, family in enumerate(FAMILIES):
                    accepted = 0
                    for attempt in range(config['synthetic']['max_attempts_per_cell']):
                        key = [config['seed'], namespace, role_number, replicate, family_index, attempt]
                        curve = gen_curve(family, np.random.default_rng(np.random.SeedSequence(key)))
                        onset = eligible(curve)
                        content = array_hash(curve.ts) if curve is not None else None
                        reason = 'ineligible' if onset is None else 'duplicate' if content in used else 'accepted'
                        audit.append(dict(role=role, key=key, reason=reason, sha256=content))
                        if reason != 'accepted':
                            continue
                        used.add(content)
                        records.append(dict(curve_id=f'{role}_{replicate}_{family_index}_{attempt}',
                                            seed_key=key, family=family, sha256=content, onset=onset,
                                            cadence=curve.time_cadence, units=curve.scale))
                        curves.append(curve)
                        accepted += 1
                        if accepted == counts[role]:
                            break
                    if accepted != counts[role]:
                        raise RuntimeError(f'Insufficient eligible independent curves: {role}/{replicate}/{family}')
            save_packed(curves, folder/(role+'.npz'))
            metadata[role+'_records'] = records
            metadata[role] = curves
            print(f'Generated independent {role}: {len(curves)} whole weekly count curves', flush=True)
        verify_disjoint(metadata, reference_hashes, roles, namespace)
        pd.DataFrame(audit).to_json(folder/'generation_audit.jsonl.gz', orient='records', lines=True)
        saved = {k: v for k, v in metadata.items() if k.endswith('_records')}
        saved.update(identity=identity, reference_overlap=0, role_overlap=0,
                     artifacts={role: file_hash(folder/(role+'.npz')) for role in roles})
        atomic_json(folder/'metadata.json', saved)
        metadata.update(saved)
        return metadata, folder


def verify_disjoint(metadata, reference_hashes, roles=ROLES, namespace=NAMESPACE):
    used = set(reference_hashes)
    for role in roles:
        curves, records = metadata[role], metadata[role+'_records']
        if len(curves) != len(records):
            raise ValueError('Synthetic provenance count mismatch')
        for curve, record in zip(curves, records):
            digest = array_hash(curve.ts)
            if digest != record['sha256'] or digest in used:
                raise ValueError('Synthetic curves overlap reference/other roles or changed content')
            if record['seed_key'][1:3] != [namespace, roles[role]]:
                raise ValueError('Invalid independent RNG namespace')
            used.add(digest)


def synthetic_cases(metadata, role, history_lengths):
    cases, observations = [], []
    for curve, record in zip(metadata[role], metadata[role+'_records']):
        start = record['onset']
        for length in history_lengths:
            end = start+length-1
            ident = record['curve_id']+f'_T{length}'
            cases.append(ForecastCase(ident, 'synthetic_'+role, record['curve_id'], curve.ts[start:end+1],
                                       end+1, end+1, 'early'))
            for horizon in range(1, 5):
                observations.append(dict(case_id=ident, horizon=horizon, target_date=f'step_{end+1+horizon}',
                                         actual=float(curve.ts[end+horizon])))
    return cases, pd.DataFrame(observations)


def select_parameters(engine, cases, observations, config, provenance):
    """Only synthetic cases are accepted; selection cannot open a real dataset."""
    if any(c.dataset != 'synthetic_validation' for c in cases):
        raise ValueError('Hyperparameter selection requires independent synthetic validation cases')
    truth = observations.set_index(['case_id', 'horizon']).actual
    actuals, forecasts = [], {}
    candidates = [(t, b) for t in config['weighting']['temperatures'] for b in config['regularization']['betas']]
    for temperature, beta in candidates:
        forecasts[(temperature, beta)] = []
    baseline = []
    for case in cases:
        target = truth.loc[[(case.case_id, h) for h in range(1, 5)]].to_numpy(float)
        actuals.extend(target)
        for temperature, beta in candidates:
            values, _ = engine.variants(case.history, temperature, beta)
            forecasts[(temperature, beta)].extend(values['M3'])
        baseline.extend(values['M0'])
    actuals = np.asarray(actuals)
    rows = []
    for (temperature, beta), point in forecasts.items():
        error = np.asarray(point)-actuals
        rows.append(dict(temperature=temperature, beta=beta, n=len(error),
                         curves=len({c.region for c in cases}), mae=float(np.abs(error).mean()),
                         rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()),
                         severe_underprediction=float(np.mean(np.asarray(point) < .5*actuals))))
    table = pd.DataFrame(rows)
    # Predeclared safeguards compare candidate RMSE and severe underprediction
    # with the original on the same synthetic cases, never with external errors.
    base_error = np.asarray(baseline)-actuals
    base_rmse = float(np.sqrt(np.mean(base_error**2)))
    base_under = float(np.mean(np.asarray(baseline) < .5*actuals))
    safe = table.loc[(table.rmse <= base_rmse*(1+config['validation']['rmse_tolerance']))
                     & (table.severe_underprediction <= base_under+config['validation']['underprediction_tolerance'])]
    safeguard_fallback = safe.empty
    if safe.empty:
        # Explicit fallback: freeze beta=0 rather than choose a rule that violates
        # the prespecified shrinkage safeguards; weighting is still investigated.
        safe = table.loc[table.beta.eq(0)]
    best = float(safe.mae.min())
    near = safe.loc[safe.mae <= best*(1+config['validation']['simplicity_tolerance'])]
    selected = near.sort_values(['beta', 'temperature'], ascending=[True, False]).iloc[0]
    frozen = dict(temperature=float(selected.temperature), beta=float(selected.beta),
                  closest=engine.closest, selected_on='independent_synthetic_curves_only',
                  selection_metric='pooled_mae', selected_metrics=selected.to_dict(),
                  original_validation_mae=float(np.abs(base_error).mean()),
                  original_validation_rmse=base_rmse, original_severe_underprediction=base_under,
                  safeguard_fallback=bool(safeguard_fallback), validation_cases=len(cases),
                  validation_curves=len({c.region for c in cases}), reference=provenance,
                  policy=config['validation'], seed=config['seed'])
    return frozen, table
