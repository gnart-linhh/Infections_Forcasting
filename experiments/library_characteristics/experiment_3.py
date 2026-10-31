"""E3: original-library family ablations with curve-count and window-count controls."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments'))
sys.path.insert(0, str(ROOT/'experiments/forecasting'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'infections-e3-matplotlib'))
import numpy as np
import pandas as pd
from runtime import atomic_json, file_hash, acquire_run_lock
from neighbors import ExactNeighbors
from experiments.sensitivity.library import embed
from bank import FAMILIES, load_packed
from experiments.exploration.synthetic_eda import curve_metrics
from experiments.error_analysis.conditions import features

LIBRARIES = ['rollercoaster', 'wiggle', 'seasonal', 'mixed']
REGIMES = ['full', 'equal_curves', 'equal_windows']
SEEDS = [0, 1, 2]
N, K, M, H = 18387, 5, 4422, 4
CURVES, WINDOWS = 15000, 3000000
DRAW_SEED = 20261103
KEYS = ['series', 'forecast_origin', 'target_end_date', 'horizon']
RULES = dict(
    parameters=dict(attempts_per_family=N, k=K, m=M, h=H, smoothing='gam', bridge='omit'),
    libraries=LIBRARIES, regimes=REGIMES, seeds=SEEDS, generator_offsets=[0, 1000000, 2000000],
    cohort='unchanged E2/E4: all 50 states, 50 scheduled origins, 4 horizons; 10000 targets per bank/config',
    full='all accepted curves from original 18387 attempts/family; mixed is the unchanged three-family bank',
    equal_curves='15000 accepted curves per library; mixed = 5000 per family, nested in each single-family selection',
    equal_windows='3000000 valid windows per library; mixed = 1000000 per family, nested in single-family window selections',
    sampling='uniform without replacement, independent of COVID features/outcomes; sorted original source order for deterministic ties',
    sampling_rng=f'NumPy SeedSequence([{DRAW_SEED},bank_seed,family_index,kind]); kind=0 curves, kind=1 windows',
    matching='original exact L1; median future increments; cumulative sum + last raw input floored at 1; prediction floor 1',
    distance='minimum L1 / (5 * max(mean(last 6 origin-input counts floored at 1),1)); median top-m also reported',
    normalized_error='absolute error / max(mean(last 4 origin-input counts floored at 1),1)',
    phase='origin-only last2 versus previous2 means, pseudocount 1; rising >10%, declining <-10%, stable otherwise',
    case_level='origin-only last4 mean: low <1000, medium <10000, high >=10000',
    aggregation='MAE and RMSE per seed; primary summary = mean and sample SD of seed-specific metrics, not pooled RMSE',
    paired='each single family minus same-regime mixed, average paired errors over three banks; whole-state bootstrap 500 draws',
    morphology='existing EDA peak rule: centered median3, prominence 10% amplitude, separation 4 original steps; cadence stratified',
    constant_windows='preserve original np.var(ddof=1)>0 criterion, including floating-point retention of exactly constant histories; quantify per library',
    scope='point forecasting only; no BO, UQ, new generator or COVID-driven model selection',
    interpretation='three seeds are descriptive; matched regimes vary generator AND independent subsampling randomness; no claim of infinite-seed stability',
)


def select_curves(family_codes, seed, library):
    """Uniform accepted-curve control; mixed quotas are subsets of the single selections."""
    result = []
    for family in range(3):
        if library != 'mixed' and LIBRARIES[family] != library:
            continue
        candidates = np.flatnonzero(np.asarray(family_codes) == family)
        if len(candidates) < CURVES:
            raise ValueError('Not enough accepted curves for prespecified control')
        rng = np.random.default_rng(np.random.SeedSequence([DRAW_SEED, seed, family, 0]))
        chosen = rng.choice(candidates, CURVES, replace=False)
        result.extend(chosen[:CURVES//3] if library == 'mixed' else chosen)
    return np.sort(np.asarray(result, dtype=np.int64))


def select_windows(curve_codes, seed, library):
    result = []
    family_codes = (np.asarray(curve_codes, dtype=np.int64)-1)//N
    for family in range(3):
        if library != 'mixed' and LIBRARIES[family] != library:
            continue
        candidates = np.flatnonzero(family_codes == family)
        if len(candidates) < WINDOWS:
            raise ValueError('Not enough valid windows for prespecified control')
        rng = np.random.default_rng(np.random.SeedSequence([DRAW_SEED, seed, family, 1]))
        chosen = rng.choice(len(candidates), WINDOWS, replace=False)
        result.append(candidates[chosen[:WINDOWS//3] if library == 'mixed' else chosen])
    return np.sort(np.concatenate(result))


def aggregate(last, future):
    return np.maximum(1, last+np.cumsum(np.median(future, axis=0)))


def engine_signature():
    tree = ast.parse(Path(__file__).read_text())
    names = ['select_curves', 'select_windows', 'aggregate', 'worker', 'save_run', 'replay_full', 'prepare_cases', 'curve_statistics']
    content = [ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    return hashlib.sha256(json.dumps([RULES, content], sort_keys=True).encode()).hexdigest()


def connect(out):
    db = sqlite3.connect(Path(out)/'results.sqlite', timeout=60)
    db.execute('PRAGMA foreign_keys=ON')
    return db


def initialize(db):
    db.executescript('''
      CREATE TABLE IF NOT EXISTS runs (
        seed INTEGER, regime TEXT, library TEXT, metadata TEXT NOT NULL,
        PRIMARY KEY(seed,regime,library));
      CREATE TABLE IF NOT EXISTS predictions (
        seed INTEGER, regime TEXT, library TEXT, case_id INTEGER, horizon INTEGER, prediction REAL NOT NULL,
        PRIMARY KEY(seed,regime,library,case_id,horizon));
      CREATE TABLE IF NOT EXISTS matches (
        seed INTEGER, regime TEXT, library TEXT, case_id INTEGER,
        minimum_distance REAL, median_distance REAL, maximum_distance REAL,
        PRIMARY KEY(seed,regime,library,case_id));
      CREATE TABLE IF NOT EXISTS curve_stats (
        seed INTEGER, curve_id INTEGER, family TEXT, cadence TEXT, scale TEXT,
        observations INTEGER, maximum REAL, peaks INTEGER, peaks_per_100_steps REAL,
        cv REAL, relative_change REAL, relative_volatility REAL,
        valid_windows INTEGER, constant_windows INTEGER,
        PRIMARY KEY(seed,curve_id));
    ''')


def save_run(db, seed, regime, library, predictions, matches, metadata):
    if len(predictions) != 10000 or len(matches) != 2500:
        raise ValueError('Every completed run must retain the entire shared target cohort')
    if not np.isfinite(np.asarray(predictions, float)).all() or not np.isfinite(np.asarray(matches, float)).all():
        raise ValueError('Nonfinite forecast or distance')
    if any(row[2] < 1 for row in predictions) or any(not (0 <= row[1] <= row[2] <= row[3]) for row in matches):
        raise ValueError('Invalid point forecast or distance ordering')
    with db:
        db.executemany('INSERT INTO predictions VALUES (?,?,?,?,?,?)',
                       [(seed, regime, library, *row) for row in predictions])
        db.executemany('INSERT INTO matches VALUES (?,?,?,?,?,?,?)',
                       [(seed, regime, library, *row) for row in matches])
        db.execute('INSERT INTO runs VALUES (?,?,?,?)', (seed, regime, library, json.dumps(metadata)))


def load_bank(e2, seed):
    packed = e2/f'packed/seed_{seed}'
    curves = load_packed(packed/'curves.npz')
    with np.load(packed/'attempts.npz', allow_pickle=False) as data:
        ids, family, attempts = data['curve_id'], data['family'], data['index']
    if len(curves) != len(ids) or not np.array_equal(family, (ids-1)//N) or not (attempts < N).all():
        raise ValueError('Incorrect original bank namespace')
    if any(curve.curve_type != FAMILIES[int(f)] for curve, f in zip(curves, family)):
        raise ValueError('Curve family metadata mismatch')
    return curves, ids, family


def worker(args):
    out, seed, library = args.output_dir, args.seed, args.library
    db = connect(out)
    completed = {row[0] for row in db.execute('SELECT regime FROM runs WHERE seed=? AND library=?', (seed, library))}
    wanted = [r for r in REGIMES if r not in completed]
    if not wanted:
        return
    start = time.perf_counter()
    curves, ids, families = load_bank(args.e2_dir, seed)
    if library != 'mixed':
        keep = np.flatnonzero(families == LIBRARIES.index(library))
        curves = [curves[i] for i in keep]; ids = ids[keep]
    scratch = out/'scratch'
    x, y, codes = embed(curves, ids, K, scratch, h=H)
    del curves
    expected = db.execute('SELECT SUM(valid_windows) FROM curve_stats WHERE seed=?'+
                          ('' if library == 'mixed' else ' AND family=?'),
                          (seed,) if library == 'mixed' else (seed, library)).fetchone()[0]
    if len(x) != expected:
        raise ValueError('Embedding count differs from morphology audit')
    with np.load(args.e2_dir/'contexts.npz', allow_pickle=False) as raw:
        ctx = {name:raw[name] for name in ['last', 'query_k5']}
    case_table = pd.read_sql_query('SELECT * FROM cases ORDER BY case_id', db)
    assert len(case_table) == 2500
    for regime in wanted:
        tick = time.perf_counter()
        if regime == 'full':
            selected = None; xx, yy, selected_codes = x, y, codes
        else:
            if regime == 'equal_curves':
                chosen = select_curves((ids-1)//N, seed, library)
                selected = np.flatnonzero(np.isin(codes, ids[chosen]))
            else:
                selected = select_windows(codes, seed, library)
            xx, yy, selected_codes = np.asarray(x[selected]), np.asarray(y[selected]), np.asarray(codes[selected])
        if len(xx) < M or (regime == 'equal_windows' and len(xx) != WINDOWS):
            raise ValueError('Incorrect embedding budget')
        unique_ids = np.unique(selected_codes)
        if regime == 'equal_curves' and len(unique_ids) != CURVES:
            raise ValueError('Some selected curves have no valid windows; curve control not satisfied')
        family_counts = {name:int(np.sum((unique_ids-1)//N == f)) for f, name in enumerate(LIBRARIES[:3])}
        window_counts = {name:int(np.sum((np.asarray(selected_codes, dtype=np.int64)-1)//N == f)) for f, name in enumerate(LIBRARIES[:3])}
        constant = int(np.count_nonzero(np.all(xx == 0, axis=1)))
        selection_hash = hashlib.sha256((ids.tobytes() if selected is None else np.asarray(selected, dtype='<i8').tobytes())).hexdigest()
        index = ExactNeighbors(xx, workers=1)
        predictions, matching = [], []
        for begin in range(0, 2500, 32):
            neighbors, distances = index.query(ctx['query_k5'][begin:begin+32], M)
            for j, (neighbors_j, distance) in enumerate(zip(neighbors, distances), begin):
                point = aggregate(ctx['last'][j], yy[neighbors_j])
                predictions.extend((j, h, float(p)) for h, p in enumerate(point, 1))
                matching.append((j, float(distance[0]), float(np.median(distance)), float(distance[-1])))
        metadata = dict(accepted_curves=len(unique_ids), valid_windows=len(xx), family_curves=family_counts,
            family_windows=window_counts, constant_history_windows=constant, constant_fraction=constant/len(xx),
            selection_sha256=selection_hash, reused=False, seed=seed, regime=regime, library=library,
            seconds=time.perf_counter()-tick, group_seconds=time.perf_counter()-start)
        save_run(db, seed, regime, library, predictions, matching, metadata)
        print(f'Completed seed={seed} {regime}/{library}: C={len(unique_ids):,}, M={len(xx):,}', flush=True)
        del index, xx, yy, selected_codes, unique_ids, selected
    del x, y, codes
    for name in ['x', 'y', 'curve_ids']:
        (scratch/f'{name}.dat').unlink()
    scratch.rmdir()
    db.close()


def prepare_cases(e2, panel_dir, db):
    with np.load(e2/'contexts.npz', allow_pickle=False) as data:
        ctx = {name:data[name] for name in data.files}
    if ctx['query_k5'].shape != (2500, K) or ctx['actual'].shape != (2500, H):
        raise ValueError('Require unchanged E2 contexts')
    panel = pd.read_csv(panel_dir/'panel.csv')
    panel_groups = {key:g.sort_values('target_end_date') for key,g in panel.groupby(['series', 'forecast_origin'])}
    cases, observations = [], []
    for j, (series, origin) in enumerate(zip(ctx['series'], ctx['origin'])):
        history = panel_groups[(str(series), str(origin))]
        expected = pd.date_range('2020-01-25', str(origin), freq='7D').strftime('%Y-%m-%d').tolist()
        if history.target_end_date.tolist() != expected:
            raise ValueError('Invalid historical panel')
        values = np.maximum(history.value.to_numpy(float), 1)
        f = features(history.value.to_numpy(float))
        if not np.isclose(f['normalization'], ctx['normalization'][j], rtol=0, atol=1e-10) or values[-1] != ctx['last'][j]:
            raise ValueError('Context/panel input mismatch')
        cases.append(dict(case_id=j, series=str(series), forecast_origin=str(origin),
                          level_scale=float(np.mean(values[-6:])), **f))
        for h in range(1, H+1):
            target = str(ctx['targets'][j, h-1])
            if target != (pd.Timestamp(str(origin))+pd.Timedelta(weeks=h)).date().isoformat():
                raise ValueError('Target horizon mismatch')
            observations.append(dict(case_id=j, horizon=h, target_end_date=target, actual=float(ctx['actual'][j, h-1])))
    cases = pd.DataFrame(cases); observations = pd.DataFrame(observations)
    if len(cases) != 2500 or cases.duplicated(['series', 'forecast_origin']).any():
        raise ValueError('Duplicate or missing cases')
    cases.to_sql('cases', db, if_exists='replace', index=False)
    observations.to_sql('observations', db, if_exists='replace', index=False)
    return cases


def curve_statistics(e2, seed, db):
    count = db.execute('SELECT COUNT(*) FROM curve_stats WHERE seed=?', (seed,)).fetchone()[0]
    expected = json.loads((e2/f'packed/seed_{seed}/metadata.json').read_text())['accepted']
    if count == expected:
        return
    if count:
        raise ValueError('Incomplete curve statistics')
    curves, ids, families = load_bank(e2, seed)
    rows = []
    for curve, ident, family in zip(curves, ids, families):
        v = np.maximum(curve.ts, 1e-8)
        windows = np.lib.stride_tricks.sliding_window_view(v, K+H+2)[:, :K+1] if len(v) >= K+H+2 else np.empty((0, K+1))
        valid = np.var(windows, axis=1, ddof=1)>0
        constant = int(np.sum(valid & (np.ptp(windows, axis=1)==0)))
        stats = curve_metrics(curve.ts)
        rows.append((seed, int(ident), LIBRARIES[int(family)], curve.time_cadence, curve.scale,
            stats['observations'], stats['max_value'], stats['epidemic_peaks'],
            100*stats['epidemic_peaks']/max(stats['observations']-1, 1), stats['cv'],
            stats['mean_absolute_change_over_mean'], stats['change_std_over_mean'], int(valid.sum()), constant))
    with db:
        db.executemany('INSERT INTO curve_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    print(f'Morphology audited: seed={seed}, {len(rows):,} curves', flush=True)


def replay_full(e2, seed, cases, db):
    if db.execute("SELECT COUNT(*) FROM runs WHERE seed=? AND regime='full' AND library='mixed'", (seed,)).fetchone()[0]:
        return
    folder = e2/f'groups/seed_{seed}_N18387_k5'
    f = pd.read_csv(folder/'forecasts.csv.gz', float_precision='round_trip').query('m==4422').copy()
    matches = pd.read_csv(folder/'matches.csv.gz', float_precision='round_trip').query('m==4422').copy()
    case_keys = cases[['case_id', 'series', 'forecast_origin']]
    f = f.merge(case_keys, on=['series','forecast_origin'], validate='many_to_one')
    obs = pd.read_sql_query('SELECT * FROM observations', db)
    f = f.merge(obs, on=['case_id','horizon','target_end_date'], suffixes=('', '_stored'), validate='one_to_one')
    if len(f) != 10000 or not np.array_equal(f.actual, f.actual_stored):
        raise ValueError('Reference target/label mismatch')
    matches = matches.merge(case_keys, on=['series','forecast_origin'], validate='one_to_one')
    runtime = json.loads((folder/'runtime.json').read_text())
    meta = dict(accepted_curves=runtime['accepted_curves'], valid_windows=runtime['M'], reused=True,
                source=str(folder), seconds=None, seed=seed, regime='full', library='mixed')
    stats = pd.read_sql_query('SELECT family,COUNT(*) AS curves,SUM(valid_windows) AS windows,SUM(constant_windows) AS constant FROM curve_stats WHERE seed=? GROUP BY family', db, params=(seed,))
    meta.update(family_curves=stats.set_index('family').curves.astype(int).to_dict(),
        family_windows=stats.set_index('family').windows.astype(int).to_dict(),
        constant_history_windows=int(stats.constant.sum()), constant_fraction=float(stats.constant.sum()/runtime['M']))
    if stats.windows.sum() != runtime['M']:
        raise ValueError('Full mixed audit differs from existing embedding')
    save_run(db, seed, 'full', 'mixed', list(f[['case_id','horizon','prediction']].itertuples(index=False,name=None)),
             list(matches[['case_id','minimum_distance','median_distance','maximum_distance']].itertuples(index=False,name=None)), meta)


def validate_inputs(args):
    e2 = args.e2_dir
    paths = [e2/'contexts.npz', e2/'context_metadata.json', args.panel_dir/'panel.csv', args.panel_dir/'provenance.json',
             ROOT/'outputs/truth_states/manifest.csv']
    ctx = json.loads((e2/'context_metadata.json').read_text())
    provenance = json.loads((args.panel_dir/'provenance.json').read_text())
    if file_hash(e2/'contexts.npz') != ctx['contexts_sha256'] or file_hash(args.panel_dir/'panel.csv') != provenance['panel_sha256']:
        raise ValueError('E2 contexts or historical panel changed')
    original = json.loads((e2/'protocol.json').read_text())
    paths.append(e2/'protocol.json')
    for p, digest in original['inputs_sha256'].items():
        if file_hash(Path(p)) != digest:
            raise ValueError(f'Original E2 input changed: {p}')
        paths.append(Path(p))
    for seed in SEEDS:
        packed = e2/f'packed/seed_{seed}'
        metadata = json.loads((packed/'metadata.json').read_text())
        if metadata['seed_offset'] != seed*1000000 or metadata['attempts'] != 3*N:
            raise ValueError('Wrong original generation budget/seed')
        for name, field in [('curves.npz','packed_sha256'), ('attempts.npz','attempts_sha256')]:
            if file_hash(packed/name) != metadata[field]:
                raise ValueError('Modified packed synthetic bank')
            paths.append(packed/name)
        paths.append(packed/'metadata.json')
        source = Path(metadata['source'])
        if file_hash(source) != metadata['source_sha256']:
            raise ValueError('Modified original synthetic bank')
        paths.append(source)
        folder = e2/f'groups/seed_{seed}_N18387_k5'
        complete = json.loads((folder/'complete.json').read_text())
        for name in ['forecasts.csv.gz','matches.csv.gz','runtime.json']:
            if file_hash(folder/name) != complete['sha256'][name]:
                raise ValueError('Modified E2 reference forecasts')
            paths.append(folder/name)
        paths.append(folder/'complete.json')
    for name in ['experiments/sensitivity/library.py','experiments/synthetic/bank.py','experiments/synthetic/generator.py',
                 'experiments/forecasting/model.py','experiments/forecasting/neighbors.py','experiments/forecasting/smoothing.py',
                 'experiments/error_analysis/conditions.py','experiments/exploration/synthetic_eda.py','experiments/runtime.py']:
        paths.append(ROOT/name)
    return {str(p.resolve()):file_hash(p) for p in set(paths)}


def read_results(db):
    data = pd.read_sql_query('''SELECT p.*,c.series,c.forecast_origin,c.phase,c.case_level,c.history_stage,
      c.history_weeks,c.normalization,c.level_scale,o.target_end_date,o.actual,
      m.minimum_distance,m.median_distance,m.maximum_distance
      FROM predictions p JOIN cases c USING(case_id) JOIN observations o USING(case_id,horizon)
      JOIN matches m USING(seed,regime,library,case_id)''', db)
    if len(data) != 360000 or data.duplicated(['seed','regime','library',*KEYS]).any():
        raise ValueError('Require all 36 complete, aligned library/seed runs')
    if not data.groupby(['seed','regime','library']).size().eq(10000).all():
        raise ValueError('Unbalanced target sets')
    data['error'] = data.prediction-data.actual
    data['absolute_error'] = data.error.abs()
    data['normalized_absolute_error'] = data.absolute_error/data.normalization
    data['normalized_minimum_distance'] = data.minimum_distance/(K*data.level_scale)
    data['normalized_median_distance'] = data.median_distance/(K*data.level_scale)
    return data


def strata(data):
    yield 'overall','all',data
    for dim in ['horizon','phase','case_level','history_stage']:
        for label,g in data.groupby(dim, sort=True):
            yield dim,str(label),g


def score_tables(data):
    rows = []
    for (seed, regime, library), frame in data.groupby(['seed','regime','library'], sort=True):
        for dimension, label, g in strata(frame):
            cases = g.drop_duplicates('case_id')
            rows.append(dict(seed=seed,regime=regime,library=library,dimension=dimension,group=label,
                n=len(g),cases=len(cases),MAE=float(g.absolute_error.mean()),RMSE=float(np.sqrt(np.mean(g.error**2))),
                normalized_MAE=float(g.normalized_absolute_error.mean()),bias=float(g.error.mean()),
                normalized_NN=float(cases.normalized_minimum_distance.mean()),
                normalized_top_m_median=float(cases.normalized_median_distance.mean())))
    scores = pd.DataFrame(rows)
    rows = []
    for key,g in scores.groupby(['regime','library','dimension','group'], sort=True):
        if sorted(g.seed) != SEEDS:
            raise ValueError('Missing independent seed')
        row = dict(zip(['regime','library','dimension','group'],key),seeds=3,targets_per_seed=int(g.n.iloc[0]))
        for metric in ['MAE','RMSE','normalized_MAE','bias','normalized_NN','normalized_top_m_median']:
            row[metric+'_mean'] = float(g[metric].mean());row[metric+'_sd'] = float(g[metric].std(ddof=1))
        rows.append(row)
    return scores,pd.DataFrame(rows)


def paired_comparisons(data):
    ref = data.loc[data.library.eq('mixed'), ['seed','regime','case_id','horizon','actual','absolute_error']]
    paired = data.loc[~data.library.eq('mixed')].merge(ref,on=['seed','regime','case_id','horizon'],
                validate='many_to_one',suffixes=('','_mixed'))
    if not np.array_equal(paired.actual,paired.actual_mixed):
        raise ValueError('Paired reference labels differ')
    paired['delta'] = paired.absolute_error-paired.absolute_error_mixed
    rows = []
    for (regime,library),frame in paired.groupby(['regime','library']):
        for dimension,label,g in strata(frame):
            clusters = g.groupby('series').delta.agg(['sum','size'])
            rng = np.random.default_rng(DRAW_SEED)
            draw = rng.integers(0,len(clusters),size=(500,len(clusters)))
            estimates = clusters['sum'].to_numpy()[draw].sum(axis=1)/clusters['size'].to_numpy()[draw].sum(axis=1)
            lo,hi = np.quantile(estimates,[.025,.975])
            rows.append(dict(regime=regime,library=library,dimension=dimension,group=label,
                delta_MAE=float(g.delta.mean()),lo=float(lo),hi=float(hi),unique_targets=g[['case_id','horizon']].drop_duplicates().shape[0],
                states=len(clusters),seeds=3))
    return pd.DataFrame(rows)


def morphology_summary(db):
    return pd.read_sql_query('''SELECT seed,family,cadence,scale,COUNT(*) AS curves,
      AVG(observations) AS length_mean,AVG(peaks) AS peaks_mean,AVG(peaks_per_100_steps) AS peaks_per_100_mean,
      AVG(relative_change) AS relative_change_mean,AVG(relative_volatility) AS relative_volatility_mean,
      AVG(cv) AS CV_mean,SUM(valid_windows) AS valid_windows,SUM(constant_windows) AS constant_windows
      FROM curve_stats GROUP BY seed,family,cadence,scale''',db)


def verify_controls(db):
    rows = db.execute('SELECT seed,regime,library,metadata FROM runs').fetchall()
    expected = {(s,r,l) for s in SEEDS for r in REGIMES for l in LIBRARIES}
    if {(s,r,l) for s,r,l,_ in rows} != expected:
        raise ValueError('Missing or unexpected library/seed/size control')
    metadata = {(s,r,l):json.loads(m) for s,r,l,m in rows}
    for (seed,regime,lib),m in metadata.items():
        if sum(m['family_curves'].values()) != m['accepted_curves'] or sum(m['family_windows'].values()) != m['valid_windows']:
            raise ValueError('Family counts do not sum to library size')
        if regime == 'equal_curves':
            if m['accepted_curves'] != CURVES or (lib=='mixed' and list(m['family_curves'].values()) != [CURVES//3]*3):
                raise ValueError('Curve-count control failed')
        if regime == 'equal_windows':
            if m['valid_windows'] != WINDOWS or (lib=='mixed' and list(m['family_windows'].values()) != [WINDOWS//3]*3):
                raise ValueError('Window-count control failed')
        if not 0 <= m['constant_history_windows'] <= m['valid_windows']:
            raise ValueError('Invalid constant-history audit')
    for seed in SEEDS:
        for metric in ['accepted_curves','valid_windows','constant_history_windows']:
            if sum(metadata[(seed,'full',lib)][metric] for lib in LIBRARIES[:3]) != metadata[(seed,'full','mixed')][metric]:
                raise ValueError('Full family union differs from original mixed bank')
    return dict(size_controls_verified=True,full_family_unions_verified=True,
                reference_forecasts_reused=30000,new_forecasts=330000)


def plot_results(out, db, data, summary, comparisons, libraries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors = dict(zip(LIBRARIES,['tab:blue','tab:orange','tab:green','tab:red']))
    directory = out/'figures';directory.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(directory/f'{name}.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig)
    morphology = pd.read_sql_query('SELECT * FROM curve_stats',db)
    fig,axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,cadence in enumerate(['weekly','monthly']):
        g = morphology.loc[morphology.cadence.eq(cadence)]
        present = [lib for lib in LIBRARIES[:3] if g.family.eq(lib).any()]
        for ax,col,title in zip(axes[i],['observations','peaks_per_100_steps','relative_volatility'],
                               ['Length in original steps','Peaks per 100 original steps','SD of changes / curve mean']):
            arrays = [g.loc[g.family.eq(lib),col].to_numpy() for lib in present]
            ax.boxplot(arrays,tick_labels=present,showfliers=False)
            ax.set_title(cadence+': '+title);ax.tick_params(axis='x',rotation=15)
    fig.suptitle('Three independent original banks; boxes pool curves, outliers hidden only for display\nWeekly and monthly steps have different time units')
    save(fig,'curve_distributions')

    fig,axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,regime in enumerate(REGIMES):
        g = libraries.loc[libraries.regime.eq(regime)]
        for i,(column,label) in enumerate([('accepted_curves','Accepted curves'),('valid_windows','Valid matching windows')]):
            values = [g.loc[g.library.eq(lib),column].mean() for lib in LIBRARIES]
            axes[i,j].bar(LIBRARIES,values,color=[colors[lib] for lib in LIBRARIES])
            axes[i,j].set(title=regime,ylabel=label);axes[i,j].tick_params(axis='x',rotation=20)
    save(fig,'size_controls')

    fig,axes = plt.subplots(2,3,figsize=(13,7),sharey='row',layout='constrained')
    overall = summary.loc[summary.dimension.eq('overall')]
    for j,regime in enumerate(REGIMES):
        for i,metric in enumerate(['MAE','RMSE']):
            ax = axes[i,j]
            for position,lib in enumerate(LIBRARIES):
                row = overall.loc[overall.regime.eq(regime)&overall.library.eq(lib)].iloc[0]
                ax.errorbar(position,row[metric+'_mean'],yerr=row[metric+'_sd'],fmt='D',
                            color=colors[lib],capsize=4,ms=5,zorder=2)
                scores = data.loc[data.regime.eq(regime)&data.library.eq(lib)].groupby('seed').error.apply(
                    lambda e:np.mean(np.abs(e)) if metric=='MAE' else np.sqrt(np.mean(e**2)))
                ax.scatter(position+np.linspace(-.08,.08,3),scores,color='black',s=13,zorder=3)
            ax.set(xticks=range(4),xticklabels=LIBRARIES,ylabel=metric,title=regime)
            ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Identical 10,000 targets per run; black dots = banks, diamonds = mean ± seed SD\nPoint plots use a zoomed y-axis; SD is descriptive over 3 seeds, not a confidence interval')
    save(fig,'accuracy')

    fig,axes = plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    matches = data.drop_duplicates(['seed','regime','library','case_id'])
    for ax,regime in zip(axes,REGIMES):
        for lib in LIBRARIES:
            values = np.sort(matches.loc[matches.regime.eq(regime)&matches.library.eq(lib),'normalized_minimum_distance'])
            ax.plot(values,np.arange(1,len(values)+1)/len(values),color=colors[lib],label=lib)
        ax.set(title=regime,xlabel='Minimum L1 / (k × origin case scale)',ylabel='Empirical CDF')
        ax.set_xscale('symlog',linthresh=.001)
        ax.legend(fontsize=7)
    fig.suptitle('Normalized nearest-neighbor distance: 2,500 cases × 3 banks (descriptive, not independent replicates)')
    save(fig,'normalized_matching')

    fig,axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,regime in enumerate(REGIMES):
        for i,(dimension,labels) in enumerate([('phase',['declining','stable','rising']),('horizon',['1','2','3','4'])]):
            ax = axes[i,j]
            for lib in LIBRARIES[:3]:
                g = comparisons.loc[comparisons.regime.eq(regime)&comparisons.library.eq(lib)&comparisons.dimension.eq(dimension)].set_index('group').reindex(labels)
                ax.plot(labels,g.delta_MAE,'o-',color=colors[lib],label=lib)
            ax.axhline(0,color='k',ls='--',lw=1);ax.set(title=regime,xlabel=dimension,ylabel='MAE(single) − MAE(mixed)')
            ax.legend(fontsize=7)
    fig.suptitle('Paired error difference on identical targets; positive values favor mixed')
    save(fig,'conditional_errors')

    fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,metric,title in zip(axes,['MAE','normalized_NN'],['Across-bank SD of MAE','Across-bank SD of normalized NN distance']):
        for j,lib in enumerate(LIBRARIES):
            g = overall.loc[overall.library.eq(lib)].set_index('regime').reindex(REGIMES)
            ax.bar(np.arange(3)+(j-1.5)*.19,g[metric+'_sd'],width=.18,color=colors[lib],label=lib)
        ax.set(xticks=range(3),xticklabels=REGIMES,title=title,ylabel='Sample SD over only 3 seeds');ax.legend(fontsize=7)
    fig.suptitle('Seed stability is descriptive; three seeds do not establish a population ranking')
    save(fig,'seed_stability')


def markdown_table(frame, columns):
    result = '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    for row in frame[columns].itertuples(index=False,name=None):
        result += '| '+' | '.join(f'{v:,.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |\n'
    return result


def write_report(out,summary,comparisons,morphology,libraries,db):
    overall = summary.loc[summary.dimension.eq('overall')]
    delta = comparisons.loc[comparisons.dimension.eq('overall')]
    winners = []
    for regime in REGIMES:
        g = overall.loc[overall.regime.eq(regime)]
        winner = g.loc[g.MAE_mean.idxmin()]
        stable = g.loc[g.MAE_sd.idxmin()]
        winners.append(f'- **{regime}**: MAE thấp nhất trên cohort là {winner.library} ({winner.MAE_mean:,.2f}); SD MAE nhỏ nhất là {stable.library} ({stable.MAE_sd:,.2f}). Đây là mô tả E3, không chọn lại model từ test.')
    seasonal = delta.loc[delta.library.eq('seasonal')].set_index('regime')
    phase = summary.loc[summary.regime.eq('equal_windows') & summary.dimension.eq('phase')]
    relative_sd = overall.MAE_sd/overall.MAE_mean
    matching_notes = []
    for regime in REGIMES:
        g = overall.loc[overall.regime.eq(regime)]
        nearest = g.loc[g.normalized_NN_mean.idxmin()]
        accurate = g.loc[g.MAE_mean.idxmin()]
        matching_notes.append(f'- {regime}: mean normalized NN nhỏ nhất là **{nearest.library} ({nearest.normalized_NN_mean:.6f})**, trong khi MAE nhỏ nhất là **{accurate.library} ({accurate.MAE_mean:,.2f})**.')
    weekly = morphology.loc[morphology.cadence.eq('weekly')].groupby('family')[['peaks_mean','relative_volatility_mean']].mean()
    raw_volatility = pd.read_sql_query('SELECT family,cadence,relative_volatility FROM curve_stats',db)
    pooled_volatility = raw_volatility.groupby(['family','cadence']).relative_volatility.agg(
        mean='mean',median='median',p90=lambda s:s.quantile(.9),p99=lambda s:s.quantile(.99)).reset_index()
    report = f'''# Experiment 3 — Impact of Synthetic Library Characteristics

Giữ nguyên sMOA Python: k=5, m=4.422, GAM/GCV, omit bridge, median increments. **3 banks độc lập × 4 loại thư viện × 3 chế độ kích thước = 36 runs**, mỗi run **50 bang ×50 origins ×4 horizons =10.000 targets**, tổng **360.000 point forecasts**. Cohort giống E2/E4, có history rất ngắn và giai đoạn muộn; không phải toàn bộ lịch 112 origins của paper_full.

## Thiết kế kiểm soát

1. **full**: toàn bộ accepted curves của 18.387 attempts/family. Mixed giữ nguyên bản ba họ, gồm khoảng 54.000 curves; từng họ khoảng 17.000–18.000. Nhánh này chứa cả ảnh hưởng thành phần và kích thước.
2. **equal_curves**: 15.000 accepted curves/library; mixed lấy 5.000/họ. Uniform sampling không replacement, không nhìn COVID outcomes. Mixed subsets nằm trong lựa chọn single-family tương ứng.
3. **equal_windows**: đúng 3.000.000 valid matching windows/library; mixed lấy 1.000.000/họ. Giữ m=4.422. Đây là control bổ sung vì cùng số curves vẫn có M khác nhau do độ dài và cửa sổ bị lọc. Window sampling giữ các x/y của segment nguyên vẹn; không sửa generator.

Tất cả chọn mẫu dùng namespace RNG cố định trong protocol, giữ thứ tự nguồn khi có distance ties. Seeds bank=0/1/2, offsets=0/1.000.000/2.000.000. Không dùng phần mở rộng N22k/30k của E2. Không thay generator, không regenerate bank, không chạy BO hay UQ. Ba full-mixed runs tái dùng và kiểm chứng 30.000 predictions của E2; 33 runs còn lại được tính mới.

![Sizes](figures/size_controls.png)

## Sai số và variability across seeds

{markdown_table(overall,['regime','library','MAE_mean','MAE_sd','RMSE_mean','RMSE_sd','normalized_NN_mean','normalized_NN_sd'])}

{chr(10).join(winners)}

Full seasonal−mixed chỉ **{seasonal.loc['full','delta_MAE']:+.2f} MAE**, CI [{seasonal.loc['full','lo']:+.2f}, {seasonal.loc['full','hi']:+.2f}] chứa 0: kết quả chưa phân biệt rõ seasonal và full-mixed trên ba banks này. Khi đồng nhất curves, seasonal−mixed là **{seasonal.loc['equal_curves','delta_MAE']:+.2f}**, CI [{seasonal.loc['equal_curves','lo']:+.2f}, {seasonal.loc['equal_curves','hi']:+.2f}]; khi đồng nhất windows là **{seasonal.loc['equal_windows','delta_MAE']:+.2f}**, CI [{seasonal.loc['equal_windows','lo']:+.2f}, {seasonal.loc['equal_windows','hi']:+.2f}]. Vì thế không thể quy lợi thế full-mixed thành “trộn ba họ luôn tốt hơn”.

Relative SD của MAE qua các cấu hình nằm trong **{relative_sd.min():.3%}–{relative_sd.max():.3%}** trên ba seeds này. Mixed có SD nhỏ nhất ở full và equal_windows, nhưng không ở equal_curves. Seed variation nhỏ so với mức MAE không bảo đảm mọi ranking bền vững, nhất là full seasonal và mixed chỉ chênh vài ca MAE.

MAE/RMSE được tính riêng từng seed, sau đó lấy mean và sample SD của ba scores; RMSE_mean không phải pooled RMSE. Seed SD chỉ mô tả ba banks. Với hai nhánh đồng nhất kích thước, biến thiên gồm cả RNG của bank và subsampling; chưa phân rã hai nguồn này. Bộ test là chung, không có 30.000 targets độc lập chỉ vì lặp ba seeds.

![Accuracy](figures/accuracy.png)
![Seed variability](figures/seed_stability.png)

## So sánh paired với mixed trong cùng chế độ kích thước

{markdown_table(delta,['regime','library','delta_MAE','lo','hi','unique_targets','states','seeds'])}

Delta = MAE(single)−MAE(mixed): dương ưu thế mixed, âm ưu thế single. CI bootstrap nguyên bang 500 lần, giữ origins/horizons và ba seeds cùng nhau. CI điều kiện trên ba bank seeds này, không đại diện đầy đủ generator-seed uncertainty hoặc đồng biến dịch giữa các bang; không hiệu chỉnh multiple comparisons.

![Conditional errors](figures/conditional_errors.png)

## Matching theo diễn biến thực tế

Distance chính = minimum exact L1/(5×mean last6 input đã floor tại 1). Chỉ chuẩn hóa khi báo cáo, không đổi neighbor search. Có cả median top-m normalized distance; tính một record/state-origin, không đếm lại bốn horizons. Phase và case-level lấy từ lịch sử vintage tại origin, theo ngưỡng E4. `scores.csv` và `comparisons.csv` chứa overall, horizons, rising/declining/stable, low/medium/high và history stage.

{chr(10).join(matching_notes)}

Trong equal_windows, mixed match lịch sử gần nhất ở cả ba phase, nhưng seasonal có MAE thấp hơn ở cả ba phase trên trung bình ba banks. Đây là ví dụ trực tiếp rằng minimum distance đo chất lượng biểu diễn history, không đo tính đúng của future analogues. Bảng dưới giữ cùng M để đọc đặc điểm này:

{markdown_table(phase,['library','group','MAE_mean','normalized_NN_mean'])}

![Matching](figures/normalized_matching.png)

NN distance thấp cho thấy đoạn lịch sử được biểu diễn gần hơn, không bảo đảm future trajectory gần truth. Cần đọc matching và MAE cùng nhau, đặc biệt trong nhánh equal_windows; không quy mọi cải thiện full-mixed cho riêng sự đa dạng khi M cũng lớn hơn.

## Ba họ synthetic khác nhau ra sao?

{markdown_table(morphology.groupby(['family','cadence','scale'],as_index=False).agg(curves=('curves','sum'),length_mean=('length_mean','mean'),peaks_mean=('peaks_mean','mean'),relative_volatility_mean=('relative_volatility_mean','mean')),['family','cadence','scale','curves','length_mean','peaks_mean','relative_volatility_mean'])}

Mỗi curve giữ cadence và scale gốc. Peak rule dùng median3, prominence 10% amplitude, separation 4 original steps; endpoints không tính là peak. Các thống kê pooled phải đọc cùng cadence; một monthly step không phải một weekly step. Relative volatility dùng SD(diff)/curve mean, so sánh shape thay vì trộn raw counts với proportions. Boxes pooled ba banks, bỏ hiển thị fliers nhưng không bỏ curves khỏi thống kê.

Ở cadence weekly, số peaks trung bình theo rule là rollercoaster **{weekly.loc['rollercoaster','peaks_mean']:.2f}**, wiggle **{weekly.loc['wiggle','peaks_mean']:.2f}**, seasonal **{weekly.loc['seasonal','peaks_mean']:.2f}**. Relative volatility tương ứng **{weekly.loc['rollercoaster','relative_volatility_mean']:.3f} / {weekly.loc['wiggle','relative_volatility_mean']:.3f} / {weekly.loc['seasonal','relative_volatility_mean']:.3f}**. Wiggle tăng biến động so với rollercoaster nhưng không tăng số peaks sau median/prominence filtering; không đồng nhất wiggles nhỏ với các đỉnh dịch được đếm. Chỉ seasonal có monthly curves. Các bảng morphology lấy trung bình seed-specific means trong từng cadence/scale, số curves là tổng ba banks; boxplots dùng toàn bộ curves pooled.

**Mean không mô tả toàn bộ phân phối volatility.** Seasonal weekly có median thấp hơn rollercoaster/wiggle nhưng đuôi phải rất dài, làm mean cao hơn. Không suy ra mọi seasonal curve đều biến động hơn. Bảng dưới dùng quantiles trên toàn bộ curves pooled, tách cadence:

{markdown_table(pooled_volatility,['family','cadence','mean','median','p90','p99'])}

![Morphology](figures/curve_distributions.png)

## Kiểm chứng và giới hạn

- Frozen E2 contexts, packed banks, source CSV, original predictions và historical panel đều kiểm tra hashes/provenance; dùng nguyên final truth thô để chấm. Input tại origin floor 1 giống core; negative scoring labels được giữ.
- Giữ tiêu chí valid windows `np.var(...,ddof=1)>0`. Một số history thật sự constant vẫn vượt tiêu chí do floating-point rounding. Đếm và lưu `constant_history_windows`/fraction trong library_summary.csv; không âm thầm sửa core giữa các ablations. Nearest distance có thể chịu ảnh hưởng của các rows này.
- Curve equality không đồng nghĩa window equality; window equality không đồng nghĩa số curves hữu ích ngang nhau. Vì vậy báo cả C, M, số windows/curves theo họ và constant fraction cho từng run.
- Chỉ 3 seeds và một fixed COVID panel; kết quả không đủ khẳng định library tốt nhất cho mọi epidemics. Giữ algorithm/matching, không thiết kế generator mới hoặc tuning trên test.
- `results.sqlite` lưu curves' morphology, origin features, raw outcomes, từng forecast và từng matching distance; `runs` lưu metadata/selection hashes. Không xuất thêm một bản CSV forecast trùng database. Các tables nhỏ và sáu bộ hình dùng chung trong notebook/report.

Rerun/resume bằng cùng lệnh `--resume`; completed groups giữ nguyên, scratch embeddings được xóa sau từng worker. `protocol.json` chứa design, source và input hashes; `progress.json` chứa trạng thái và kiểm chứng cuối.
'''
    (out/'report.md').write_text(report)


def analyze(out):
    db = connect(out)
    if db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] != 36:
        raise ValueError('Do not analyze an incomplete experiment as final')
    verification = verify_controls(db)
    data = read_results(db)
    scores,summary = score_tables(data)
    comparisons = paired_comparisons(data)
    morphology = morphology_summary(db)
    metadata = [json.loads(r[0]) for r in db.execute('SELECT metadata FROM runs ORDER BY seed,regime,library')]
    libraries = pd.DataFrame(metadata)
    for col in ['family_curves','family_windows']:
        libraries[col] = libraries[col].map(lambda v:json.dumps(v,sort_keys=True))
    for name,frame in [('scores',scores),('seed_summary',summary),('comparisons',comparisons),
                       ('morphology',morphology),('library_summary',libraries)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    plot_results(out,db,data,summary,comparisons,libraries)
    write_report(out,summary,comparisons,morphology,libraries,db)
    db.close()
    return verification


def run(args):
    start = time.perf_counter()
    out = args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock = acquire_run_lock(out/'run.lock')
    prior_progress = json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    inputs = validate_inputs(args)
    signature = engine_signature()
    path = out/'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if not args.resume or previous['inputs_sha256'] != inputs or previous['engine_signature'] != signature:
            raise ValueError('Existing run requires --resume with unchanged design, engine and inputs')
    else:
        atomic_json(path,dict(rules=RULES,inputs_sha256=inputs,engine_signature=signature,
            source=Path(__file__).read_text(),created_utc=datetime.now(timezone.utc).isoformat(),
            software={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','matplotlib']}))
    db = connect(out);initialize(db)
    if not db.execute("SELECT name FROM sqlite_master WHERE name='cases'").fetchone():
        prepare_cases(args.e2_dir,args.panel_dir,db)
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS cases_id ON cases(case_id)')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS observation_key ON observations(case_id,horizon)')
    db.commit()
    cases = pd.read_sql_query('SELECT * FROM cases ORDER BY case_id',db)
    status = dict(status='running',total_runs=36,completed_runs=db.execute('SELECT COUNT(*) FROM runs').fetchone()[0])
    atomic_json(out/'progress.json',status)
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for seed in SEEDS:
        curve_statistics(args.e2_dir,seed,db)
        replay_full(args.e2_dir,seed,cases,db)
        for library in LIBRARIES:
            status.update(seed=seed,library=library,completed_runs=db.execute('SELECT COUNT(*) FROM runs').fetchone()[0])
            atomic_json(out/'progress.json',status)
            if db.execute('SELECT COUNT(*) FROM runs WHERE seed=? AND library=?',(seed,library)).fetchone()[0] == 3:
                continue
            print(f'E3 worker: seed={seed}, library={library}',flush=True)
            subprocess.run([sys.executable,str(Path(__file__)),'--worker','--seed',str(seed),'--library',library,
                '--output-dir',str(out),'--e2-dir',str(args.e2_dir.resolve())],env=env,check=True)
            status['completed_runs'] = db.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
            atomic_json(out/'progress.json',status)
    db.close()
    status.update(status='reporting');atomic_json(out/'progress.json',status)
    verification = analyze(out)
    if any(file_hash(Path(p)) != digest for p,digest in inputs.items()):
        raise RuntimeError('An input changed during E3')
    status.update(status='complete',completed_runs=36,forecasts=360000,matching_cases=90000,
        inputs_unchanged=True,verification=verification,analysis_source_sha256=file_hash(Path(__file__)),wall_seconds=time.perf_counter()-start,
        initial_full_run_wall_seconds=prior_progress.get('initial_full_run_wall_seconds',time.perf_counter()-start),
        reused_all_completed_runs=prior_progress.get('status')=='complete',
        completed_utc=datetime.now(timezone.utc).isoformat(),
        artifacts_sha256={str(p.relative_to(out)):file_hash(p) for p in [out/'results.sqlite',out/'report.md',*out.glob('*.csv'),*out.joinpath('figures').glob('*')]})
    atomic_json(out/'progress.json',status)
    print('Complete:',out/'report.md',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/experiment_3')
    parser.add_argument('--e2-dir',type=Path,default=ROOT/'outputs/experiment_2')
    parser.add_argument('--panel-dir',type=Path,default=ROOT/'outputs/epidemic_panel')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--library',choices=LIBRARIES)
    parser.add_argument('--execute-notebook',action='store_true')
    args = parser.parse_args()
    if args.execute_notebook:
        from notebooks import execute_notebook
        execute_notebook(Path(__file__).with_suffix('.ipynb'), working_dir=ROOT, timeout=600)
    elif args.worker:worker(args)
    else:run(args)
