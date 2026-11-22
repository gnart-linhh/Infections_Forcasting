"""Reference provenance and outcome-free, strictly truncated forecasting inputs."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from experiments.runtime import file_hash
from experiments.robust.method import RobustSMOA

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = ROOT/'experiments/synthetic'
if str(SYNTHETIC) not in sys.path:
    sys.path.insert(0, str(SYNTHETIC))
from bank import load_packed


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


@dataclass(frozen=True)
class ForecastCase:
    case_id: str
    dataset: str
    region: str
    history: np.ndarray
    reference_date: str | int
    issuance_date: str | int
    design: str

    def __post_init__(self):
        history = np.asarray(self.history, float).copy()
        if history.ndim != 1 or not np.isfinite(history).all() or len(history) not in (6, 8, 11):
            raise ValueError('Each case must supply exactly 6, 8, or 11 finite observations')
        history.flags.writeable = False
        object.__setattr__(self, 'history', history)


def load_reference(config):
    """Verify and reuse the original disk embedding, never old target rankings."""
    cfg = config['reference']
    folder, packed = resolve(cfg['embedding']), resolve(cfg['packed'])
    meta = json.loads((folder/'metadata.json').read_text())
    pmeta = json.loads((packed/'metadata.json').read_text())
    if (meta['k'], meta['horizon'], meta['window_filter']) != (5, 5, 'nonconstant_v2'):
        raise ValueError('Require original k5, h4 omit-bridge nonconstant_v2 embedding')
    if pmeta['source_sha256'] != meta['library_sha256']:
        raise ValueError('Reference packed curves and embedding are from different libraries')
    for name in ['x', 'y']:
        if file_hash(folder/(name+'.dat')) != meta[name+'_sha256']:
            raise ValueError('Reference embedding checksum mismatch: '+name)
    if file_hash(packed/'curves.npz') != pmeta['packed_sha256']:
        raise ValueError('Reference curve provenance changed')
    source = resolve(cfg['library'])
    if file_hash(source) != meta['library_sha256']:
        raise ValueError('Original library checksum mismatch')
    curves = load_packed(packed/'curves.npz')
    if len(curves) != meta['curves'] or len(curves) != pmeta['accepted']:
        raise ValueError('Reference curve counts differ')
    hashes = {array_hash(curve.ts) for curve in curves}
    del curves
    n = meta['rows']
    x = np.memmap(folder/'x.dat', mode='r', dtype='<f8', shape=(n, 5))
    y = np.memmap(folder/'y.dat', mode='r', dtype='<f8', shape=(n, 5))[:, 1:]
    engine = RobustSMOA(x, y, cfg['closest'], cfg['smoothing'])
    provenance = dict(library_sha256=meta['library_sha256'], packed_sha256=pmeta['packed_sha256'],
                      x_sha256=meta['x_sha256'], y_sha256=meta['y_sha256'],
                      curves=meta['curves'], windows=n, k=5, horizons=4, bridge='omit',
                      closest=cfg['closest'], smoothing=cfg['smoothing'],
                      generator_sha256=file_hash(SYNTHETIC/'generator.py'),
                      original_source={name: file_hash(ROOT/'experiments/forecasting'/name)
                                       for name in ['model.py', 'smoothing.py', 'neighbors.py', 'uncertainty.py']},
                      numerical_packages={name: importlib.metadata.version(name) for name in ['numpy', 'scipy']})
    return engine, hashes, provenance


def load_real_cases(config, smoke=False):
    """Read observed inputs only. Stored queries, fits and future labels are ignored.

    'early' uses the first T observations since the recorded observed onset;
    'fixed_origin' uses the most recent T at a common later origin. These designs
    get separate dataset identifiers and may not be pooled into one experiment.
    """
    cfg = config['evaluation']
    arm = {'early': 'A', 'fixed_origin': 'B'}[cfg['design']]
    path = resolve(cfg['database'])
    with closing(sqlite3.connect(f'file:{path}?mode=ro', uri=True)) as connection:
        frame = pd.read_sql_query('SELECT case_id,series,origin,T,onset,window_start,history,history_sha256,data_cutoff,issuance_date FROM cases WHERE domain=? AND arm=?',
                                  connection, params=('real', arm))
    frame = frame.loc[frame['T'].isin(config['history_lengths'])].copy()
    if cfg['regions']:
        if not set(cfg['regions']) <= set(frame.series):
            raise ValueError('Requested region is unavailable')
        frame = frame.loc[frame.series.isin(cfg['regions'])]
    if smoke:
        frame = frame.loc[frame.series.isin(sorted(frame.series.unique())[:config['smoke']['regions']])]
    result = []
    for row in frame.sort_values(['series', 'origin', 'T']).itertuples():
        history = np.frombuffer(row.history, dtype='<f8').copy()
        if len(history) != row.T or array_hash(np.maximum(history, 1)) != row.history_sha256:
            raise ValueError('Stored permitted history checksum/length mismatch')
        reference, issued = pd.Timestamp(row.origin), pd.Timestamp(row.issuance_date)
        if pd.Timestamp(row.data_cutoff) > issued or issued != reference+pd.Timedelta(days=2):
            raise ValueError('Historical input was not available at the recorded issuance')
        if pd.Timestamp(row.window_start) != reference-pd.Timedelta(weeks=row.T-1):
            raise ValueError('Nonconsecutive permitted history')
        if arm == 'A' and pd.Timestamp(row.onset) != pd.Timestamp(row.window_start):
            raise ValueError('Early cases must start at observed outbreak onset')
        result.append(ForecastCase(str(row.case_id), 'COVID19_'+cfg['design'], row.series, history,
                                   row.origin, row.issuance_date, cfg['design']))
    if not result:
        raise ValueError('No real forecast inputs satisfy the configured design')
    return result


def load_real_outcomes(config, cases):
    """Called by evaluation only after issued predictions are persisted."""
    allowed = {c.case_id for c in cases}
    with closing(sqlite3.connect(f'file:{resolve(config["evaluation"]["database"])}?mode=ro', uri=True)) as connection:
        observations = pd.read_sql_query('SELECT case_id,horizon,target,actual FROM observations', connection)
    observations['case_id'] = observations.case_id.astype(str)
    observations = observations.loc[observations.case_id.isin(allowed)].copy()
    observations = observations.rename(columns={'target': 'target_date'})
    if observations.duplicated(['case_id', 'horizon']).any() or not np.isfinite(observations.actual).all():
        raise ValueError('Invalid external scoring outcomes')
    return observations


def attach_outcomes(predictions, observations):
    """No fitting occurs here. Reject incomplete, duplicate, or misaligned targets."""
    if 'actual' in predictions or observations.duplicated(['case_id', 'horizon']).any():
        raise ValueError('Outcomes must be attached once, after forecasting')
    merged = predictions.merge(observations, on=['case_id', 'horizon', 'target_date'],
                               how='left', validate='many_to_one')
    if len(merged) != len(predictions) or not np.isfinite(merged.actual).all():
        raise ValueError('Incomplete or misaligned scoring targets')
    return merged
