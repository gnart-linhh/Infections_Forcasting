"""Disk-backed full sMOA embedding; same windows and ordering as forecasting.model."""
from __future__ import annotations
import csv
from itertools import groupby
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments'))
sys.path.insert(0, str(ROOT/'experiments/forecasting'))
from runtime import atomic_json, file_hash, acquire_run_lock
from model import WINDOW_FILTER, valid_history_windows


def embeddings(source, out, k=5, horizon=4):
    out.mkdir(parents=True, exist_ok=True)
    with acquire_run_lock(out/'run.lock'):
        return _embeddings(source,out,k,horizon)


def _embeddings(source,out,k,horizon):
    config = dict(version=2, window_filter=WINDOW_FILTER, library_sha256=file_hash(source), k=k, horizon=horizon)
    info = out/'metadata.json'
    if info.exists():
        meta = json.loads(info.read_text())
        if any(meta.get(key) != value for key, value in config.items()):
            raise ValueError('Embedding cache differs from requested source/configuration; '
                             'use a new cache directory for nonconstant_v2; keep old run artifacts unchanged')
        for name, width in [('x',k), ('y',horizon)]:
            if (out/f'{name}.dat').stat().st_size != meta['rows']*width*8:
                raise ValueError('Incomplete embedding cache')
            if file_hash(out/f'{name}.dat') != meta[f'{name}_sha256']:
                raise ValueError('Embedding checksum mismatch')
    else:
        count, seen = 0, set()
        with source.open() as handle, (out/'x.tmp').open('wb') as xf, (out/'y.tmp').open('wb') as yf:
            for curve_id, rows in groupby(csv.DictReader(handle), key=lambda r:r['curve_id']):
                if curve_id in seen:
                    raise ValueError('Curve rows must be grouped')
                seen.add(curve_id)
                observations = sorted((int(r['step']), float(r['value'])) for r in rows)
                positions, values = map(np.asarray, zip(*observations))
                if not np.array_equal(positions, np.arange(1,len(values)+1)):
                    raise ValueError(f'Invalid curve steps: {curve_id}')
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f'Invalid curve values: {curve_id}')
                if len(values) >= k+1+horizon:
                    windows = np.lib.stride_tricks.sliding_window_view(np.maximum(values,1e-8), k+1+horizon)
                    keep = valid_history_windows(windows[:,:k+1])
                    diff = np.diff(windows[keep],axis=1)
                    np.ascontiguousarray(diff[:,:k],dtype='<f8').tofile(xf)
                    np.ascontiguousarray(diff[:,k:],dtype='<f8').tofile(yf)
                    count += len(diff)
                if len(seen)%5000 == 0:
                    print(f'Embedded {len(seen)} curves; {count} windows', flush=True)
        if not count:
            raise ValueError('Empty synthetic library')
        (out/'x.tmp').replace(out/'x.dat')
        (out/'y.tmp').replace(out/'y.dat')
        meta = dict(config, rows=count, curves=len(seen),
                    x_sha256=file_hash(out/'x.dat'),y_sha256=file_hash(out/'y.dat'))
        atomic_json(info,meta)
    return (np.memmap(out/'x.dat',mode='r',dtype='<f8',shape=(meta['rows'],k)),
            np.memmap(out/'y.dat',mode='r',dtype='<f8',shape=(meta['rows'],horizon)))
