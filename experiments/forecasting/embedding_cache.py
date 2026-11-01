"""Full disk-backed embeddings with exact curve/window lookup for match audits."""
from __future__ import annotations
import csv
from itertools import groupby
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments'))
from experiments.error_analysis.library import embeddings
from runtime import acquire_run_lock,atomic_json,file_hash
from model import valid_history_windows


class CurveLookup:
    def __init__(self,names,codes):
        self.names=np.asarray(names)
        self.codes=codes

    def __getitem__(self,item):
        return self.names[self.codes[item]]

    def __len__(self):
        return len(self.codes)


def load_cached_library(source,out,k,h,bridge_mode='keep'):
    if bridge_mode not in ['keep','omit']:
        raise ValueError('Unknown bridge mode')
    omit=bridge_mode=='omit'
    h=h+int(omit)
    x,y=embeddings(source,out,k,h)
    with acquire_run_lock(out/'lookup.lock'):
        meta_path=out/'lookup.json'
        config=dict(version=1,embedding_metadata_sha256=file_hash(out/'metadata.json'),rows=len(x),k=k,h=h)
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if any(meta.get(key)!=value for key,value in config.items()):
                raise ValueError('Embedding audit lookup configuration changed')
            for name in ['curve_codes','steps']:
                path=out/f'{name}.dat'
                if path.stat().st_size!=len(x)*4 or file_hash(path)!=meta[name+'_sha256']:
                    raise ValueError('Incomplete or modified embedding audit lookup')
        else:
            names=[];count=0
            with source.open() as handle,(out/'curve_codes.tmp').open('wb') as codes,(out/'steps.tmp').open('wb') as steps:
                for curve,rows in groupby(csv.DictReader(handle),key=lambda r:r['curve_id']):
                    observations=sorted((int(r['step']),float(r['value'])) for r in rows)
                    values=np.asarray([v for _,v in observations])
                    name_index=len(names);names.append(curve)
                    if len(values)<k+1+h: continue
                    windows=np.lib.stride_tricks.sliding_window_view(np.maximum(values,1e-8),k+1+h)
                    starts=np.flatnonzero(valid_history_windows(windows[:,:k+1]))+1
                    np.full(len(starts),name_index,dtype='<u4').tofile(codes)
                    starts.astype('<u4').tofile(steps)
                    count+=len(starts)
            if count!=len(x): raise ValueError('Audit lookup differs from embedding row count')
            for name in ['curve_codes','steps']: (out/f'{name}.tmp').replace(out/f'{name}.dat')
            meta=dict(config,names=names,**{name+'_sha256':file_hash(out/f'{name}.dat') for name in ['curve_codes','steps']})
            atomic_json(meta_path,meta)
    codes=np.memmap(out/'curve_codes.dat',mode='r',dtype='<u4',shape=(len(x),))
    steps=np.memmap(out/'steps.dat',mode='r',dtype='<u4',shape=(len(x),))
    return x,y[:,1:] if omit else y,CurveLookup(meta['names'],codes),steps
