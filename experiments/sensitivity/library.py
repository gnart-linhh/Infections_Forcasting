"""Validate full banks, select exact nested family prefixes, embed to scratch files."""
import csv
from itertools import groupby
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'experiments/synthetic'))
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
from generator import SyntheticCurve
from model import valid_history_windows
from bank import FAMILIES,save_packed,load_packed
from runtime import atomic_json,file_hash


def reuse_banks(source,out,levels=(18387,22000,30000),seeds=(0,1,2)):
    """Verify immutable banks/prefixes and link them; never copy CSVs or forecasts."""
    source=Path(source).resolve();out=Path(out).resolve()
    if source==out:raise ValueError('Reproduction needs a fresh output directory')
    hashes={};verified=[]
    for seed in seeds:
        previous=None;old_attempts=None
        for i,n in enumerate(levels):
            relative=Path(f'packed/seed_{seed}') if i==0 else Path(f'amendments/N{n}/packed/seed_{seed}')
            packed=source/relative;meta=json.loads((packed/'metadata.json').read_text())
            files={'curves.npz':meta['packed_sha256'],'attempts.npz':meta['attempts_sha256']} if i==0 else meta['sha256']
            if meta['attempts']!=3*n:raise ValueError('Bank budget differs')
            if (meta['seed_offset'] if i==0 else meta['config']['seed_offset'])!=seed*1000000:raise ValueError('Bank seed differs')
            for name in ['metadata.json',*files]:
                digest=file_hash(packed/name)
                if name in files and digest!=files[name]:raise ValueError('Modified reused bank: '+str(packed/name))
                hashes[str((packed/name).resolve())]=digest
            if i==0:
                csv=Path(meta['source']);digest=file_hash(csv)
                if digest!=meta['source_sha256']:raise ValueError('Modified original bank CSV')
                hashes[str(csv.resolve())]=digest
            current=load_packed(packed/'curves.npz')
            with np.load(packed/'attempts.npz',allow_pickle=False) as data:attempts={k:data[k].copy() for k in data.files}
            if len(current)!=meta['accepted'] or len(current)!=len(attempts['index']) or (attempts['index']>=n).any():raise ValueError('Invalid reused bank')
            if previous is not None:
                if meta['config']['base_n']!=levels[i-1] or not meta['original_prefix_verified']:raise ValueError('Incorrect extension chain')
                for key,values in old_attempts.items():
                    if not np.array_equal(attempts[key][:len(values)],values):raise ValueError('Extension changed attempt identities')
                for before,after in zip(previous,current):
                    if not np.array_equal(before.ts,after.ts) or (before.curve_type,before.scale,before.time_cadence,before.seasonal)!=(after.curve_type,after.scale,after.time_cadence,after.seasonal):
                        raise ValueError('Extension changed a prefix curve')
            dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() or dest.is_symlink():
                if not dest.is_symlink() or dest.resolve()!=packed.resolve():raise ValueError('Bank link differs')
            else:dest.symlink_to(packed,target_is_directory=True)
            verified.append(dict(seed=seed,N=n,accepted=len(current),source=str(packed),prefix_verified=i>0))
            previous=current;old_attempts=attempts
    return dict(root=str(source),banks=verified,sha256=hashes)


def pack_bank(source,out,offset,max_n=18387,replay=None):
    """Keep original accepted row order; missing/rejected attempt indices stay missing."""
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads(source.with_name('metadata.json').read_text())
    source_hash=file_hash(source)
    if meta['status']!='complete' or meta['curves_sha256']!=source_hash:
        raise ValueError('Incomplete/modified synthetic library')
    if meta['num_curves_per_family']!=max_n or meta['config']['seed_offset']!=offset:
        raise ValueError('Unexpected synthetic bank namespace or maximum N')
    current_generator=file_hash(ROOT/'experiments/synthetic/generator.py')
    replay_verified=False
    if meta['config']['source_sha256']['generator.py']!=current_generator:
        if replay is None:raise ValueError('Bank generator source changed; require a complete identical replay')
        replay=Path(replay)
        replay_meta=json.loads(replay.with_name('metadata.json').read_text())
        if replay_meta['config']['source_sha256']['generator.py']!=current_generator or file_hash(replay)!=source_hash:
            raise ValueError('Full seed0 replay differs from reference library')
        replay_verified=True
    curves=[];indices=[];families=[];ids=[];seen=set()
    tick=time.perf_counter()
    with source.open() as handle:
        for curve_id,rows in groupby(csv.DictReader(handle),key=lambda r:int(r['curve_id'])):
            if curve_id in seen:raise ValueError('Non-grouped curve IDs')
            seen.add(curve_id);rows=list(rows);first=rows[0]
            family_index=(curve_id-1)//max_n;attempt=(curve_id-1)%max_n
            if family_index>=3 or first['curve_type']!=FAMILIES[family_index]:raise ValueError('Curve family/attempt mapping differs')
            if int(first['seed'])!=offset+curve_id:raise ValueError('Curve seed differs from fixed family namespace')
            if [int(r['step']) for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Nonconsecutive curve steps')
            values=np.asarray([float(r['value']) for r in rows])
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid curve')
            curves.append(SyntheticCurve(values,first['curve_type'],first['time_cadence'],first['scale'],first['seasonal'].lower()=='true'))
            indices.append(attempt);families.append(family_index);ids.append(curve_id)
    if len(curves)!=meta['accepted_total']:raise ValueError('Accepted count differs')
    save_packed(curves,out/'curves.npz')
    np.savez(out/'attempts.npz',index=indices,family=families,curve_id=ids)
    for n in [500,1000,2000,5000,10000,max_n]:
        mask=np.asarray(indices)<n
        counts={f:int(np.sum(mask & (np.asarray(families)==i))) for i,f in enumerate(FAMILIES)}
        atomic_json(out/f'prefix_{n}.json',dict(N=n,accepted=int(mask.sum()),families=counts,attempts=3*n))
    generation=None
    timed_source=Path(replay) if replay_verified else source
    if (offset or replay_verified) and timed_source.parent.joinpath('run.lock').exists():
        generation=timed_source.with_name('metadata.json').stat().st_mtime-timed_source.parent.joinpath('run.lock').stat().st_mtime
    info=dict(source=str(source.resolve()),source_sha256=source_hash,seed_offset=offset,
        accepted=len(curves),attempts=3*max_n,pack_seconds=time.perf_counter()-tick,
        generation_export_seconds=generation,generation_reused=offset==0,full_replay_verified=replay_verified,
        packed_sha256=file_hash(out/'curves.npz'),attempts_sha256=file_hash(out/'attempts.npz'))
    atomic_json(out/'metadata.json',info)
    return info


def load_prefix(packed,n):
    packed=Path(packed)
    curves=load_packed(packed/'curves.npz')
    with np.load(packed/'attempts.npz',allow_pickle=False) as data:
        keep=data['index']<n
        ids=data['curve_id'][keep]
    return [c for c,selected in zip(curves,keep) if selected],ids


def embed(curves,curve_ids,k,scratch,h=4):
    scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    paths=[scratch/f'{name}.dat' for name in ['x','y','curve_ids']]
    count=0
    with paths[0].open('wb') as xf,paths[1].open('wb') as yf,paths[2].open('wb') as cf:
        for curve,curve_id in zip(curves,curve_ids):
            values=np.maximum(curve.ts,1e-8);width=k+h+2
            if len(values)<width:continue
            windows=np.lib.stride_tricks.sliding_window_view(values,width)
            valid=valid_history_windows(windows[:,:k+1])
            differences=np.diff(windows[valid],axis=1)
            np.ascontiguousarray(differences[:,:k],dtype='<f8').tofile(xf)
            np.ascontiguousarray(differences[:,k+1:],dtype='<f8').tofile(yf)
            np.full(len(differences),curve_id,dtype='<u4').tofile(cf)
            count+=len(differences)
    if not count:raise ValueError('Empty embedding')
    return (np.memmap(paths[0],mode='r',dtype='<f8',shape=(count,k)),
        np.memmap(paths[1],mode='r',dtype='<f8',shape=(count,h)),
        np.memmap(paths[2],mode='r',dtype='<u4',shape=(count,)))
