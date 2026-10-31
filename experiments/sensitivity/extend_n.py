"""Extend completed E2 libraries incrementally without rerunning existing forecasts."""
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'experiments/synthetic'))
from runtime import acquire_run_lock,atomic_json,file_hash
from bank import FAMILIES,generate_ordered,load_packed,save_packed,write_audit
from design import configurations,protocol,SEEDS,SEED_OFFSETS

EXTENSION_NAMESPACE=2000202


def extension_key(offset,family,index):
    # Target N is deliberately absent, so later extensions retain these attempts.
    return [int(offset),EXTENSION_NAMESPACE,int(family),int(index)]


def extend_bank(parent,out,offset,target_n=22000,base_n=18387,workers=2):
    parent=Path(parent);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if target_n<=base_n:raise ValueError('Extension must increase N')
    parent_meta=json.loads((parent/'metadata.json').read_text())
    parent_hash=parent_meta.get('packed_sha256',parent_meta.get('sha256',{}).get('curves.npz'))
    attempts_hash=parent_meta.get('attempts_sha256',parent_meta.get('sha256',{}).get('attempts.npz'))
    if file_hash(parent/'curves.npz')!=parent_hash or file_hash(parent/'attempts.npz')!=attempts_hash:
        raise ValueError('Parent packed bank modified')
    config=dict(base_n=base_n,target_n=target_n,seed_offset=offset,namespace=EXTENSION_NAMESPACE,
        parent_packed_sha256=parent_hash,parent_attempts_sha256=attempts_hash,
        generator_sha256=file_hash(ROOT/'experiments/synthetic/generator.py'))
    path=out/'metadata.json'
    if path.exists():
        meta=json.loads(path.read_text())
        if meta['config']!=config:raise ValueError('Extension config differs')
        for name in ['curves.npz','attempts.npz','generation_audit.jsonl.gz']:
            if file_hash(out/name)!=meta['sha256'][name]:raise ValueError('Modified extension artifact')
        return meta
    curves=load_packed(parent/'curves.npz')
    with np.load(parent/'attempts.npz',allow_pickle=False) as data:
        old={name:data[name].copy() for name in data.files}
    if len(curves)!=len(old['index']) or np.any(old['index']>=base_n):raise ValueError('Invalid parent prefix')
    indices=old['index'].tolist();families=old['family'].tolist();ids=old['curve_id'].tolist()
    previous_count=len(curves)
    tasks=[(f,j,extension_key(offset,f,j)) for f in range(3) for j in range(base_n,target_n)]
    audit=[];start=time.perf_counter()
    for task,curve in zip(tasks,generate_ordered(tasks,workers)):
        f,j,key=task
        audit.append(dict(family=FAMILIES[f],index=j,seed_key=key,accepted=curve is not None))
        if curve is not None:
            curves.append(curve);indices.append(j);families.append(f)
            ids.append(3*base_n+3*(j-base_n)+f+1)
    generation_seconds=time.perf_counter()-start
    if len(set(ids))!=len(ids):raise ValueError('Duplicate curve identifiers')
    save_packed(curves,out/'curves.npz')
    np.savez(out/'attempts.npz',index=indices,family=families,curve_id=ids)
    write_audit(out/'generation_audit.jsonl.gz',audit)
    # Full old prefix, including every value and every ID, must remain identical.
    extended=load_packed(out/'curves.npz')
    for before,after in zip(curves[:previous_count],extended[:previous_count]):
        if not np.array_equal(before.ts,after.ts) or (before.curve_type,before.time_cadence,before.scale,before.seasonal)!=(after.curve_type,after.time_cadence,after.scale,after.seasonal):
            raise ValueError('Extension changed an original curve')
    if not all(np.array_equal(np.asarray(values)[:previous_count],old[name]) for name,values in [('index',indices),('family',families),('curve_id',ids)]):
        raise ValueError('Extension changed original attempt identities')
    meta=dict(status='complete',config=config,accepted=len(curves),attempts=3*target_n,
        original_accepted=previous_count,added_attempts=len(tasks),added_accepted=len(curves)-previous_count,
        original_prefix_verified=True,generation_seconds=generation_seconds,
        accepted_by_family={name:int(np.sum(np.asarray(families)==f)) for f,name in enumerate(FAMILIES)},
        sha256={name:file_hash(out/name) for name in ['curves.npz','attempts.npz','generation_audit.jsonl.gz']})
    atomic_json(path,meta)
    return meta


def run(out,workers=2,target_n=None):
    target_n=target_n or max(configurations().N)
    if target_n not in configurations().N.tolist() or target_n<=18387:raise ValueError('Choose a configured extension N')
    out=Path(out).resolve();lock=acquire_run_lock(out/'run.lock')
    extension=out/f'amendments/N{target_n}';extension.mkdir(parents=True,exist_ok=True)
    original=json.loads((out/'protocol.json').read_text())
    # The extension is solely library size, not a silent numerical model change.
    for name in ['experiments/forecasting/model.py','experiments/forecasting/neighbors.py',
                 'experiments/forecasting/smoothing.py','experiments/synthetic/generator.py','experiments/synthetic/bank.py',
                 'experiments/sensitivity/engine.py','experiments/sensitivity/library.py']:
        if file_hash(ROOT/name)!=original['source_sha256'][name]:raise ValueError('Original numerical core changed: '+name)
    for name,expected in original['inputs_sha256'].items():
        if file_hash(name)!=expected:raise ValueError('Original data changed')
    for name in ['contexts.npz']:
        if file_hash(out/name)!=json.loads((out/'context_metadata.json').read_text())['contexts_sha256']:raise ValueError('Original contexts changed')
    completions=sorted((out/'groups').glob('seed_*/complete.json'))
    old_hashes={}
    for completion in completions:
        if f'_N{target_n}_' in completion.parent.name:continue
        for name,expected in json.loads(completion.read_text())['sha256'].items():
            path=completion.parent/name
            if file_hash(path)!=expected:raise ValueError('Original forecast artifacts changed')
            old_hashes[str(path.relative_to(out))]=expected
    old_groups=len([p for p in completions if f'_N{target_n}_' not in p.parent.name])
    expected_old_groups=3*(configurations().groupby(['N','k']).ngroups-1)
    if old_groups!=expected_old_groups:raise ValueError('Require complete preceding E2 configurations')
    sources=[Path(__file__),Path(__file__).with_name('design.py'),Path(__file__).with_name('analysis.py')]
    amendment=dict(reason=f'User-requested post-study addition of N={target_n} to E2.A; not new BO selection',
        original_protocol_sha256=file_hash(out/'protocol.json'),design=protocol(),
        retained_original_artifacts_sha256=old_hashes,source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        source_text={str(p.relative_to(ROOT)):p.read_text() for p in sources},
        previous_amendments_sha256={str(p.relative_to(out)):file_hash(p) for p in (out/'amendments').glob('N*/protocol.json') if int(p.parent.name[1:])<target_n})
    saved=extension/'protocol.json'
    if saved.exists() and json.loads(saved.read_text())!=amendment:raise ValueError('Extension sources/config changed; do not resume silently')
    atomic_json(saved,amendment)
    status=dict(status=f'extending_N{target_n}',started_utc=datetime.now(timezone.utc).isoformat(),
        completed_groups=old_groups,total_groups=old_groups+3,original_protocol_preserved=True)
    atomic_json(out/'progress.json',status)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    generation=[]
    for seed in SEEDS:
        status.update(seed=seed,phase='extend_library');atomic_json(out/'progress.json',status)
        packed=extension/f'packed/seed_{seed}'
        parent=out/f'packed/seed_{seed}';base_n=18387
        for candidate in (out/'amendments').glob(f'N*/packed/seed_{seed}/metadata.json'):
            candidate_n=int(candidate.parents[2].name[1:])
            if base_n<candidate_n<target_n:base_n=candidate_n;parent=candidate.parent
        meta=extend_bank(parent,packed,SEED_OFFSETS[seed],target_n=target_n,base_n=base_n,workers=workers)
        job=out/f'groups/seed_{seed}_N{target_n}_k5'
        if (job/'complete.json').exists():
            for name,expected in json.loads((job/'complete.json').read_text())['sha256'].items():
                if file_hash(job/name)!=expected:raise ValueError('Modified extension result')
        else:
            status.update(phase='forecasting',N=target_n,k=5,m=4422);atomic_json(out/'progress.json',status)
            subprocess.run([sys.executable,str(Path(__file__).with_name('engine.py')),'--out',str(job),
                '--packed',str(packed),'--context',str(out/'contexts.npz'),'--seed',str(seed),
                '--N',str(target_n),'--k','5','--m','4422'],check=True,env=env)
        status['completed_groups']+=1;atomic_json(out/'progress.json',status)
        print(f'N={target_n} seed={seed} complete; accepted={meta["accepted"]}',flush=True)
    import pandas as pd
    for path in sorted((out/'amendments').glob('N*/packed/seed_*/metadata.json')):
        meta=json.loads(path.read_text());seed=int(path.parent.name.split('_')[1])
        generation.append(dict(seed=seed,N=meta['config']['target_n'],**{k:v for k,v in meta.items() if k not in ['sha256','config','accepted_by_family']}))
    pd.DataFrame(generation).to_csv(out/'extension_generation.csv',index=False)
    configurations().to_csv(out/'configurations.csv',index=False)
    atomic_json(out/'effective_protocol.json',dict(design=protocol(),original_protocol='protocol.json',
        original_protocol_sha256=file_hash(out/'protocol.json'),amendment=str(saved.relative_to(out)),
        amendment_sha256=file_hash(saved)))
    status.update(status='reporting',phase='analyze_extended_study');atomic_json(out/'progress.json',status)
    from analysis import analyze
    analyze(out)
    for name,expected in old_hashes.items():
        if file_hash(out/name)!=expected:raise ValueError('Extension modified an original job')
    status.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'progress.json',status)
    atomic_json(extension/'validation.json',dict(status='complete',new_configurations=1,new_seed_runs=3,new_targets=30000,
        old_groups_unchanged=old_groups,original_protocol_preserved=True,all_original_artifacts_unchanged=True))
    print('Extended E2 complete:',out/'report.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/experiment_2')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--target-n',type=int,default=None)
    args=parser.parse_args()
    run(args.output_dir,args.workers,args.target_n)
