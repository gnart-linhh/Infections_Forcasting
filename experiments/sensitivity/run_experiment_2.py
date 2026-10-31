"""Run the complete prespecified E2.A/B/C study on a common historical COVID panel."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
from runtime import atomic_json,file_hash,acquire_run_lock
from smoothing import thin_plate_gcv
from design import configurations,protocol,SEEDS,SEED_OFFSETS
from library import pack_bank


def prepare_context(panel_dir,manifest,out):
    panel=pd.read_csv(panel_dir/'panel.csv')
    provenance=json.loads((panel_dir/'provenance.json').read_text())
    if provenance['status']!='complete' or file_hash(panel_dir/'panel.csv')!=provenance['panel_sha256']:
        raise ValueError('Panel incomplete/changed')
    if panel.duplicated(['series','forecast_origin','target_end_date']).any():raise ValueError('Duplicate panel rows')
    entries=pd.read_csv(manifest)
    if len(entries)!=50 or entries.series.tolist()!=provenance['states']:raise ValueError('Require identical 50 states')
    truths={r.series:pd.read_csv(manifest.parent/r.truth).set_index('target_end_date').value for r in entries.itertuples()}
    origins=sorted(panel.forecast_origin.unique())
    if len(origins)!=50 or origins!=sorted(provenance['available_origins']):raise ValueError('Require unchanged 50 origins')
    rows=[];queries={k:[] for k in [3,5,7,9]};early={};positions=[]
    smooth_seconds=0.
    for origin in origins:
        for state in entries.series:
            history=panel.loc[panel.forecast_origin.eq(origin)&panel.series.eq(state)].sort_values('target_end_date')
            expected=pd.date_range('2020-01-25',origin,freq='7D').strftime('%Y-%m-%d').tolist()
            if history.target_end_date.tolist()!=expected:raise ValueError('Missing/misaligned historical weeks')
            raw=history.value.to_numpy(float);values=np.maximum(raw,1)
            if len(values)<10 or not np.isfinite(values).all():raise ValueError('Insufficient common history')
            tick=time.perf_counter();fitted=np.maximum(1,thin_plate_gcv(values));smooth_seconds+=time.perf_counter()-tick
            for k in queries:queries[k].append(np.diff(fitted[-k-1:]))
            targets=[(pd.Timestamp(origin)+pd.Timedelta(weeks=h)).date().isoformat() for h in range(1,5)]
            actual=[float(truths[state][t]) for t in targets]
            if not np.isfinite(actual).all():raise ValueError('Invalid actuals')
            if origin in origins[:16]:
                positions.append(len(rows))
                for length in [4,6,8,10]:
                    if length<5:continue
                    short=np.maximum(1,thin_plate_gcv(values[-length:]))
                    for k in queries:
                        if length>=k+1:early.setdefault(f'early_L{length}_k{k}',[]).append(np.diff(short[-k-1:]))
            rows.append(dict(series=state,origin=origin,last=values[-1],normalization=float(np.mean(values[-4:])),
                             actual=actual,targets=targets,history_weeks=len(values)))
    np.savez(out/'contexts.npz',series=[r['series'] for r in rows],origin=[r['origin'] for r in rows],
        last=[r['last'] for r in rows],normalization=[r['normalization'] for r in rows],actual=[r['actual'] for r in rows],
        targets=[r['targets'] for r in rows],history_weeks=[r['history_weeks'] for r in rows],early_positions=positions,
        **{f'query_k{k}':q for k,q in queries.items()},**early)
    pd.DataFrame([{key:value for key,value in row.items() if key not in ['actual','targets']} for row in rows]).to_csv(out/'cases.csv',index=False)
    atomic_json(out/'context_metadata.json',dict(cases=len(rows),targets=len(rows)*4,smoothing_seconds=smooth_seconds,
        contexts_sha256=file_hash(out/'contexts.npz')))


def run(args):
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=acquire_run_lock(out/'run.lock')
    design=protocol()
    sources=[Path(__file__),Path(__file__).with_name('design.py'),Path(__file__).with_name('library.py'),
        Path(__file__).with_name('extend_n.py'),
        Path(__file__).with_name('engine.py'),ROOT/'experiments/synthetic/generator.py',ROOT/'experiments/synthetic/bank.py',
        ROOT/'experiments/synthetic/generate_python.py',ROOT/'experiments/forecasting/model.py',
        ROOT/'experiments/forecasting/smoothing.py',ROOT/'experiments/forecasting/neighbors.py',ROOT/'experiments/runtime.py']
    inputs=[args.panel_dir/'panel.csv',args.panel_dir/'provenance.json',args.manifest,
        *[args.manifest.parent/r.truth for r in pd.read_csv(args.manifest).itertuples()]]
    config=dict(design=design,inputs_sha256={str(p.resolve()):file_hash(p) for p in inputs},
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','matplotlib']})
    saved=out/'protocol.json'
    if saved.exists():
        if json.loads(saved.read_text())!=config or not args.resume:raise ValueError('Existing run: require --resume with identical sources/design/inputs')
    else:atomic_json(saved,config)
    for path in sources:
        target=out/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    status=dict(status='preparing',started_utc=datetime.now(timezone.utc).isoformat(),completed_groups=0,
                total_groups=3*configurations().groupby(['N','k']).ngroups)
    atomic_json(out/'progress.json',status)
    if not (out/'contexts.npz').exists():prepare_context(args.panel_dir,args.manifest,out)
    else:
        ctx=json.loads((out/'context_metadata.json').read_text())
        if file_hash(out/'contexts.npz')!=ctx['contexts_sha256']:raise ValueError('Modified forecast contexts')
    configurations().to_csv(out/'configurations.csv',index=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    groups=list(configurations().groupby(['N','k'],sort=True))
    for seed in SEEDS:
        source=args.reference_library if seed==0 else out/f'libraries/seed_{seed}/curves.csv'
        if not source.with_name('metadata.json').exists():
            command=[sys.executable,str(ROOT/'experiments/synthetic/generate_python.py'),'--num-curves','18387',
                '--seed-offset',str(SEED_OFFSETS[seed]),'--workers',str(args.workers),'--output-dir',str(source.parent)]
            if source.parent.exists() and any(source.parent.iterdir()):command.append('--resume')
            subprocess.run(command,check=True,env=env)
        packed=out/f'packed/seed_{seed}'
        if not (packed/'metadata.json').exists():
            status.update(status='packing_library',seed=seed);atomic_json(out/'progress.json',status)
            replay=out/'libraries/seed_0_replay/curves.csv' if seed==0 else None
            if seed==0 and not replay.with_name('metadata.json').exists():
                subprocess.run([sys.executable,str(ROOT/'experiments/synthetic/generate_python.py'),
                    '--seed-offset','0','--workers',str(args.workers),'--output-dir',str(replay.parent)],check=True,env=env)
            pack_bank(source,packed,SEED_OFFSETS[seed],replay=replay)
        else:
            meta=json.loads((packed/'metadata.json').read_text())
            if file_hash(source)!=meta['source_sha256'] or file_hash(packed/'curves.npz')!=meta['packed_sha256'] or file_hash(packed/'attempts.npz')!=meta['attempts_sha256']:
                raise ValueError('Modified packed library/source')
        for (n,k),group in groups:
            job_packed=packed
            if n>18387:
                from extend_n import extend_bank
                job_packed=out/f'amendments/N{n}/packed/seed_{seed}'
                parent=packed;base_n=18387
                for prior in sorted(configurations().N.unique()):
                    candidate=out/f'amendments/N{prior}/packed/seed_{seed}'
                    if base_n<prior<n and (candidate/'metadata.json').exists():parent=candidate;base_n=int(prior)
                extend_bank(parent,job_packed,SEED_OFFSETS[seed],target_n=int(n),base_n=base_n,workers=args.workers)
            job=out/f'groups/seed_{seed}_N{n}_k{k}'
            if (job/'complete.json').exists():
                completion=json.loads((job/'complete.json').read_text())
                for name,expected in completion['sha256'].items():
                    if file_hash(job/name)!=expected:raise ValueError('Modified completed group artifact')
            if not (job/'complete.json').exists():
                status.update(status='forecasting',seed=seed,N=int(n),k=int(k));atomic_json(out/'progress.json',status)
                command=[sys.executable,str(Path(__file__).with_name('engine.py')),'--out',str(job),'--packed',str(job_packed),
                    '--context',str(out/'contexts.npz'),'--seed',str(seed),'--N',str(n),'--k',str(k),'--m',*map(str,group.m)]
                print(f'E2: seed={seed}, N={n}, k={k}, m={group.m.tolist()}',flush=True)
                subprocess.run(command,check=True,env=env)
            status['completed_groups']+=1;atomic_json(out/'progress.json',status)
    status.update(status='reporting');atomic_json(out/'progress.json',status)
    extension_stats=[]
    for seed in SEEDS:
        for path in (out/'amendments').glob(f'N*/packed/seed_{seed}/metadata.json'):
            meta=json.loads(path.read_text())
            extension_stats.append(dict(seed=seed,N=meta['config']['target_n'],**{k:v for k,v in meta.items() if k not in ['sha256','config','accepted_by_family']}))
    if extension_stats:pd.DataFrame(extension_stats).to_csv(out/'extension_generation.csv',index=False)
    from analysis import analyze
    analyze(out)
    status.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'progress.json',status)
    print('Complete:',out/'report.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/experiment_2')
    p.add_argument('--panel-dir',type=Path,default=ROOT/'outputs/epidemic_panel')
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/truth_states/manifest.csv')
    p.add_argument('--reference-library',type=Path,default=ROOT/'outputs/reproduction_full/01_synthetic/curves.csv')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        atomic_json(a.output_dir/'failure.json',dict(type=type(exc).__name__,reason=str(exc)))
        raise
