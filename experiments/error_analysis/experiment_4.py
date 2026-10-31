"""Experiment 4: paper-parameter sMOA errors across origin-known epidemic conditions."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
import numpy as np
import pandas as pd
from runtime import atomic_json, file_hash, acquire_run_lock
from model import forecast
from neighbors import ExactNeighbors
from conditions import RULES, features, errors
from library import embeddings
from report import report


def run(args):
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    lock=acquire_run_lock(out/'run.lock')
    provenance=json.loads((args.panel_dir/'provenance.json').read_text())
    if provenance['status']!='complete' and not args.allow_incomplete:
        raise ValueError('Historical panel is incomplete; fetch it first or explicitly --allow-incomplete')
    if file_hash(args.panel_dir/'panel.csv') != provenance['panel_sha256']:
        raise ValueError('Historical panel checksum mismatch')
    library_meta=json.loads(args.library.with_name('metadata.json').read_text())
    if library_meta.get('num_curves_per_family') != 18387 or library_meta.get('attempted_total') != 55161:
        raise ValueError('Use the full paper-size synthetic library: 18387 attempts/family')
    library_hash=file_hash(args.library)
    if library_meta.get('curves_sha256') != library_hash or library_meta.get('status')!='complete':
        raise ValueError('Synthetic library checksum mismatch')
    truth_manifest=pd.read_csv(args.manifest)
    if truth_manifest.series.tolist()!=provenance['states']:
        raise ValueError('Panel and evaluation state sets differ')
    truths={}
    for row in truth_manifest.itertuples():
        path=args.manifest.parent/row.truth
        frame=pd.read_csv(path)
        if frame.target_end_date.duplicated().any() or not np.isfinite(frame.value).all():
            raise ValueError(f'Invalid evaluation truth: {path}')
        truths[row.series]=frame.set_index('target_end_date').value
    source_paths=[Path(__file__),Path(__file__).with_name('conditions.py'),Path(__file__).with_name('library.py'),
                  Path(__file__).with_name('report.py'),ROOT/'experiments/forecasting/model.py',
                  ROOT/'experiments/forecasting/smoothing.py',ROOT/'experiments/forecasting/neighbors.py',
                  ROOT/'experiments/runtime.py']
    rules=deepcopy(RULES)
    rules['smoa']['bridge_mode']=args.bridge_mode
    config=dict(rules=rules, panel_sha256=provenance['panel_sha256'],
                panel_provenance_sha256=file_hash(args.panel_dir/'provenance.json'),
                panel_complete=provenance['status']=='complete', library_sha256=library_hash,
                library_accepted=library_meta['accepted_total'], library_attempted=library_meta['attempted_total'],
                truth_sha256={r.series:file_hash(args.manifest.parent/r.truth) for r in truth_manifest.itertuples()},
                source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in source_paths},
                packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','matplotlib']})
    config_path=out/'protocol.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Resume inputs/code/protocol differ; use a new output directory')
        if not args.resume:
            raise ValueError('Output already initialized; use --resume')
    else:
        atomic_json(config_path,config)  # Freeze group definitions before generating/scoring forecasts.
    progress=dict(status='building_embedding',started_utc=datetime.now(timezone.utc).isoformat(),
                  completed_cases=0, total_cases=0, output_dir=str(out))
    atomic_json(out/'progress.json',progress)
    start=time.perf_counter()
    omit=args.bridge_mode=='omit'
    cache=args.embedding_cache or ROOT/('outputs/paper_embedding_k5_h4_omit' if omit else 'outputs/paper_embedding_k5_h4')
    # Embed one extra future difference, then omit the history/future bridge,
    # exactly as forecasting.embedding_cache.load_cached_library does in full.
    x,y=embeddings(args.library,cache,horizon=4+int(omit))
    if omit:y=y[:,1:]
    index=ExactNeighbors(x,workers=1)
    setup_seconds=time.perf_counter()-start
    print(f'Full library: {len(x)} windows; embedding/index setup {setup_seconds:.1f}s',flush=True)
    panel=pd.read_csv(args.panel_dir/'panel.csv')
    if panel.duplicated(['forecast_origin','series','target_end_date']).any():
        raise ValueError('Duplicate historical observations')
    origins=sorted(panel.forecast_origin.unique())
    expected_origins=provenance['available_origins']
    if origins!=sorted(expected_origins):
        raise ValueError('Panel calendar differs from provenance')
    progress.update(status='forecasting',total_cases=len(origins)*len(truths))
    cases=[]
    checkpoint=out/'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    for i,origin in enumerate(origins):
        path=checkpoint/f'{origin}.json'
        if path.exists():
            completed=json.loads(path.read_text())
            if len(completed)!=len(truths):
                raise ValueError('Incomplete origin checkpoint')
            cases.extend(completed)
        else:
            completed=[]
            for state in truths:
                history=panel.loc[(panel.forecast_origin==origin)&(panel.series==state)].sort_values('target_end_date')
                expected=pd.date_range('2020-01-25',origin,freq='7D').strftime('%Y-%m-%d').tolist()
                if history.target_end_date.tolist()!=expected:
                    raise ValueError(f'Historical availability error: {state} {origin}')
                raw=history.value.to_numpy(dtype=float)
                record=dict(series=state,forecast_origin=origin,**features(raw))
                targets=[(pd.Timestamp(origin)+pd.Timedelta(weeks=h)).date().isoformat() for h in range(1,5)]
                if not all(t in truths[state].index for t in targets):
                    raise ValueError(f'Missing test target: {state} {origin}')
                record.update(targets=targets,actual=[float(truths[state][t]) for t in targets])
                tick=time.perf_counter()
                try:
                    point,_,_=forecast(raw,x,y,closest=4422,smoothing='gam',neighbor_index=index)
                    if not np.isfinite(point).all():
                        raise ValueError('Nonfinite sMOA forecast')
                    record.update(status='ok',prediction=point.tolist())
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    record.update(status='failed',reason=f'{type(exc).__name__}: {exc}',prediction=None)
                record['forecast_seconds']=time.perf_counter()-tick
                completed.append(record)
            atomic_json(path,completed)
            cases.extend(completed)
        progress.update(completed_cases=len(cases),last_origin=origin,
                        failed_cases=sum(c['status']!='ok' for c in cases))
        atomic_json(out/'progress.json',progress)
        print(f'{i+1}/{len(origins)} origins; {len(cases)} cases; last {origin}',flush=True)
    status=pd.DataFrame([{k:v for k,v in c.items() if k not in ['targets','actual','prediction']} for c in cases])
    status.to_csv(out/'case_status.csv',index=False)
    rows=[]
    for case in cases:
        if case['status']!='ok': continue  # Failures stay visible in case_status.
        base={k:v for k,v in case.items() if k not in ['targets','actual','prediction','status','reason']}
        for h,(target,pred,actual) in enumerate(zip(case['targets'],case['prediction'],case['actual']),1):
            rows.append(dict(base,model='Original_sMOA_Python',horizon=h,target_end_date=target,prediction=pred,actual=actual))
    if not rows: raise ValueError('No valid forecasts to evaluate')
    evaluated=errors(pd.DataFrame(rows))
    evaluated.to_csv(out/'instances.csv',index=False)
    progress.update(status='reporting')
    atomic_json(out/'progress.json',progress)
    report(evaluated,status,panel,out,config,truths=truths)
    progress.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
                    setup_seconds=setup_seconds,forecast_count=len(evaluated),
                    artifacts=['instances.csv','case_status.csv','overall.csv','report.md','figures/'])
    atomic_json(out/'progress.json',progress)
    print(pd.read_csv(out/'overall.csv')[['model','n','MAE','RMSE','normalized_MAE','bias','extreme_rate']].to_string(index=False),flush=True)
    print(f'Report: {out/"report.md"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-dir',type=Path,default=ROOT/'outputs/epidemic_panel')
    parser.add_argument('--manifest',type=Path,default=ROOT/'outputs/truth_states/manifest.csv')
    parser.add_argument('--library',type=Path,default=ROOT/'outputs/reproduction_full/01_synthetic/curves.csv')
    parser.add_argument('--embedding-cache',type=Path)
    parser.add_argument('--bridge-mode',choices=['omit','keep'],default='omit')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/experiment_4_omit')
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args)
