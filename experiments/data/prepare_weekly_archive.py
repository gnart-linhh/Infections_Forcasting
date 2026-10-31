"""Complete weekly JHU vintage archive for a shared 50-state, four-horizon test."""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import URLError

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from runtime import acquire_run_lock,atomic_json,file_hash
from fetch_paper_data import download,weekly_states


def common_schedule(manifest,start='2020-08-15',horizon=4):
    entries=pd.read_csv(manifest)
    if len(entries)!=50 or entries.series.nunique()!=50:
        raise ValueError('Full benchmark requires exactly 50 unique states')
    endings=[]
    for row in entries.itertuples():
        frame=pd.read_csv(manifest.parent/row.truth)
        dates=pd.DatetimeIndex(pd.to_datetime(frame.target_end_date))
        if dates[0]!=pd.Timestamp('2020-01-25') or not dates.equals(pd.date_range(dates[0],dates[-1],freq='7D')):
            raise ValueError(f'Unexpected truth calendar: {row.series}')
        endings.append(dates[-1])
    end=min(endings)-pd.Timedelta(weeks=horizon)
    origins=pd.date_range(start,end,freq='7D')
    if len(origins)==0 or any(origins.dayofweek!=5):
        raise ValueError('Need a nonempty Saturday origin calendar')
    return entries,[d.date().isoformat() for d in origins]


def prepare(manifest,out,index_path,caches,start='2020-08-15'):
    out.mkdir(parents=True,exist_ok=True)
    with acquire_run_lock(out/'run.lock'):
        return _prepare(manifest,out,index_path,caches,start)


def _prepare(manifest,out,index_path,caches,start):
    entries,origins=common_schedule(manifest,start)
    config=dict(version=1,origins=origins,states=entries.series.tolist(),index_sha256=file_hash(index_path),
                truth_sha256={r.series:file_hash(manifest.parent/r.truth) for r in entries.itertuples()},
                source_sha256={p.name:file_hash(p) for p in [Path(__file__),Path(__file__).with_name('fetch_paper_data.py')]})
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=config:
            raise ValueError('Archive inputs/calendar/code changed; use a different archive directory')
    else: atomic_json(out/'config.json',config)
    if (out/'provenance.json').exists():
        meta=json.loads((out/'provenance.json').read_text())
        if meta.get('status')=='complete':
            for name,digest in meta['artifacts_sha256'].items():
                if file_hash(out/name)!=digest: raise ValueError(f'Archive artifact changed: {name}')
            print(f'Complete weekly archive verified: {len(origins)} origins, {len(entries)} states',flush=True)
            return meta
    checkpoint_path=out/'checkpoint.json'
    completed=json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    links=pd.read_csv(index_path).sort_values('date')
    (out/'vintages').mkdir(exist_ok=True)
    source_rows=[]
    for i,origin in enumerate(origins):
        requested=(pd.Timestamp(origin)+pd.Timedelta(days=1)).date().isoformat()
        candidates=links.loc[links.date<=requested]
        if candidates.empty or requested>links.date.max():
            raise ValueError(f'Missing indexed source vintage for {origin}')
        selected=candidates.iloc[-1]
        if (pd.Timestamp(requested)-pd.Timestamp(selected.date)).days>2:
            raise ValueError(f'Stale vintage for {origin}')
        normalized=out/f'vintages/{origin}.csv'
        record=completed.get(origin)
        if record:
            if file_hash(normalized)!=record['weekly_sha256']:
                raise ValueError(f'Modified normalized vintage: {origin}')
        else:
            name=f'jhu_us_cases_{selected.date}.csv'
            raw=next((c/'raw'/name for c in caches if (c/'raw'/name).exists()),out/'raw'/name)
            log=[]
            for attempt in range(3):
                try:
                    download(selected.file_link,raw,log)
                    break
                except (URLError,TimeoutError):
                    if attempt==2: raise
                    time.sleep(2)
            weekly=weekly_states(raw,entries.series.tolist())
            weekly=weekly.loc[weekly.target_end_date<=pd.Timestamp(origin)].copy()
            expected=pd.date_range('2020-01-25',origin,freq='7D')
            if weekly.series.nunique()!=50: raise ValueError('Missing state in JHU vintage')
            for state,group in weekly.groupby('series'):
                if not pd.DatetimeIndex(group.target_end_date).equals(expected):
                    raise ValueError(f'Missing weekly history: {state}, {origin}')
            temporary=normalized.with_suffix('.tmp')
            weekly.to_csv(temporary,index=False)
            temporary.replace(normalized)
            record=dict(forecast_origin=origin,data_cutoff=requested,
                        issuance_date=(pd.Timestamp(origin)+pd.Timedelta(days=2)).date().isoformat(),
                        source_date=selected.date,source_url=selected.file_link,
                        raw_file=str(raw.resolve()),raw_sha256=file_hash(raw),weekly_sha256=file_hash(normalized))
            completed[origin]=record
            atomic_json(checkpoint_path,completed)
        source_rows.append(record)
        atomic_json(out/'progress.json',dict(status='preparing',completed_origins=i+1,total_origins=len(origins),last_origin=origin))
        print(f'Archive {i+1}/{len(origins)}: {origin}',flush=True)
    # Build each state's strict SnapshotTruth input. No stale/final substitution.
    grouped={state:[] for state in entries.series}
    for origin in origins:
        vintage=pd.read_csv(out/f'vintages/{origin}.csv')
        for state,group in vintage.groupby('series'):
            group=group[['target_end_date','value']].copy()
            group.insert(0,'as_of',origin)
            grouped[state].append(group)
    manifest_rows=[]
    for i,row in enumerate(entries.itertuples(),1):
        path=out/f'{i:02d}_asof.csv'
        pd.concat(grouped[row.series],ignore_index=True).to_csv(path,index=False)
        manifest_rows.append(dict(series=row.series,truth=str((manifest.parent/row.truth).resolve()),as_of_truth=path.name))
    pd.DataFrame(manifest_rows).to_csv(out/'manifest.csv',index=False)
    pd.DataFrame(source_rows).to_csv(out/'origins.csv',index=False)
    artifacts=['manifest.csv','origins.csv']+[r['as_of_truth'] for r in manifest_rows]
    meta=dict(status='complete',states=50,origins=len(origins),start=origins[0],end=origins[-1],
              targets_per_model=50*len(origins)*4,source_index_sha256=config['index_sha256'],
              timing='Saturday last observed week, Sunday origin+1 data cutoff, Monday issuance (LANL convention)',
              aggregation='Raw county cumulative -> daily differences -> Sunday-Saturday state sums; retain negative revisions',
              boundary='All weekly origins in the common final-truth window with four future targets for every state. Arizona-only later dates excluded.',
              artifacts_sha256={name:file_hash(out/name) for name in artifacts})
    atomic_json(out/'provenance.json',meta)
    atomic_json(out/'progress.json',dict(status='complete',completed_origins=len(origins),total_origins=len(origins)))
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/truth_states/manifest.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/paper_weekly_archive')
    p.add_argument('--index',type=Path,default=ROOT/'outputs/paper_data/raw/jhu_us_cases_data_links.csv')
    p.add_argument('--caches',type=Path,nargs='+',default=[ROOT/'outputs/paper_data',ROOT/'outputs/epidemic_panel'])
    args=p.parse_args()
    print(json.dumps(prepare(args.manifest,args.output_dir,args.index,args.caches),indent=2))
