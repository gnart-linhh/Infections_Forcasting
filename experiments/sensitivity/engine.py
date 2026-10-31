"""One isolated (seed,N,k) job, exact shared matching and measured aggregation."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
from runtime import atomic_json,file_hash
from neighbors import ExactNeighbors
from library import load_prefix,embed


def aggregate(last,futures):
    increments=np.median(futures,axis=0)
    return np.maximum(1,last+np.cumsum(increments))


def run_group(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    curves,curve_ids=load_prefix(args.packed,args.N)
    prefix_load_seconds=time.perf_counter()-start
    accepted=len(curves)
    tick=time.perf_counter();x,y,codes=embed(curves,curve_ids,args.k,out/'scratch')
    embed_seconds=time.perf_counter()-tick
    del curves
    if max(args.m)>len(x):raise ValueError(f'm exceeds M: {max(args.m)} > {len(x)}')
    tick=time.perf_counter();index=ExactNeighbors(x,workers=1)
    index_seconds=time.perf_counter()-tick
    with np.load(args.context,allow_pickle=False) as data:
        contexts={name:data[name] for name in data.files}
    queries=contexts[f'query_k{args.k}']
    benchmark=np.linspace(0,len(queries)-1,min(32,len(queries)),dtype=int)
    benchmarks={}
    for m in args.m:
        tick=time.perf_counter();index.query(queries[benchmark],m)
        benchmarks[m]=(time.perf_counter()-tick)/len(benchmark)
    forecasts=[];matches=[];used={m:set() for m in args.m};aggregate_seconds=dict.fromkeys(args.m,0.)
    shared_seconds=0.
    for begin in range(0,len(queries),32):
        tick=time.perf_counter();ids,distances=index.query(queries[begin:begin+32],max(args.m))
        shared_seconds+=time.perf_counter()-tick
        for j,(neighbors,distance) in enumerate(zip(ids,distances),begin):
            for m in args.m:
                selected=neighbors[:m];future=np.asarray(y[selected])
                tick=time.perf_counter();point=aggregate(contexts['last'][j],future)
                aggregate_seconds[m]+=time.perf_counter()-tick
                if not np.isfinite(point).all():raise ValueError('Nonfinite point forecast')
                iqr=np.quantile(future,[.25,.75],axis=0)
                mean_path=np.maximum(1,contexts['last'][j]+np.cumsum(np.mean(future,axis=0)))
                used[m].update(codes[selected].tolist())
                base=dict(seed=args.seed,N=args.N,k=args.k,m=m,series=str(contexts['series'][j]),
                    forecast_origin=str(contexts['origin'][j]),normalization=float(contexts['normalization'][j]))
                matches.append(dict(base,minimum_distance=float(distance[0]),median_distance=float(np.median(distance[:m])),
                    maximum_distance=float(distance[m-1]),median_distance_per_difference=float(np.median(distance[:m])/args.k),
                    increment_iqr_mean=float(np.mean(iqr[1]-iqr[0])),mean_median_path_gap=float(np.mean(np.abs(mean_path-point)))))
                for h,pred in enumerate(point,1):
                    actual=float(contexts['actual'][j,h-1])
                    forecasts.append(dict(base,horizon=h,target_end_date=str(contexts['targets'][j,h-1]),
                        prediction=float(pred),actual=actual,error=float(pred-actual)))
    pd.DataFrame(forecasts).to_csv(out/'forecasts.csv.gz',index=False)
    pd.DataFrame(matches).to_csv(out/'matches.csv.gz',index=False)
    # Separate controlled history-availability study, never pooled into main accuracy.
    early=[];availability=[]
    if args.seed==0 and args.N==18387:
        positions=contexts['early_positions']
        for length in [4,6,8,10]:
            eligible=length>=max(args.k+1,5)
            for j in positions:
                availability.append(dict(k=args.k,history_length=length,series=str(contexts['series'][j]),
                    forecast_origin=str(contexts['origin'][j]),status='ok' if eligible else 'insufficient_history',
                    required_history=max(args.k+1,5)))
            if not eligible:continue
            early_ids,_=index.query(contexts[f'early_L{length}_k{args.k}'],4422)
            for j,neighbors in zip(positions,early_ids):
                point=aggregate(contexts['last'][j],y[neighbors])
                for h,pred in enumerate(point,1):
                    actual=float(contexts['actual'][j,h-1])
                    early.append(dict(seed=0,N=18387,k=args.k,m=4422,history_length=length,
                        series=str(contexts['series'][j]),forecast_origin=str(contexts['origin'][j]),horizon=h,
                        target_end_date=str(contexts['targets'][j,h-1]),normalization=float(contexts['normalization'][j]),
                        prediction=float(pred),actual=actual,error=float(pred-actual)))
        pd.DataFrame(early).to_csv(out/'early_forecasts.csv.gz',index=False)
        pd.DataFrame(availability).to_csv(out/'early_availability.csv',index=False)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    stats=dict(seed=args.seed,N=args.N,k=args.k,M=len(x),accepted_curves=accepted,
        prefix_load_seconds=prefix_load_seconds,
        embedding_seconds=embed_seconds,index_seconds=index_seconds,
        shared_query_seconds=shared_seconds,shared_query_max_m=max(args.m),cases=len(queries),
        group_wall_seconds=time.perf_counter()-start,peak_rss_bytes=int(peak),
        per_m={m:dict(aggregation_seconds=aggregate_seconds[m],query_benchmark_mean_seconds=benchmarks[m],
            unique_matched_curves=len(used[m]),matched_curve_fraction=len(used[m])/accepted) for m in args.m})
    atomic_json(out/'runtime.json',stats)
    # These are private scratch files produced by this job, not library inputs.
    del index,x,y,codes
    for name in ['x','y','curve_ids']:(out/'scratch'/f'{name}.dat').unlink()
    (out/'scratch').rmdir()
    artifacts=[out/'forecasts.csv.gz',out/'matches.csv.gz',out/'runtime.json',
               *out.glob('early_*.csv*')]
    atomic_json(out/'complete.json',dict(status='complete',sha256={p.name:file_hash(p) for p in artifacts}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['out','packed','context']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['seed','N','k']:p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--m',type=int,nargs='+',required=True)
    run_group(p.parse_args())
