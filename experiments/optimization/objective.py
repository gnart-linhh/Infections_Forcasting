"""Synthetic holdout MAE objective based on LANL optimize_smoa.R."""
from __future__ import annotations
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing
import shutil
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import make_smoothing_spline

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/synthetic'))
from generator import gen_curve
from bank import generate_ordered,save_packed,load_packed,write_audit,read_audit
sys.path.insert(0,str(ROOT/'experiments'))
from runtime import atomic_json
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
from neighbors import ExactNeighbors
from smoothing import thin_plate_gcv
from model import valid_history_windows

FAMILIES=('sir_rollercoaster','sir_rollercoaster_wiggle','seasonal')

# Sinh ngân hàng đường cong
def generate_bank(n,seed,role,trial,workers=1,on_progress=None):
    """Disjoint SeedSequence namespaces: 0=train, 1=validation, 2=heldout test."""
    curves=[]; audit=[] 
    # Tạo danh sách các tác vụ để tạo ra các đường cong tổng hợp, mỗi tác vụ bao gồm chỉ số của họ đường cong, chỉ số của đường cong trong họ và khóa ngẫu nhiên để tạo ra đường cong
    tasks=[(f,j,[int(seed),int(role),int(trial),f,j]) for f in range(len(FAMILIES)) for j in range(n)] 
    if workers==1:
        # Nếu chỉ có một worker, tạo ra các đường cong tổng hợp một cách tuần tự bằng cách sử dụng hàm generate_one
        generated=(gen_curve(FAMILIES[f],np.random.default_rng(np.random.SeedSequence(key))) for f,j,key in tasks)
    else:
        # Nếu có nhiều worker, tạo ra các đường cong tổng hợp song song bằng cách sử dụng hàm generate_ordered với số lượng worker được chỉ định
        generated=generate_ordered(tasks,workers)
    # Đánh giá từng đường cong được tạo ra, kiểm tra tính hợp lệ của nó và lưu trữ thông tin về việc chấp nhận hay từ chối đường cong đó trong danh sách audit. Nếu đường cong hợp lệ, thêm nó vào danh sách curves.
    for completed,((family_idx,j,key),curve) in enumerate(zip(tasks,generated),1):
        valid=curve is not None and len(curve.ts)>0 and np.all(np.isfinite(curve.ts)) and np.all(curve.ts>0)
        audit.append(dict(role=role,trial=trial,family=FAMILIES[family_idx],index=j,seed_key=key,accepted=bool(valid)))
        if valid: curves.append(curve)
        if on_progress and (completed%500==0 or completed==len(tasks)):
            on_progress(completed,len(tasks),len(curves))
    if not curves: raise ValueError('No valid curves generated')
    return curves,audit

#Biến đường cong thành các mẫu đầu vào và đầu ra
def embeddings(curves,k,h=4,mode='upstream'):
    if mode not in ['upstream','pipeline']: raise ValueError('Unknown embedding mode')
    xs=[];ys=[]
    for curve in curves:
        values=np.maximum(curve.ts,1e-8) 
        width=k+1+h+(mode=='upstream')
        if len(values)<width: continue
        windows=np.lib.stride_tricks.sliding_window_view(values,width) # Tạo cửa sổ trượt
        windows=windows[valid_history_windows(windows[:,:k+1])]
        xs.append(np.diff(windows[:,:k+1],axis=1)) # Sai phân đầu vào
        # optimize_smoa.R uses diff(y), skipping the X-to-y bridge. Preserve this
        # difference from run_smoa.R unless the pipeline objective is requested.
        ys.append(np.diff(windows[:,k+1:],axis=1) if mode=='upstream' 
                  else np.diff(windows[:,k:],axis=1)) # Sai phân đầu ra
    if not xs or not sum(map(len,xs)): raise ValueError('No valid training windows')
    return np.vstack(xs),np.vstack(ys)

# Chọn thời điểm bắt đầu dự báo
def origin_indices(length,k,h,max_origins,min_history=None):
    # R: (k+1):(length(ts)-h-1), expressed as number of known observations.
    indices=np.arange(max(k+1,min_history or 0),length-h) # Tạo một mảng các chỉ số từ k+1 hoặc min_history đến length-h
    if max_origins and len(indices)>max_origins: 
        indices=indices[np.linspace(0,len(indices)-1,max_origins,dtype=int)]
    return indices

def _curve_queries(task):
    curve_id,curve,k,h,max_origins,smoothing,min_history=task
    queries=[];contexts=[]
    # Only a small validation curve goes to workers, never training embeddings.
    # Worker-local basis caches survive across BO trials via the persistent pool.
    # Upstream validation adds 100 to the entire synthetic holdout series.
    values=np.asarray(curve.ts,float)+100
    # Tạo các origin cho mỗi đường cong
    for n in origin_indices(len(values),k,h,max_origins,min_history):
        history=values[:n]
        if smoothing=='gam' and n>=3:
            smooth=thin_plate_gcv(history)
        elif smoothing=='spline' and n>=5:
            t=np.arange(n,dtype=float)
            smooth=make_smoothing_spline(t,history)(t)
        elif smoothing=='none' or n<5:
            smooth=history
        else: raise ValueError('Unknown smoother')
        queries.append(np.diff(smooth[-k-1:]))
        contexts.append((curve_id,curve.curve_type,int(n),history[-1],values[n:n+h]))
    return queries,contexts


@lru_cache(maxsize=1)
def _query_pool(workers):
    return ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))


def evaluate(curves,x,y,k,closest,max_origins=0,smoothing='gam',cap_curves=50,min_history=None,
             workers=1,on_progress=None,on_preprocess=None):
    records=[];queries=[];contexts=[]
    if closest>len(x): raise ValueError('closest exceeds available embedding rows')
    tasks=[(i,c,k,y.shape[1],max_origins,smoothing,min_history) for i,c in enumerate(curves[:cap_curves])]
    prepared=(_query_pool(workers).map(_curve_queries,tasks,chunksize=1)
              if workers>1 and smoothing=='gam' else map(_curve_queries,tasks))
    for done,(curve_queries,curve_contexts) in enumerate(prepared,1):
        queries.extend(curve_queries);contexts.extend(curve_contexts)
        if on_preprocess:on_preprocess(done,len(tasks))
    index=ExactNeighbors(x,workers)
    for start in range(0,len(queries),32):
        neighbors,_=index.query(queries[start:start+32],closest)
        for ids,(curve_id,family,n,last,actuals) in zip(neighbors,contexts[start:start+32]):
            point=np.maximum(0,last+np.cumsum(np.median(y[ids],axis=0)))
            for horizon,pred in enumerate(point,1):
                actual=actuals[horizon-1]
                records.append(dict(curve_id=curve_id,family=family,known_steps=n,horizon=horizon,
                    forecast=float(pred),actual=float(actual),abs_error=float(abs(pred-actual)),
                    baseline=float(last),baseline_abs_error=float(abs(last-actual))))
        if on_progress: on_progress(min(start+32,len(queries)),len(queries))
    if not records: raise ValueError('No holdout forecasts scored')
    return float(np.mean([r['abs_error'] for r in records])),records


class SyntheticObjective:
    def __init__(self,seed=13,max_origins=0,smoothing='gam',mode='upstream',
                 validation_protocol='resampled',validation_curves_per_family=3,min_history=None,
                 workers=1,work_dir=None):
        self.seed=seed;self.max_origins=max_origins;self.smoothing=smoothing;self.mode=mode
        self.validation_protocol=validation_protocol
        self.validation_curves_per_family=validation_curves_per_family;self.min_history=min_history
        self.best_score=float('inf');self.best_bank=None;self.best_audit=None
        self.audit=[]
        self.workers=workers;self.work_dir=Path(work_dir) if work_dir else None
        if self.work_dir and (self.work_dir/'best_state.json').exists():
            state=__import__('json').loads((self.work_dir/'best_state.json').read_text())
            self.best_score=state['score'];self.best_bank=load_packed(self.work_dir/state['bank'])
            self.best_audit=read_audit(self.work_dir/state['audit'])

    def progress(self,trial,phase,**details):
        if self.work_dir:
            atomic_json(self.work_dir/'progress.json',dict(status='running',trial=trial,phase=phase,**details))

    def bank(self,n,role,data_trial,trial):
        if self.work_dir is None:
            return generate_bank(n,self.seed,role,data_trial,workers=self.workers)
        name='train' if role==0 else 'validation'
        packed=self.work_dir/f'working/{name}.npz'
        audit_path=self.work_dir/f'audit/trial_{trial:03d}_{name}.jsonl.gz'
        if packed.exists() and audit_path.exists():
            curves,audit=load_packed(packed),read_audit(audit_path)
        else:
            curves,audit=generate_bank(n,self.seed,role,data_trial,workers=self.workers,
                on_progress=lambda done,total,accepted:self.progress(trial,f'generate_{name}',
                    attempted=done,total_attempts=total,accepted=accepted))
            write_audit(audit_path,audit);save_packed(curves,packed)
        self.audit.append(dict(trial=trial,role=role,attempted=len(audit),accepted=len(curves),
            audit=str(audit_path.relative_to(self.work_dir))))
        atomic_json(self.work_dir/f'audit/trial_{trial:03d}_{name}_summary.json',self.audit[-1])
        return curves,audit

    def __call__(self,params,trial):
        if self.work_dir:
            import json
            marker=self.work_dir/'working/trial.json'
            identity=dict(trial=trial,params=params)
            if not marker.exists() or json.loads(marker.read_text())!=identity:
                for name in ('train.npz','validation.npz'):
                    (marker.parent/name).unlink(missing_ok=True)
                atomic_json(marker,identity)
        data_trial=trial if self.validation_protocol=='resampled' else 0
        train,train_audit=self.bank(params['num_curves'],0,data_trial,trial)
        # A minimum of one only serves explicitly overridden unit-test ranges.
        validation_n=max(1,params['num_curves']//1000) if self.validation_protocol=='resampled' else self.validation_curves_per_family
        validation,validation_audit=self.bank(validation_n,1,data_trial,trial)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,3,data_trial]))
        validation=[validation[i] for i in rng.permutation(len(validation))][:50]
        self.progress(trial,'embedding',training_curves=len(train))
        x,y=embeddings(train,params['k'],mode=self.mode)
        self.progress(trial,'validation',embedding_rows=len(x),validation_curves=len(validation))
        score,records=evaluate(validation,x,y,params['k'],params['closest'],self.max_origins,self.smoothing,
            min_history=self.min_history,workers=self.workers,
            on_progress=lambda done,total:self.progress(trial,'validation',scored_origins=done,total_origins=total),
            on_preprocess=lambda done,total:self.progress(trial,'gam_preprocessing',completed_curves=done,total_curves=total))
        if self.work_dir is None:
            self.audit.extend(dict(entry,evaluation_trial=trial) for entry in train_audit+validation_audit)
        if score<self.best_score:
            self.best_score=score;self.best_bank=train;self.best_audit=train_audit
            if self.work_dir:
                target=self.work_dir/f'best_checkpoint/trial_{trial:03d}.npz';target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix('.tmp');shutil.copyfile(self.work_dir/'working/train.npz',temporary);temporary.replace(target)
                atomic_json(self.work_dir/'best_state.json',dict(score=score,trial=trial,
                    bank=str(target.relative_to(self.work_dir)),audit=f'audit/trial_{trial:03d}_train.jsonl.gz'))
                for previous in target.parent.glob('trial_*.npz'):
                    if previous!=target: previous.unlink()
        return score,dict(training_curves=len(train),validation_curves=len(validation),
            embedding_rows=len(x),scored_forecasts=len(records),validation_baseline_mae=float(np.mean([r['baseline_abs_error'] for r in records])))


def save_library(curves,path,audit):
    path.parent.mkdir(parents=True,exist_ok=True)
    accepted=[r for r in audit if r['accepted']]
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n')
        writer.writerow(['curve_id','seed','curve_type','time_cadence','scale','seasonal','step','value'])
        for curve_id,(curve,entry) in enumerate(zip(curves,accepted),1):
            seed=':'.join(map(str,entry['seed_key']))
            writer.writerows((curve_id,seed,curve.curve_type,curve.time_cadence,curve.scale,curve.seasonal,i,float(value))
                             for i,value in enumerate(curve.ts,1))
