"""E1: limited-history baselines versus sMOA on independent synthetic and vintage COVID data."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time
import warnings

ROOT=Path(__file__).resolve().parents[2]
for path in [ROOT,ROOT/'experiments',ROOT/'experiments/forecasting',ROOT/'experiments/synthetic']:
    sys.path.insert(0,str(path))
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'infections-e1-matplotlib'))
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from runtime import atomic_json,file_hash,acquire_run_lock
from generator import gen_curve
from bank import FAMILIES,load_packed
from neighbors import ExactNeighbors
from smoothing import thin_plate_gcv
from experiments.sensitivity.library import embed

PRIMARY=['Persistence','MA3','Drift','SES','sMOA']
BASELINES=PRIMARY[:4]
SECONDARY=['Holt','ARIMA']
LENGTHS=[6,8,11,16,24]
SEEDS=[0,1,2]
RNG_ROOT=20261104
QUOTA=100
ATTEMPTS=800
N,K,M,H=18387,5,4422,4
RULES=dict(experiment='E1 limited historical data',version=1,
    parameters=dict(N=N,k=K,m=M,h=H,bridge='omit',smoothing='gam'),
    primary=PRIMARY,secondary=SECONDARY,secondary_scope='B T=16,24 only',
    A_lengths=[6,8,11],B_lengths=LENGTHS,train_seeds=SEEDS,test_seeds=SEEDS,
    test_namespace=[RNG_ROOT,310,'test_seed','family','attempt'],attempts_per_cell=ATTEMPTS,quota_per_cell=QUOTA,
    test_eligibility='original generator, weekly counts, first raw >=10 onset, >=28 steps from onset, first 100 eligible per cell',
    real_onset='frozen first >=10 raw count in 2020-04-04 vintage, known before all A forecast issuances',
    real_B='same 50 states x37 E2/E4 origins with >=24 records; last T input observations',
    preprocessing='all model inputs and point predictions floored at 1; scoring labels remain raw',
    SES='known first level; conditional SSE, 101 alpha grid plus local bounded refinements; endpoints; relative tie tolerance 1e-10, smaller alpha',
    ARIMA='d=1, trend=t; (0,1,0),(1,1,0),(0,1,1); finite AICc; state-space ML L-BFGS maxiter200; no fallback',
    masks='all primary including 3 sMOA banks; B common valid blocks across all five T; secondary pairwise and all-model masks across T16/24',
    MASE='mean absolute error / mean abs diff of permitted floored history; zero denominator = undefined',
    inference='1000 whole-curve/state cluster bootstrap draws; mean bank losses, no ensemble; seed20261104; 5% practical margin',
    planned=dict(cases=16600,primary_predictions=464800,secondary_scope_blocks=11000),
    timing='measured baselines fit+predict; per-bank GAM charged from shared preprocessing plus measured batched query+aggregation; index/offline separate')


def digest_array(values):
    return hashlib.sha256(np.asarray(values,dtype='<f8').tobytes()).hexdigest()


def pack(values):
    return np.asarray(values,dtype='<f8').tobytes()


def unpack(blob):
    return np.frombuffer(blob,dtype='<f8').copy()


def ses(history):
    z=np.asarray(history,float)
    if len(z)<2 or not np.isfinite(z).all():raise ValueError('SES requires finite history, n>=2')
    if np.ptp(z)==0:return np.repeat(z[0],H),dict(alpha=0.,flat_objective=True)
    scale=max(float(np.mean(np.abs(z))),1.)
    standardized=z/scale
    def objective(alpha):
        alpha=np.atleast_1d(alpha)
        level=np.full(len(alpha),standardized[0]);loss=np.zeros(len(alpha))
        for observed in standardized[1:]:
            loss+=(observed-level)**2
            level=alpha*observed+(1-alpha)*level
        return loss,level*scale
    grid=np.linspace(0,1,101);loss,_=objective(grid)
    candidates=[(float(a),float(v)) for a,v in zip(grid,loss)]
    for i in range(1,100):
        if loss[i]<=loss[i-1] and loss[i]<=loss[i+1]:
            optimum=minimize_scalar(lambda a:float(objective(a)[0][0]),bounds=(grid[i-1],grid[i+1]),method='bounded')
            if not optimum.success or not np.isfinite(optimum.fun):raise RuntimeError('SES optimizer failed')
            candidates.append((float(optimum.x),float(optimum.fun)))
    minimum=min(v for _,v in candidates)
    tolerance=1e-10*max(abs(minimum),1e-20)
    alpha=min(a for a,v in candidates if v<=minimum+tolerance)
    _,level=objective(alpha)
    return np.repeat(level[0],H),dict(alpha=alpha,conditional_SSE=minimum*scale**2,flat_objective=False)


def baseline(model,raw):
    z=np.maximum(np.asarray(raw,float),1)
    if not np.isfinite(z).all() or len(z)<6:raise ValueError('Invalid E1 history')
    if model=='Persistence':point=np.repeat(z[-1],H);info={}
    elif model=='MA3':point=np.repeat(z[-3:].mean(),H);info=dict(window=3)
    elif model=='Drift':
        slope=(z[-1]-z[0])/(len(z)-1);point=z[-1]+np.arange(1,H+1)*slope;info=dict(slope=float(slope))
    elif model=='SES':point,info=ses(z)
    else:raise ValueError('Unknown baseline')
    info['clipped_predictions']=int(np.count_nonzero(point<1))
    if not np.isfinite(point).all():raise ValueError('Nonfinite forecast block')
    return np.maximum(point,1),info


def synthetic_eligibility(curve):
    if curve is None:return 'generator_rejected',None
    values=np.asarray(curve.ts,float)
    if not len(values) or not np.isfinite(values).all() or np.any(values<=0):return 'invalid_values',None
    if curve.time_cadence!='weekly':return 'monthly',None
    if curve.scale!='counts':return 'proportion',None
    onset=np.flatnonzero(values>=10)
    if not len(onset):return 'no_onset',None
    tau=int(onset[0])
    if len(values)-tau<28:return 'insufficient_followup',tau
    return 'eligible',tau


def connect(out):
    db=sqlite3.connect(Path(out)/'results.sqlite',timeout=60)
    db.execute('PRAGMA foreign_keys=ON')
    return db


def initialize(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS jobs(name TEXT PRIMARY KEY, metadata TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS test_audit(test_seed INTEGER,family INTEGER,attempt INTEGER,seed_key TEXT,
      reason TEXT,selected INTEGER,content_sha256 TEXT,metadata TEXT,PRIMARY KEY(test_seed,family,attempt));
    CREATE TABLE IF NOT EXISTS test_curves(curve_id TEXT PRIMARY KEY,test_seed INTEGER,family TEXT,
      attempt INTEGER,onset INTEGER,values_blob BLOB,content_sha256 TEXT);
    CREATE TABLE IF NOT EXISTS training_hashes(seed INTEGER,curve_id INTEGER,content_sha256 TEXT,
      PRIMARY KEY(seed,curve_id));
    CREATE INDEX IF NOT EXISTS train_hash ON training_hashes(content_sha256);
    CREATE TABLE IF NOT EXISTS cases(case_id INTEGER PRIMARY KEY,domain TEXT,arm TEXT,series TEXT,
      cluster TEXT,family TEXT,test_seed INTEGER,origin TEXT,origin_index INTEGER,T INTEGER,
      onset TEXT,reporting_age INTEGER,window_start TEXT,history BLOB,history_sha256 TEXT,
      last REAL,normalization REAL,mase_den REAL,phase TEXT,case_level TEXT,
      floored_values INTEGER,negative_values INTEGER,query BLOB,gam_seconds REAL,
      data_cutoff TEXT,issuance_date TEXT);
    CREATE TABLE IF NOT EXISTS observations(case_id INTEGER,horizon INTEGER,target TEXT,actual REAL,
      PRIMARY KEY(case_id,horizon),FOREIGN KEY(case_id) REFERENCES cases(case_id));
    CREATE TABLE IF NOT EXISTS fits(model TEXT,bank_seed INTEGER,case_id INTEGER,status TEXT,seconds REAL,
      metadata TEXT,PRIMARY KEY(model,bank_seed,case_id),FOREIGN KEY(case_id) REFERENCES cases(case_id));
    CREATE TABLE IF NOT EXISTS predictions(model TEXT,bank_seed INTEGER,case_id INTEGER,horizon INTEGER,prediction REAL,
      PRIMARY KEY(model,bank_seed,case_id,horizon),FOREIGN KEY(case_id,horizon) REFERENCES observations(case_id,horizon));
    ''')


def completed(db,name):
    row=db.execute('SELECT metadata FROM jobs WHERE name=?',(name,)).fetchone()
    return json.loads(row[0]) if row else None


def mark(db,name,metadata):
    db.execute('INSERT INTO jobs VALUES (?,?)',(name,json.dumps(metadata)))


def generate_test(db):
    for seed in SEEDS:
        for f,family in enumerate(FAMILIES):
            name=f'test_{seed}_{f}'
            if completed(db,name):continue
            tick=time.perf_counter();audit=[];selected=[];eligible=0
            for j in range(ATTEMPTS):
                key=[RNG_ROOT,310,seed,f,j]
                curve=gen_curve(family,np.random.default_rng(np.random.SeedSequence(key)))
                reason,tau=synthetic_eligibility(curve)
                content=digest_array(curve.ts) if curve is not None else None
                take=reason=='eligible' and len(selected)<QUOTA
                eligible+=reason=='eligible'
                info={} if curve is None else dict(cadence=curve.time_cadence,scale=curve.scale,observations=len(curve.ts),onset=tau)
                audit.append((seed,f,j,json.dumps(key),reason,int(take),content,json.dumps(info)))
                if take:selected.append((f'syn_{seed}_{f}_{j}',seed,family,j,tau,pack(curve.ts),content))
            with db:
                db.executemany('INSERT INTO test_audit VALUES (?,?,?,?,?,?,?,?)',audit)
                db.executemany('INSERT INTO test_curves VALUES (?,?,?,?,?,?,?)',selected)
                mark(db,name,dict(selected=len(selected),eligible=eligible,attempts=ATTEMPTS,seconds=time.perf_counter()-tick))
            print(f'Fresh test: seed={seed}, {family}, eligible={eligible}, selected={len(selected)}',flush=True)
            if len(selected)!=QUOTA:raise ValueError('Insufficient prespecified test quota; stop before scoring')
    if db.execute('SELECT COUNT(*) FROM test_curves').fetchone()[0]!=900:raise ValueError('Test quota mismatch')
    groups=db.execute('SELECT test_seed,family,COUNT(*) FROM test_curves GROUP BY test_seed,family').fetchall()
    if len(groups)!=9 or any(n!=QUOTA for _,_,n in groups):raise ValueError('Unbalanced synthetic test')


def make_case(case_id,domain,arm,series,family,test_seed,origin,origin_index,T,onset,reporting_age,
              window_start,raw,data_cutoff='',issuance_date=''):
    raw=np.asarray(raw,float)
    if len(raw)!=T or T not in LENGTHS or not np.isfinite(raw).all():raise ValueError('Invalid permitted history')
    z=np.maximum(raw,1);tick=time.perf_counter()
    fitted=np.maximum(thin_plate_gcv(z),1);query=np.diff(fitted[-K-1:]);gam=time.perf_counter()-tick
    growth=(z[-2:].mean()+1)/(z[-4:-2].mean()+1)-1
    level=float(z[-4:].mean())
    return (case_id,domain,arm,series,domain+':'+series,family,test_seed,str(origin),origin_index,T,str(onset),
        reporting_age,str(window_start),pack(raw),digest_array(z),float(z[-1]),level,float(np.mean(np.abs(np.diff(z)))),
        'rising' if growth>.1 else 'declining' if growth<-.1 else 'stable',
        'low' if level<1000 else 'medium' if level<10000 else 'high',int((raw<1).sum()),int((raw<0).sum()),pack(query),gam,data_cutoff,issuance_date)


def prepare_real(panel_dir,db):
    if completed(db,'real_cases'):return
    panel=pd.read_csv(panel_dir/'panel.csv');timing=pd.read_csv(panel_dir/'origins.csv').set_index('forecast_origin')
    groups={(s,o):g.sort_values('target_end_date') for (s,o),g in panel.groupby(['series','forecast_origin'])}
    origins=sorted(panel.forecast_origin.unique());first=origins[0]
    if first!='2020-04-04':raise ValueError('Unexpected initial onset vintage')
    states=sorted(panel.series.unique())
    manifest=ROOT/'outputs/truth_states/manifest.csv'
    truths={r.series:pd.read_csv(manifest.parent/r.truth).set_index('target_end_date').value for r in pd.read_csv(manifest).itertuples()}
    onsets={s:groups[(s,first)].loc[groups[(s,first)].value.ge(10),'target_end_date'].iloc[0] for s in states}
    common=[o for o in origins if all(len(groups[(s,o)])>=24 for s in states)]
    if len(states)!=50 or len(common)!=37:raise ValueError('Real feasibility changed')
    rows=[];outcomes=[]
    for arm in ['A','B']:
        for s in states:
            schedule=[((pd.Timestamp(onsets[s])+pd.Timedelta(weeks=T-1)).date().isoformat(),T) for T in [6,8,11]] if arm=='A' else [(o,T) for o in common for T in LENGTHS]
            for origin,T in schedule:
                g=groups[(s,origin)]
                hist=g.loc[g.target_end_date.ge(onsets[s])] if arm=='A' else g.tail(T)
                if len(hist)!=T or hist.target_end_date.max()!=origin:raise ValueError('History/date mismatch')
                if pd.Timestamp(timing.loc[first,'issuance_date'])>=pd.Timestamp(timing.loc[origin,'issuance_date']):raise ValueError('Onset metadata from future')
                ident=len(rows)
                age=(pd.Timestamp(origin)-pd.Timestamp(onsets[s])).days//7+1
                rows.append(make_case(ident,'real',arm,s,'',-1,origin,-1,T,onsets[s],age,hist.target_end_date.iloc[0],hist.value,
                                      str(timing.loc[origin,'data_cutoff']),str(timing.loc[origin,'issuance_date'])))
                for h in range(1,H+1):
                    target=(pd.Timestamp(origin)+pd.Timedelta(weeks=h)).date().isoformat()
                    actual=float(truths[s].loc[target])
                    if not np.isfinite(actual):raise ValueError('Missing final scoring label')
                    outcomes.append((ident,h,target,actual))
    if len(rows)!=9400:raise ValueError('Real case count mismatch')
    with db:
        db.executemany('INSERT INTO cases VALUES ('+','.join(['?']*26)+')',rows)
        db.executemany('INSERT INTO observations VALUES (?,?,?,?)',outcomes)
        mark(db,'real_cases',dict(cases=len(rows),onset_vintage=first,common_origins=common))
    print('Prepared real cases:',len(rows),flush=True)


def prepare_synthetic(db):
    if completed(db,'synthetic_cases'):return
    start=db.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
    rows=[];outcomes=[]
    for ident,seed,family,attempt,tau,blob,_ in db.execute('SELECT * FROM test_curves ORDER BY test_seed,family,attempt'):
        values=unpack(blob)
        for arm,lengths in [('A',[6,8,11]),('B',LENGTHS)]:
            for T in lengths:
                origin=tau+T-1 if arm=='A' else tau+23
                raw=values[origin-T+1:origin+1]
                case_id=start+len(rows)
                rows.append(make_case(case_id,'synthetic',arm,ident,family,seed,str(origin+1),origin,T,str(tau+1),origin-tau+1,str(origin-T+2),raw))
                outcomes.extend((case_id,h,str(origin+h+1),float(values[origin+h])) for h in range(1,H+1))
    if len(rows)!=7200:raise ValueError('Synthetic case count mismatch')
    with db:
        db.executemany('INSERT INTO cases VALUES ('+','.join(['?']*26)+')',rows)
        db.executemany('INSERT INTO observations VALUES (?,?,?,?)',outcomes)
        mark(db,'synthetic_cases',dict(cases=len(rows)))
    print('Prepared synthetic cases:',len(rows),flush=True)


def insert_fit(db,model,seed,case_id,status,seconds,metadata,point=None):
    if point is not None and (len(point)!=H or not np.isfinite(point).all() or np.any(np.asarray(point)<1)):
        raise ValueError('Invalid forecast block')
    db.execute('INSERT INTO fits VALUES (?,?,?,?,?,?)',(model,seed,case_id,status,float(seconds),json.dumps(metadata)))
    if point is not None:
        db.executemany('INSERT INTO predictions VALUES (?,?,?,?,?)',[(model,seed,case_id,h,float(value)) for h,value in enumerate(point,1)])


def fit_secondary(model,z):
    """Optional verified statsmodels implementations; missing dependency never falls back."""
    if model=='Holt':
        from statsmodels.tsa.holtwinters import Holt
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=Holt(z,exponential=False,damped_trend=False,initialization_method='estimated').fit(optimized=True,method='L-BFGS-B',minimize_kwargs={'options':{'maxiter':200}})
            point=np.asarray(result.forecast(H),float)
        status=getattr(result,'mle_retvals',{})
        if status is not None and not bool(status.get('success',False)):raise RuntimeError('Holt optimizer did not converge')
        return np.maximum(point,1),dict(warnings=[str(w.message) for w in caught],parameters={k:float(v) for k,v in result.params.items() if np.isscalar(v) and not isinstance(v,(str,bool))},clipped_predictions=int((point<1).sum()))
    from statsmodels.tsa.arima.model import ARIMA
    candidates=[];valid=[]
    for order in [(0,1,0),(1,1,0),(0,1,1)]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result=ARIMA(z,order=order,trend='t',enforce_stationarity=True,enforce_invertibility=True).fit(method='statespace',method_kwargs={'method':'lbfgs','maxiter':200,'disp':0})
                point=np.asarray(result.forecast(H),float)
            if not result.mle_retvals.get('converged',False) or not np.isfinite(np.r_[result.params,point,result.llf,result.aicc]).all():
                raise RuntimeError('Nonconverged or nonfinite candidate')
            info=dict(order=order,status='ok',aicc=float(result.aicc),df_model=float(result.df_model),nobs_effective=float(result.nobs_effective),
                      parameters=result.params.tolist(),warnings=[str(w.message) for w in caught])
            valid.append((float(result.aicc),len(result.params),len(candidates),point,info));candidates.append(info)
        except Exception as exc:candidates.append(dict(order=order,status='fit_failed',reason=str(exc)))
    if not valid:return None,dict(status='no_valid_candidate',candidates=candidates)
    best=min(v[0] for v in valid);tied=[v for v in valid if v[0]<=best+1e-10*max(abs(best),1.)]
    selected=min(tied,key=lambda v:(v[1],v[2]));point=selected[3]
    return np.maximum(point,1),dict(candidates=candidates,selected=selected[4],clipped_predictions=int((point<1).sum()))


def run_baselines(db,out):
    rows=db.execute('SELECT case_id,arm,T,history FROM cases ORDER BY case_id').fetchall()
    available=importlib.util.find_spec('statsmodels') is not None
    for model in BASELINES+SECONDARY:
        done={r[0] for r in db.execute('SELECT case_id FROM fits WHERE model=? AND bank_seed=-1',(model,))}
        for j,(case_id,arm,T,blob) in enumerate(rows):
            if case_id in done:continue
            tick=time.perf_counter();info={};point=None
            if model in SECONDARY and (arm!='B' or T not in [16,24]):status='outside_secondary_scope'
            elif model in SECONDARY and not available:status='dependency_missing'
            else:
                try:
                    raw=unpack(blob)
                    point,info=baseline(model,raw) if model in BASELINES else fit_secondary(model,np.maximum(raw,1))
                    status=info.get('status','ok')
                    if point is not None and not np.isfinite(point).all():raise ValueError('Nonfinite forecast')
                except Exception as exc:status='fit_failed';point=None;info=dict(reason=str(exc))
            insert_fit(db,model,-1,case_id,status,time.perf_counter()-tick,info,point)
            if j%250==249:db.commit()
        db.commit()
        print('Baseline finished:',model,flush=True)
        atomic_json(out/'progress.json',dict(status='running',phase='baselines',model=model,fit_blocks=db.execute('SELECT COUNT(*) FROM fits').fetchone()[0]))


def smoa_worker(args):
    out=args.output_dir;seed=args.bank_seed;db=connect(out)
    if json.loads((out/'protocol.json').read_text())['engine_signature']!=engine_signature():raise ValueError('Worker code changed')
    name=f'smoa_bank_{seed}'
    if completed(db,name):return
    tick=time.perf_counter();packed=args.e2_dir/f'packed/seed_{seed}'
    if seed==0:
        cache=ROOT/'outputs/paper_embedding_k5_h4_omit';meta=json.loads((cache/'metadata.json').read_text())
        x=np.memmap(cache/'x.dat',dtype='<f8',mode='r',shape=(meta['rows'],K))
        full_y=np.memmap(cache/'y.dat',dtype='<f8',mode='r',shape=(meta['rows'],H+1));y=full_y[:,1:]
        embedding_kind='verified existing cache load'
    else:
        curves=load_packed(packed/'curves.npz')
        with np.load(packed/'attempts.npz',allow_pickle=False) as a:curve_ids=a['curve_id']
        x,y,codes=embed(curves,curve_ids,K,out/'scratch',h=H)
        del curves
        embedding_kind='fresh embedding from existing packed bank'
    embedding_seconds=time.perf_counter()-tick
    tick=time.perf_counter();index=ExactNeighbors(x,workers=1);index_seconds=time.perf_counter()-tick
    done={r[0] for r in db.execute("SELECT case_id FROM fits WHERE model='sMOA' AND bank_seed=?",(seed,))}
    rows=[r for r in db.execute('SELECT case_id,last,query,gam_seconds FROM cases ORDER BY case_id') if r[0] not in done]
    query_seconds=0.;aggregation_seconds=0.
    for begin in range(0,len(rows),32):
        batch=rows[begin:begin+32];queries=np.stack([unpack(r[2]) for r in batch])
        tick=time.perf_counter();ids,distances=index.query(queries,M);batch_seconds=time.perf_counter()-tick
        query_seconds+=batch_seconds
        with db:
            for (case_id,last,query,gam),neighbors,distance in zip(batch,ids,distances):
                tick=time.perf_counter();raw_point=last+np.cumsum(np.median(y[neighbors],axis=0));point=np.maximum(raw_point,1)
                aggregation=time.perf_counter()-tick;aggregation_seconds+=aggregation
                info=dict(minimum_distance=float(distance[0]),median_distance=float(np.median(distance)),
                          query_amortized_seconds=batch_seconds/len(batch),aggregation_seconds=aggregation,
                          gam_seconds_charged=gam,clipped_predictions=int((raw_point<1).sum()),query_batch_size=len(batch))
                insert_fit(db,'sMOA',seed,case_id,'ok',gam+batch_seconds/len(batch)+aggregation,info,point)
        if begin%1024==0:print(f'sMOA seed={seed}: {min(begin+len(batch),len(rows))}/{len(rows)} new cases',flush=True)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    with db:mark(db,name,dict(seed=seed,rows=len(x),embedding_kind=embedding_kind,embedding_seconds=embedding_seconds,
          index_seconds=index_seconds,query_seconds=query_seconds,aggregation_seconds=aggregation_seconds,peak_rss_bytes=int(peak)))
    del index,x,y
    if seed!=0:
        del codes
        for name in ['x','y','curve_ids']:(out/'scratch'/f'{name}.dat').unlink()
        (out/'scratch').rmdir()
    db.close()


def training_audit(db,e2):
    for seed in SEEDS:
        if completed(db,f'train_hashes_{seed}'):continue
        folder=e2/f'packed/seed_{seed}';curves=load_packed(folder/'curves.npz')
        with np.load(folder/'attempts.npz',allow_pickle=False) as a:
            ids=a['curve_id'];attempts=a['index'];families=a['family']
        if not np.array_equal(families,(ids-1)//N) or not np.array_equal(attempts,(ids-1)%N):raise ValueError('Wrong legacy training seed namespace')
        if RNG_ROOT<=2000000+3*N:raise ValueError('Fresh scalar root must be outside training scalar seed range')
        with db:
            db.executemany('INSERT INTO training_hashes VALUES (?,?,?)',[(seed,int(i),digest_array(c.ts)) for i,c in zip(ids,curves)])
            mark(db,f'train_hashes_{seed}',dict(curves=len(curves),generation_timing=json.loads((folder/'metadata.json').read_text())))


def engine_signature():
    tree=ast.parse(Path(__file__).read_text())
    names=['ses','baseline','synthetic_eligibility','generate_test','make_case','prepare_real','prepare_synthetic',
           'insert_fit','fit_secondary','run_baselines','smoa_worker','training_audit']
    nodes=[ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    return hashlib.sha256(json.dumps([RULES,nodes],sort_keys=True).encode()).hexdigest()


def validate_inputs(args):
    paths=[args.panel_dir/'panel.csv',args.panel_dir/'origins.csv',args.panel_dir/'provenance.json',args.e2_dir/'protocol.json']
    original=json.loads((args.e2_dir/'protocol.json').read_text())
    for path,digest in original['inputs_sha256'].items():
        if file_hash(Path(path))!=digest:raise ValueError('Modified E2 input')
        paths.append(Path(path))
    prov=json.loads((args.panel_dir/'provenance.json').read_text())
    if file_hash(args.panel_dir/'panel.csv')!=prov['panel_sha256']:raise ValueError('Modified vintage panel')
    for seed in SEEDS:
        folder=args.e2_dir/f'packed/seed_{seed}';meta=json.loads((folder/'metadata.json').read_text())
        for name,key in [('curves.npz','packed_sha256'),('attempts.npz','attempts_sha256')]:
            if file_hash(folder/name)!=meta[key]:raise ValueError('Modified training bank')
            paths.append(folder/name)
        if meta['seed_offset']!=seed*1000000 or meta['attempts']!=3*N:raise ValueError('Wrong original budget')
        paths.append(folder/'metadata.json')
        if file_hash(Path(meta['source']))!=meta['source_sha256']:raise ValueError('Modified source bank')
        paths.append(Path(meta['source']))
    cache=ROOT/'outputs/paper_embedding_k5_h4_omit';meta=json.loads((cache/'metadata.json').read_text())
    if meta['k']!=K or meta['horizon']!=H+1 or meta['library_sha256']!=json.loads((args.e2_dir/'packed/seed_0/metadata.json').read_text())['source_sha256']:
        raise ValueError('Wrong omit-bridge cache')
    for name in ['x','y']:
        if file_hash(cache/f'{name}.dat')!=meta[name+'_sha256']:raise ValueError('Modified original cache')
        paths.append(cache/f'{name}.dat')
    paths.append(cache/'metadata.json')
    for name in ['experiments/synthetic/generator.py','experiments/synthetic/bank.py','experiments/sensitivity/library.py',
                 'experiments/forecasting/model.py','experiments/forecasting/smoothing.py','experiments/forecasting/neighbors.py','experiments/runtime.py']:
        paths.append(ROOT/name)
    return {str(p.resolve()):file_hash(p) for p in set(paths)}


def common_cases(cases,fits,expected):
    """A shared complete-block mask, including the same units across all allowed T."""
    expected=set(expected)
    chosen=fits.loc[[ (m,int(s)) in expected for m,s in zip(fits.model,fits.bank_seed) ]].copy()
    if chosen.duplicated(['model','bank_seed','case_id']).any():raise ValueError('Duplicate fit status')
    counts=chosen.loc[chosen.status.eq('ok')].groupby('case_id').size()
    mask=cases[['case_id','domain','arm','series','origin','T']].copy()
    mask['valid']=mask.case_id.map(counts).fillna(0).eq(len(expected))
    mask['block_origin']=np.where(mask.arm.eq('B'),mask.origin,'episode')
    for (domain,arm),g in mask.groupby(['domain','arm']):
        needed=set(g['T'])
        for _,block in g.groupby(['series','block_origin']):
            if set(block['T'])!=needed or len(block)!=len(needed) or not block.valid.all():
                mask.loc[block.index,'valid']=False
    return set(mask.loc[mask.valid,'case_id'])


def score_metrics(g):
    error=g.prediction-g.actual;absolute=error.abs();valid=g.mase_den.gt(0)
    return dict(n=len(g),cases=g.case_id.nunique(),MAE=float(absolute.mean()),RMSE=float(np.sqrt(np.mean(error**2))),
        bias=float(error.mean()),normalized_MAE=float((absolute/g.normalization).mean()),
        MASE=float((absolute[valid]/g.loc[valid,'mase_den']).mean()) if valid.any() else np.nan,
        MASE_defined=int(valid.sum()),MASE_undefined=int((~valid).sum()),negative_actuals=int(g.actual.lt(0).sum()))


def summarize_scores(data,scope='primary_common'):
    rows=[]
    for (domain,arm,T,model,seed),frame in data.groupby(['domain','arm','T','model','bank_seed'],sort=True):
        strata=[('overall','all',frame)]
        for dimension in ['horizon','phase','case_level']+(['family','test_seed'] if domain=='synthetic' else []):
            strata.extend((dimension,str(label),g) for label,g in frame.groupby(dimension,sort=True))
        for dimension,label,g in strata:
            rows.append(dict(scope=scope,domain=domain,arm=arm,T=T,model=model,bank_seed=seed,
                dimension=dimension,group=label,statistic='bank' if model=='sMOA' else 'mean',seeds=1,**score_metrics(g)))
    scores=pd.DataFrame(rows)
    if scores.empty:return scores
    means=[]
    for key,g in scores.loc[scores.model.eq('sMOA')].groupby(['scope','domain','arm','T','model','dimension','group'],sort=True):
        if sorted(g.bank_seed)!=SEEDS:raise ValueError('Do not average losses from an incomplete bank set')
        row=dict(zip(['scope','domain','arm','T','model','dimension','group'],key),bank_seed=-2,statistic='mean',seeds=3)
        for field in ['n','cases','MASE_defined','MASE_undefined','negative_actuals']:row[field]=int(g[field].iloc[0])
        for field in ['MAE','RMSE','bias','normalized_MAE','MASE']:
            row[field]=float(g[field].mean());row[field+'_seed_sd']=float(g[field].std(ddof=1))
        means.append(row)
    return pd.concat([scores,pd.DataFrame(means)],ignore_index=True)


def paired_bootstrap(frame,left='smoa_loss',right='baseline_loss'):
    blocks=frame.groupby('cluster')[[left,right]].sum()
    sizes=frame.groupby('cluster').size().reindex(blocks.index).to_numpy(float)
    delta=float((frame[left]-frame[right]).mean());den=float(frame[right].mean())
    if len(blocks)<2:return dict(delta_MAE=delta,lo=np.nan,hi=np.nan,relative_delta=delta/den if den else np.nan,
                                relative_lo=np.nan,relative_hi=np.nan,clusters=len(blocks),practical_equivalence=False)
    rng=np.random.default_rng(RNG_ROOT);draw=rng.integers(0,len(blocks),size=(1000,len(blocks)))
    values=blocks.to_numpy()[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
    differences=values[:,0]-values[:,1]
    ratios=np.divide(differences,values[:,1],out=np.full(1000,np.nan),where=values[:,1]>0)
    lo,hi=np.quantile(differences,[.025,.975])
    rel_lo,rel_hi=np.nanquantile(ratios,[.025,.975]) if np.isfinite(ratios).any() else (np.nan,np.nan)
    return dict(delta_MAE=delta,lo=float(lo),hi=float(hi),relative_delta=delta/den if den else np.nan,
        relative_lo=float(rel_lo),relative_hi=float(rel_hi),clusters=len(blocks),
        practical_equivalence=bool(rel_lo>=-.05 and rel_hi<=.05))


def paired_tables(data,scope='primary_common'):
    keys=['case_id','horizon']
    smoa=data.loc[data.model.eq('sMOA')].copy()
    smoa['smoa_loss']=(smoa.prediction-smoa.actual).abs()
    mean=smoa.groupby(keys,as_index=False).smoa_loss.mean()
    count=smoa.groupby(keys).bank_seed.nunique()
    if not count.eq(3).all():raise ValueError('Missing bank in paired loss average')
    baseline=data.loc[~data.model.eq('sMOA')].copy()
    paired=baseline.merge(mean,on=keys,validate='many_to_one')
    paired['baseline_loss']=(paired.prediction-paired.actual).abs()
    rows=[]
    for (domain,arm,T,model),frame in paired.groupby(['domain','arm','T','model']):
        strata=[('overall','all',frame)]
        for dimension in ['horizon','phase','case_level']:
            strata.extend((dimension,str(label),g) for label,g in frame.groupby(dimension))
        for dimension,label,g in strata:
            rows.append(dict(scope=scope,domain=domain,arm=arm,T=T,baseline=model,dimension=dimension,group=label,
                n=len(g),**paired_bootstrap(g)))
    return pd.DataFrame(rows)


def history_changes(data):
    frame=data.loc[data.arm.eq('B')].copy()
    frame['loss']=(frame.prediction-frame.actual).abs()
    one=frame.loc[frame.model.ne('sMOA'),['domain','series','origin','cluster','horizon','T','model','loss']]
    sm=frame.loc[frame.model.eq('sMOA')].groupby(['domain','series','origin','cluster','horizon','T','model'],as_index=False).loss.mean()
    one=pd.concat([one,sm],ignore_index=True)
    keys=['domain','series','origin','cluster','horizon','model']
    pairs=one.loc[one['T'].eq(24)].merge(one.loc[one['T'].eq(6)],on=keys,validate='one_to_one',suffixes=('_24','_6'))
    rows=[]
    for (domain,model),g in pairs.groupby(['domain','model']):
        rows.append(dict(scope='history_24_minus_6',domain=domain,arm='B',T=24,baseline=model,dimension='overall',group='all',
                         n=len(g),**paired_bootstrap(g,'loss_24','loss_6')))
    return pd.DataFrame(rows)


def availability_table(cases,fits):
    data=fits.merge(cases[['case_id','domain','arm','T']],on='case_id',validate='many_to_one')
    groups=['domain','arm','T','model','bank_seed']
    rows=[]
    for key,g in data.groupby(groups,sort=True):
        counts=g.status.value_counts();eligible=~g.status.eq('outside_secondary_scope')
        rows.append(dict(zip(groups,key),planned_blocks=len(g),eligible_blocks=int(eligible.sum()),
            valid_blocks=int(g.status.eq('ok').sum()),valid_rate=float(g.status.eq('ok').sum()/eligible.sum()) if eligible.any() else np.nan,
            outside_secondary_scope=int(counts.get('outside_secondary_scope',0)),dependency_missing=int(counts.get('dependency_missing',0)),
            fit_failed=int(counts.get('fit_failed',0)),no_valid_candidate=int(counts.get('no_valid_candidate',0))))
    return pd.DataFrame(rows)


def runtime_table(db,cases,fits):
    joined=fits.merge(cases[['case_id','domain','arm','T']],on='case_id',validate='many_to_one')
    rows=[]
    active=joined.loc[joined.status.isin(['ok','fit_failed','no_valid_candidate'])]
    for key,g in active.groupby(['domain','arm','T','model','bank_seed']):
        rows.append(dict(zip(['domain','arm','T','model','bank_seed'],key),kind='online',cases=len(g),
            seconds=float(g.seconds.sum()),seconds_per_case=float(g.seconds.mean()),
            timing='GAM charged + measured batched search/aggregation' if key[3]=='sMOA' else 'measured fit+prediction'))
    for seed in SEEDS:
        meta=completed(db,f'smoa_bank_{seed}')
        rows.append(dict(domain='all',arm='all',T=0,model='sMOA',bank_seed=seed,kind='embedding_and_index',cases=0,
            seconds=meta['embedding_seconds']+meta['index_seconds'],seconds_per_case=np.nan,timing=meta['embedding_kind'],peak_rss_bytes=meta['peak_rss_bytes']))
        legacy=completed(db,f'train_hashes_{seed}')['generation_timing']
        rows.append(dict(domain='all',arm='all',T=0,model='sMOA',bank_seed=seed,kind='historical_generation_or_replay',cases=0,
            seconds=legacy.get('generation_export_seconds'),seconds_per_case=np.nan,
            timing='historical replay/export, not original generation' if seed==0 else 'historical generation/export'))
    rows.append(dict(domain='all',arm='all',T=0,model='GAM shared',bank_seed=-1,kind='shared_query_preprocessing',cases=len(cases),
        seconds=float(cases.gam_seconds.sum()),seconds_per_case=float(cases.gam_seconds.mean()),timing='physically computed once, charged to each deployed bank'))
    return pd.DataFrame(rows)


def markdown_table(frame,columns):
    result='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    for row in frame[columns].itertuples(index=False,name=None):
        result+='| '+' | '.join(f'{v:,.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |\n'
    return result


def plots(out,scores,pairs,availability,runtime):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(PRIMARY,['tab:gray','tab:orange','tab:green','tab:purple','tab:blue']))
    directory=out/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    overall=scores.loc[scores.statistic.eq('mean') & scores.dimension.eq('overall') & scores.scope.eq('primary_common')]
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(directory/f'{name}.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig)
    for arm,name,title in [('A','early_stage','A: growing reporting history (origins/targets change with T)'),('B','fixed_origin_history','B: same origin and targets; only permitted history changes')]:
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for i,domain in enumerate(['synthetic','real']):
            for j,metric in enumerate(['MAE','normalized_MAE']):
                for model in PRIMARY:
                    g=overall.loc[overall.domain.eq(domain)&overall.arm.eq(arm)&overall.model.eq(model)].sort_values('T')
                    axes[i,j].plot(g['T'],g[metric],'o-',color=colors[model],label=model)
                axes[i,j].set(xlabel='Permitted observations T',ylabel=metric,title=domain,
                               xticks=[6,8,11] if arm=='A' else LENGTHS)
                axes[i,j].legend(fontsize=7)
        fig.suptitle(title+'\nsMOA = mean bank-specific scores; no ensemble; common valid units across T')
        save(fig,name)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    from matplotlib.colors import LogNorm
    for i,T in enumerate([6,24]):
        for j,domain in enumerate(['synthetic','real']):
            g=scores.loc[scores.statistic.eq('mean')&scores.dimension.eq('horizon')&scores.domain.eq(domain)&scores.arm.eq('B')&scores['T'].eq(T)&scores.scope.eq('primary_common')]
            panel=g.pivot(index='model',columns='group',values='normalized_MAE').reindex(PRIMARY)
            values=panel.to_numpy(float)
            image=axes[i,j].imshow(np.maximum(values,1e-8),norm=LogNorm(vmin=max(float(np.nanmin(values)),1e-8),vmax=max(float(np.nanmax(values)),1e-7)),aspect='auto',cmap='magma_r')
            axes[i,j].set(xticks=range(4),xticklabels=[1,2,3,4],yticks=range(5),yticklabels=PRIMARY,title=f'{domain}, T={T}',xlabel='Horizon (weeks)')
            for (a,b),value in np.ndenumerate(values):axes[i,j].text(b,a,f'{value:.3g}',ha='center',va='center',fontsize=8,color='white' if value>np.sqrt(np.nanmin(values)*np.nanmax(values)) else 'black')
            fig.colorbar(image,ax=axes[i,j],label='Normalized MAE (log colors)')
    fig.suptitle('Fixed-origin horizon sensitivity; each panel has its own color range, labels are exact metrics')
    save(fig,'horizon_errors')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,domain in enumerate(['synthetic','real']):
        for j,arm in enumerate(['A','B']):
            for model in BASELINES:
                g=pairs.loc[pairs.scope.eq('primary_common')&pairs.dimension.eq('overall')&pairs.domain.eq(domain)&pairs.arm.eq(arm)&pairs.baseline.eq(model)].sort_values('T')
                axes[i,j].plot(g['T'],100*g.relative_delta,'o-',color=colors[model],label=model)
                axes[i,j].fill_between(g['T'],100*g.relative_lo,100*g.relative_hi,color=colors[model],alpha=.1)
            axes[i,j].axhspan(-5,5,color='gray',alpha=.1);axes[i,j].axhline(0,color='black',lw=1,ls='--')
            axes[i,j].set(title=f'{domain}, arm {arm}',xlabel='Permitted observations T',ylabel='(MAE sMOA − baseline) / baseline (%)')
            axes[i,j].legend(fontsize=7)
    fig.suptitle('Paired skill: negative favors sMOA; 95% whole-unit bootstrap bands, ±5% practical region')
    save(fig,'paired_skill')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    a=availability.loc[availability.model.isin(PRIMARY)].groupby(['model','bank_seed'])[['valid_blocks','eligible_blocks']].sum()
    labels=[f'{m}'+(f' bank{s}' if m=='sMOA' else '') for m,s in a.index]
    axes[0].barh(labels,100*a.valid_blocks/a.eligible_blocks,color=[colors[m] for m,s in a.index]);axes[0].set(xlim=(0,105),xlabel='Valid blocks / planned primary blocks (%)',title='Primary availability')
    a=availability.loc[availability.model.isin(SECONDARY)].groupby('model')[['valid_blocks','dependency_missing','fit_failed','no_valid_candidate','eligible_blocks']].sum()
    left=np.zeros(len(a))
    for column,color in [('valid_blocks','tab:green'),('dependency_missing','gray'),('fit_failed','tab:red'),('no_valid_candidate','tab:orange')]:
        value=100*a[column]/a.eligible_blocks
        axes[1].barh(a.index,value,left=left,color=color,label=column);left+=value.to_numpy()
    axes[1].set(xlim=(0,105),xlabel='Percentage of in-scope B T16/24 blocks',title='Secondary status: dependency missing ≠ model failure');axes[1].legend(fontsize=7)
    save(fig,'availability')
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
    for T,marker in [(6,'o'),(24,'^')]:
        g=overall.loc[overall.domain.eq('real')&overall.arm.eq('B')&overall['T'].eq(T)]
        for model in PRIMARY:
            seconds=runtime.loc[runtime.kind.eq('online')&runtime.domain.eq('real')&runtime.arm.eq('B')&runtime['T'].eq(T)&runtime.model.eq(model),'seconds_per_case'].mean()
            mae=g.loc[g.model.eq(model),'MAE'].iloc[0]
            ax.scatter(seconds*1000,mae,color=colors[model],marker=marker,s=55,label=f'{model}, T{T}')
    ax.set(xscale='log',xlabel='Online composed/amortized milliseconds per history case',ylabel='MAE (weekly cases)',
           title='Real fixed-origin accuracy versus online cost\nIndex and generation costs reported separately; sMOA includes charged GAM cost')
    ax.legend(fontsize=7,ncol=2);save(fig,'error_runtime')


def write_report(out,scores,pairs,availability,runtime,verification):
    overall=scores.loc[scores.scope.eq('primary_common')&scores.statistic.eq('mean')&scores.dimension.eq('overall')].copy()
    notes=[]
    for (domain,arm,T),g in overall.groupby(['domain','arm','T']):
        best=g.loc[g.MAE.idxmin()];sm=g.loc[g.model.eq('sMOA')].iloc[0]
        notes.append(f'- {domain}/{arm}, T={T}: lowest observed MAE **{best.model} ({best.MAE:,.2f})**; sMOA **{sm.MAE:,.2f}**. Descriptive ranking, not a tuned model selection.')
    equivalent=pairs.loc[pairs.scope.eq('primary_common')&pairs.dimension.eq('overall')&pairs.practical_equivalence]
    changes=pairs.loc[pairs.scope.eq('history_24_minus_6')]
    timings=runtime.loc[runtime.kind.eq('embedding_and_index')]
    real_a=overall.loc[overall.domain.eq('real')&overall.arm.eq('A')].set_index(['T','model'])
    real_b=overall.loc[overall.domain.eq('real')&overall.arm.eq('B')].set_index(['T','model'])
    real_pairs=pairs.loc[pairs.scope.eq('primary_common')&pairs.dimension.eq('overall')&pairs.domain.eq('real')&pairs.baseline.eq('Persistence')].set_index(['arm','T'])
    normalized=verification['synthetic_B6_normalized_loss_diagnostic']
    report=f'''# Experiment 1 — Comparative Evaluation under Limited Historical Data

**Primary completed:** {verification['primary_predictions']:,} point forecasts, {verification['cases']:,} permitted-history cases. Persistence, MA(3), Drift, SES and sMOA; N=18.387 attempts/family, k=5, m=4.422, GAM, omit bridge. Three sMOA training banks, crossed with all 900 fresh synthetic test curves. Baselines run once per case; no ensemble or repeated baseline pseudo-replicates.

**Secondary status: {verification['secondary_status']}**. Missing statsmodels produces no Holt/ARIMA accuracy estimates. It is not evidence of model failure or poor accuracy. All out-of-scope and dependency statuses are in availability.csv/database.

## Setup and information availability

A uses first T=6,8,11 observations from reporting onset (first raw weekly count ≥10); origins and targets change with T. B holds origins/targets fixed and supplies last T=6,8,11,16,24 observations. GAM uses exactly permitted T values, then last six fitted levels for matching. Full-history forecasts from E2/E3 are not reused.

Synthetic: 900 selected weekly-count curves, balanced 100 per family/test seed, from 7,200 attempts using original generator and fresh namespace [20261104,310,test_seed,family,attempt]. Curves are independent RNG draws but share generator families with training: this is not an out-of-distribution pathogen test. Numeric duplicates/collisions are audited, not silently removed. Exact test–training content collisions: **{verification['test_training_hash_collisions']}**; duplicate selected-test hashes: **{verification['duplicate_test_hashes']}**.

Real A: 50 states ×3 T =150 history cases; onset metadata frozen at 2020-04-04, known before all evaluated issuance dates. Real B: 50 states ×37 common origins ×5 T =9,250 history cases. Saturday origin labels, Sunday cutoff, Monday issuance. All histories come from exact historical JHU vintages; final truth used only for scoring. This is a historical-vintage backtest, not prospective ForecastHub parity.

Every model receives input floored at 1 and predictions floored at 1. Scoring labels remain raw. No +100 offset, log transform, future-peak normalization, future tuning or upper clipping. sMOA retains the original valid-window criterion, including floating-point-retained constant histories. Training has an external synthetic prior and associated offline cost.

## Main accuracy tables

MAE/RMSE/bias/normalized MAE/MASE are computed for each bank; sMOA tables average seed-specific scores (mean RMSE is not pooled RMSE). Shared valid case mask requires every baseline and all three banks, then common units across T within each arm. Retained cases: **{verification['primary_common_cases']:,}/{verification['cases']:,}**. Baselines have one deterministic run. All per-bank/phase/level/family/horizon results are in scores.csv.

{markdown_table(overall,['domain','arm','T','model','n','MAE','RMSE','normalized_MAE','MASE'])}

![Early stage](figures/early_stage.png)
![Fixed origin](figures/fixed_origin_history.png)

## What these comparisons show

{chr(10).join(notes)}

These are observed rankings on common cohorts. A trends combine history growth with changing epidemic phases; only B holds the target event fixed. COVID is one disease and states share national waves, so row counts do not establish transfer to arbitrary emerging diseases.

### Interpretation of RQ1–RQ5

- **Very short history:** real A T6 has SES MAE {real_a.loc[(6,'SES'),'MAE']:,.2f} versus sMOA {real_a.loc[(6,'sMOA'),'MAE']:,.2f}. sMOA is not uniformly preferable at the minimum supported history. Relative to Persistence, its T6 paired delta is {real_pairs.loc[('A',6),'delta_MAE']:+,.2f}, CI [{real_pairs.loc[('A',6),'lo']:+,.2f}, {real_pairs.loc[('A',6),'hi']:+,.2f}], so the observed disadvantage is uncertain on these 50 states.
- **Raw and normalized metrics can disagree:** real A T8 raw MAE favors sMOA ({real_a.loc[(8,'sMOA'),'MAE']:,.2f} versus SES {real_a.loc[(8,'SES'),'MAE']:,.2f}), but normalized MAE favors SES ({real_a.loc[(8,'SES'),'normalized_MAE']:.4f} versus sMOA {real_a.loc[(8,'sMOA'),'normalized_MAE']:.4f}). Raw errors emphasize high-count events; per-origin normalization changes weights. T8 raw paired CI versus Persistence still crosses zero. Do not present that numeric ranking as a universal improvement.
- **More stable evidence at real A T11:** sMOA−Persistence MAE is {real_pairs.loc[('A',11),'delta_MAE']:+,.2f}, CI [{real_pairs.loc[('A',11),'lo']:+,.2f}, {real_pairs.loc[('A',11),'hi']:+,.2f}]. This is conditional evidence on the declared cohort/banks, not a universal claim about epidemics.
- **Fixed-origin availability:** real B sMOA MAE changes from {real_b.loc[(6,'sMOA'),'MAE']:,.2f} at T6 to {real_b.loc[(24,'sMOA'),'MAE']:,.2f} at T24; the best observed raw score is at T11, not the longest history. Normalized MAE goes from {real_b.loc[(6,'sMOA'),'normalized_MAE']:.4f} to {real_b.loc[(24,'sMOA'),'normalized_MAE']:.4f}, so extra history does not improve every metric or imply a monotone gain. B comparisons are on exactly the same target events.
- **Complexity:** simple methods remain much cheaper; no overall comparison establishes practical equivalence under the declared interval criterion. Failure to certify equivalence is not proof of non-equivalence. Read the raw/normalized benefits and measured preparation/online cost together rather than calling complexity automatically justified.
- **Conditions:** scores.csv contains phase/level and horizon strata. In real B T6, rising-phase sMOA is only slightly better than Persistence in raw MAE, while declining/stable strata show larger differences. These are descriptive subgroup results; all five models become less accurate as horizon grows in the displayed B panels.

### Heavy tails in synthetic normalized errors

For synthetic B T6 sMOA, mean target-level normalized absolute loss (averaged over bank losses) is **{normalized['mean']:.4f}**, median **{normalized['median']:.4f}**, p95 **{normalized['p95']:.4f}**, p99 **{normalized['p99']:.4f}**. The largest 1% of losses contributes **{normalized['top_one_percent_share']:.2%}** of the total normalized loss. **{normalized['unit_scale_cases']} of 900** histories have last-four scale exactly 1. Small observed scales followed by large generated spikes make mean normalized error highly tail-sensitive. This is not fixed by omitting those cases or changing scaling after evaluation; the complete original cohort and primary metrics are retained. The diagnostic explains why synthetic normalized means look nearly identical across methods even when raw MAE differs.

### Paired history change, T24 minus T6 in B

{markdown_table(changes,['domain','baseline','n','delta_MAE','lo','hi'])}

Persistence and MA(3) must be unchanged in B: their permitted final values are identical. SES, Drift and GAM-based sMOA can change. B query vectors differing from T6 beyond allclose rtol/atol=1e-10: **{verification['B_queries_different_from_T6']:.2%}** among longer-history cases. This checks that k=5 does not make sMOA history-invariant; the fraction alone does not measure the magnitude of a query change.

![Horizon sensitivity](figures/horizon_errors.png)
![Paired skill](figures/paired_skill.png)

### Practical equivalence

Paired losses average the three sMOA bank losses at each target, not bank predictions. CI uses 1,000 whole-curve/state resamples, preserving horizons and pairing. Synthetic/real and A/B are not pooled into independent evidence. A/B share curves/states; bootstrap does not capture all national-wave or generator-family uncertainty. No multiple-comparison correction; strata are exploratory.

Descriptive equivalence requires the entire 95% interval of relative delta MAE inside ±5%, a declared E1 margin, not formal TOST. CI crossing zero alone does not establish equivalence. Overall T/domain/arm comparisons meeting the criterion: **{len(equivalent)}**.

{markdown_table(equivalent,['domain','arm','T','baseline','relative_delta','relative_lo','relative_hi']) if len(equivalent) else 'No overall comparison met the declared equivalence interval criterion.'}

## Missing fits and MASE

{markdown_table(availability.groupby('model',as_index=False)[['eligible_blocks','valid_blocks','dependency_missing','fit_failed','no_valid_candidate']].sum(),['model','eligible_blocks','valid_blocks','dependency_missing','fit_failed','no_valid_candidate'])}

MASE uses mean abs differences of permitted floored input. Zero denominator is undefined, not epsilon-adjusted; those cases remain in MAE/RMSE. MASE denominators change with T, so cross-T effects should use common-origin MAE/RMSE or normalized MAE (same last-four scale in B). Undefined MASE target instances per model or bank: **{verification['MASE_undefined_targets']}**. Negative raw scoring targets per model/bank: **{verification['negative_scoring_targets']}**.

![Availability](figures/availability.png)

## Complexity and runtime

Online sMOA time = shared measured per-case GAM cost charged to each bank + measured batch search/aggregation. It is composed/amortized time, not individual request latency. GAM physically computed once for all banks; do not sum charged bank times to estimate actual process wall time. Baseline timings include fit and prediction; missing dependencies have no model runtime estimates.

{markdown_table(timings,['bank_seed','seconds','peak_rss_bytes','timing'])}

Embedding/index preparation is separate. Seed0 uses the verified existing cache; seeds1/2 freshly embed existing full packed banks. No synthetic training generation occurs in this run; historical generation/export or replay timings in runtime.csv are not zero-cost assumptions and not newly measured generation. Full input verification/hash I/O and test generation are outside online model cost. Software/hardware/thread limits are in protocol; index workers=1. Microsecond baseline differences are dominated by runtime/clock overhead and should not establish a reliable fine ranking.

![Error versus runtime](figures/error_runtime.png)

## Reproducibility

Four source files only: runner, tests, executed notebook, README. results.sqlite stores all fresh-test attempts, selected curves, permitted raw histories/hashes, GAM queries, actuals, fit statuses, forecasts and timing metadata. Protocol freezes design, input hashes and prediction-source identity before test selection/scoring. Completed forecast blocks are transactional/resumable; plots and summary tables reuse them. Scratch embeddings are removed after each worker. No test score feeds BO, family selection or baseline changes.
'''
    (out/'report.md').write_text(report)


def analyze(out):
    db=connect(out)
    cases=pd.read_sql_query('SELECT * FROM cases',db)
    fits=pd.read_sql_query('SELECT * FROM fits',db)
    if len(cases)!=16600:raise ValueError('Incomplete E1 cohort')
    if db.execute('SELECT COUNT(*) FROM test_audit').fetchone()[0]!=7200:raise ValueError('Incomplete test attempt audit')
    expected=[(m,-1) for m in BASELINES]+[('sMOA',s) for s in SEEDS]
    if len(fits.loc[fits.model.isin(PRIMARY)])!=len(cases)*7:raise ValueError('Missing primary attempts')
    keep=common_cases(cases,fits,expected)
    data=pd.read_sql_query('''SELECT p.*,c.domain,c.arm,c.series,c.cluster,c.family,c.test_seed,c.origin,c.T,c.phase,c.case_level,
       c.normalization,c.mase_den,o.actual FROM predictions p JOIN cases c USING(case_id)
       JOIN observations o USING(case_id,horizon)''',db)
    primary=data.loc[data.model.isin(PRIMARY)&data.case_id.isin(keep)].copy()
    if primary.groupby(['case_id','horizon']).size().ne(7).any():raise ValueError('Incomplete primary pairing')
    scores=summarize_scores(primary);pairs=paired_tables(primary)
    for model in SECONDARY:
        subset=cases.loc[cases.arm.eq('B')&cases['T'].isin([16,24])]
        ids=common_cases(subset,fits,[(model,-1)]+[('sMOA',s) for s in SEEDS])
        if ids:
            extra=data.loc[data.case_id.isin(ids)&data.model.isin([model,'sMOA'])]
            scope='secondary_pairwise_'+model
            scores=pd.concat([scores,summarize_scores(extra,scope)],ignore_index=True)
            pairs=pd.concat([pairs,paired_tables(extra,scope)],ignore_index=True)
    secondary_ids=common_cases(cases.loc[cases.arm.eq('B')&cases['T'].isin([16,24])],fits,expected+[(m,-1) for m in SECONDARY])
    if secondary_ids:
        extra=data.loc[data.case_id.isin(secondary_ids)]
        scores=pd.concat([scores,summarize_scores(extra,'all_secondary_common')],ignore_index=True)
        pairs=pd.concat([pairs,paired_tables(extra,'all_secondary_common')],ignore_index=True)
    pairs=pd.concat([pairs,history_changes(primary)],ignore_index=True)
    availability=availability_table(cases,fits);runtime=runtime_table(db,cases,fits)
    for name,frame in [('scores',scores),('paired_comparisons',pairs),('availability',availability),('runtime',runtime)]:frame.to_csv(out/f'{name}.csv',index=False)
    primary_count=int(data.model.isin(PRIMARY).sum())
    expected_count=int(fits.loc[fits.model.isin(PRIMARY)&fits.status.eq('ok')].shape[0]*4)
    if primary_count!=expected_count:raise ValueError('Prediction/fit count disagreement')
    # B observations and deterministic last-value/MA3 predictions must be identical across T.
    b=data.loc[data.arm.eq('B')]
    if b.groupby(['domain','series','origin','horizon']).actual.nunique().gt(1).any():raise ValueError('B targets changed with T')
    for model in ['Persistence','MA3']:
        if b.loc[b.model.eq(model)].groupby(['domain','series','origin','horizon']).prediction.nunique().gt(1).any():raise ValueError('History-invariant baseline changed in B')
    longer=cases.loc[cases.arm.eq('B')&cases['T'].gt(6)].merge(cases.loc[cases.arm.eq('B')&cases['T'].eq(6),['domain','series','origin','query']],
                 on=['domain','series','origin'],validate='many_to_one',suffixes=('','_6'))
    different=float(np.mean([not np.allclose(unpack(a),unpack(b),rtol=1e-10,atol=1e-10) for a,b in zip(longer['query'],longer['query_6'])]))
    obs=pd.read_sql_query('SELECT o.*,c.mase_den FROM observations o JOIN cases c USING(case_id)',db)
    collision=db.execute('SELECT COUNT(*) FROM test_curves t WHERE EXISTS (SELECT 1 FROM training_hashes r WHERE r.content_sha256=t.content_sha256)').fetchone()[0]
    duplicate=db.execute('SELECT COUNT(*)-COUNT(DISTINCT content_sha256) FROM test_curves').fetchone()[0]
    diagnostic=primary.loc[primary.domain.eq('synthetic')&primary.arm.eq('B')&primary['T'].eq(6)&primary.model.eq('sMOA')].copy()
    diagnostic['loss']=(diagnostic.prediction-diagnostic.actual).abs()/diagnostic.normalization
    loss=diagnostic.groupby(['case_id','horizon']).loss.mean().to_numpy()
    tail=loss[np.argsort(loss)[-int(np.ceil(.01*len(loss))):]]
    normalized_diagnostic=dict(mean=float(loss.mean()),median=float(np.median(loss)),p95=float(np.quantile(loss,.95)),
        p99=float(np.quantile(loss,.99)),top_one_percent_share=float(tail.sum()/loss.sum()),
        unit_scale_cases=int(diagnostic.drop_duplicates('case_id').normalization.eq(1).sum()))
    verification=dict(cases=len(cases),primary_predictions=primary_count,primary_common_cases=len(keep),test_curves=900,test_attempts=7200,
        secondary_status='dependency_missing' if availability.dependency_missing.sum() else 'evaluated',
        test_training_hash_collisions=int(collision),duplicate_test_hashes=int(duplicate),
        MASE_undefined_targets=int(obs.mase_den.eq(0).sum()),negative_scoring_targets=int(obs.actual.lt(0).sum()),
        B_queries_different_from_T6=different,fit_blocks=len(fits),primary_failures=int(fits.loc[fits.model.isin(PRIMARY),'status'].ne('ok').sum()),
        fixed_B_targets_verified=True,fixed_persistence_MA3_B_verified=True,
        synthetic_B6_normalized_loss_diagnostic=normalized_diagnostic)
    plots(out,scores,pairs,availability,runtime);write_report(out,scores,pairs,availability,runtime,verification)
    db.close();return verification


def run(args):
    start=time.perf_counter();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=acquire_run_lock(out/'run.lock')
    prior=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    inputs=validate_inputs(args);signature=engine_signature();path=out/'protocol.json'
    if path.exists():
        p=json.loads(path.read_text())
        if not args.resume or p['engine_signature']!=signature or p['inputs_sha256']!=inputs:raise ValueError('Resume requires identical design, prediction code and inputs')
    else:
        atomic_json(path,dict(rules=RULES,source=Path(__file__).read_text(),engine_signature=signature,inputs_sha256=inputs,
            created_utc=datetime.now(timezone.utc).isoformat(),hardware=dict(platform=platform.platform(),processor=platform.processor(),cpus=os.cpu_count()),
            packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib']},
            statsmodels_available=importlib.util.find_spec('statsmodels') is not None))
    atomic_json(out/'progress.json',dict(status='running',phase='preparing_test'))
    db=connect(out);initialize(db)
    generate_test(db);training_audit(db,args.e2_dir)
    prepare_real(args.panel_dir,db);prepare_synthetic(db)
    run_baselines(db,out);db.close()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for seed in SEEDS:
        atomic_json(out/'progress.json',dict(status='running',phase='smoa',bank_seed=seed))
        subprocess.run([sys.executable,str(Path(__file__)),'--worker','--bank-seed',str(seed),
                        '--output-dir',str(out),'--e2-dir',str(args.e2_dir.resolve())],env=env,check=True)
    atomic_json(out/'progress.json',dict(status='running',phase='analysis'))
    verification=analyze(out)
    if any(file_hash(Path(p))!=digest for p,digest in inputs.items()):raise RuntimeError('Input changed during experiment')
    atomic_json(out/'progress.json',dict(status='complete',primary_complete=True,**verification,
        wall_seconds=time.perf_counter()-start,initial_full_run_wall_seconds=prior.get('initial_full_run_wall_seconds',time.perf_counter()-start),
        inputs_unchanged=True,analysis_source_sha256=file_hash(Path(__file__)),completed_utc=datetime.now(timezone.utc).isoformat(),
        artifacts_sha256={str(p.relative_to(out)):file_hash(p) for p in [out/'results.sqlite',out/'report.md',*out.glob('*.csv'),*out.joinpath('figures').glob('*')]}))
    print('Complete:',out/'report.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/experiment_1')
    parser.add_argument('--e2-dir',type=Path,default=ROOT/'outputs/experiment_2')
    parser.add_argument('--panel-dir',type=Path,default=ROOT/'outputs/epidemic_panel')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--bank-seed',type=int,choices=SEEDS)
    parser.add_argument('--execute-notebook',action='store_true')
    args=parser.parse_args()
    if args.execute_notebook:
        from notebooks import execute_notebook
        execute_notebook(Path(__file__).with_suffix('.ipynb'), working_dir=ROOT, timeout=600)
    elif args.worker:smoa_worker(args)
    else:run(args)
