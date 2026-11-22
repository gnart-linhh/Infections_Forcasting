"""Size-controlled reference composition study; no target outcomes in selection."""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from experiments.robust.data import (resolve, array_hash, load_reference, load_real_cases,
    load_real_outcomes, attach_outcomes, ForecastCase, ROOT)
from experiments.robust.method import RobustSMOA
from experiments.robust.pipeline import forecast_cases
from experiments.robust.tuning import generate_independent, synthetic_cases, eligible
from experiments.robust.evaluation import point_metrics, interval_metrics
from experiments.runtime import atomic_json, file_hash, acquire_run_lock
from bank import load_packed, save_packed, FAMILIES
from generator import change_point_sir


def balanced_rows(labels, budget, seed):
    labels = np.asarray(labels)
    if not 0 < budget <= len(labels):
        raise ValueError('Window budget exceeds available rows')
    groups, counts = np.unique(labels, return_counts=True)
    quotas = np.minimum(counts, budget//len(groups))
    remaining = budget-int(quotas.sum())
    while remaining:
        available = np.flatnonzero(quotas < counts)
        increment = min(max(remaining//len(available), 1), int(np.min(counts[available]-quotas[available])))
        for i in available:
            take = min(increment, remaining)
            quotas[i] += take;remaining -= take
            if not remaining:break
    rng = np.random.default_rng(seed)
    rows = np.concatenate([rng.choice(np.flatnonzero(labels == group), int(n), replace=False)
                           for group, n in zip(groups, quotas)])
    return np.sort(rows)


def dynamics_bins(x):
    """Twelve fixed historical-difference strata: trend, volatility, turning."""
    x = np.asarray(x)
    magnitude = np.mean(np.abs(x), axis=1)
    trend = np.sum(x, axis=1)/(np.sum(np.abs(x), axis=1)+1e-12)
    volatility = np.std(x, axis=1)/(magnitude+1e-12) > .8
    turning = np.any(x[:, 1:]*x[:, :-1] < 0, axis=1)
    return (np.where(trend > .25, 2, np.where(trend < -.25, 0, 1))*4+
            volatility.astype(int)*2+turning.astype(int)).astype('u1')


def audit_reference(engine, config, out):
    folder = resolve(config['reference']['embedding'])
    lookup = json.loads((folder/'lookup.json').read_text())
    for name in ['curve_codes', 'steps']:
        if file_hash(folder/(name+'.dat')) != lookup[name+'_sha256']:
            raise ValueError('Modified reference row-to-curve mapping')
    codes = np.memmap(folder/'curve_codes.dat', mode='r', dtype='<u4', shape=(len(engine.x),))
    steps = np.memmap(folder/'steps.dat', mode='r', dtype='<u4', shape=(len(engine.x),))
    curves = load_packed(resolve(config['reference']['packed'])/'curves.npz')
    if len(curves) != len(lookup['names']):raise ValueError('Curve mapping count mismatch')
    curve_families = np.array([FAMILIES.index(c.curve_type) for c in curves], dtype='u1')
    family = curve_families[codes]
    windows_per_curve = np.bincount(codes, minlength=len(curves))
    records = []
    for i, curve in enumerate(curves):
        possible = max(len(curve.ts)-10, 0)  # k+1 history, bridge, four future levels
        records.append(dict(curve_id=lookup['names'][i], family=curve.curve_type, cadence=curve.time_cadence,
                            units=curve.scale, length=len(curve.ts), windows=int(windows_per_curve[i]),
                            filtered_windows=possible-int(windows_per_curve[i]),
                            sha256=array_hash(curve.ts), peaks=int(np.sum((np.diff(curve.ts)[:-1]>0)&(np.diff(curve.ts)[1:]<=0)))))
    frame = pd.DataFrame(records)
    frame.to_csv(out/'curve_diagnostics.csv.gz', index=False)
    frame.groupby(['family','units','cadence'], as_index=False).agg(curves=('curve_id','size'), windows=('windows','sum'),
         mean_length=('length','mean'), median_length=('length','median'), filtered_windows=('filtered_windows','sum')).to_csv(out/'library_diagnostics.csv',index=False)
    bins = np.empty(len(engine.x), dtype='u1')
    for start in range(0, len(bins), 200000):
        bins[start:start+200000] = dynamics_bins(engine.x[start:start+200000])
    sample = np.random.default_rng(config['seed']).choice(len(bins), min(100000,len(bins)),replace=False)
    x = engine.x[sample]; y = np.cumsum(engine.y[sample],axis=1)
    levels = np.array([curves[int(codes[row])].ts[int(steps[row])-1:int(steps[row])+5].mean() for row in sample])
    sampled = pd.DataFrame(dict(family=family[sample], stratum=bins[sample], level=levels,
        relative_change=x.sum(axis=1)/np.maximum(levels,1), slope=x.mean(axis=1), volatility=x.std(axis=1),
        sudden_change=np.max(np.abs(x),axis=1)/np.maximum(levels,1), future_h4=y[:,-1]))
    sampled.to_csv(out/'sampled_dynamics.csv.gz', index=False)
    normalized = x/np.maximum(np.abs(x).sum(axis=1,keepdims=True),1e-12)
    atomic_json(out/'audit.json',dict(curves=len(curves), windows=len(bins),
        family_windows={FAMILIES[i]:int((family==i).sum()) for i in range(3)},
        family_curves=frame.groupby('family').size().to_dict(),
        sampled_windows=len(sample), sample_exact_duplicate_fraction=1-len(np.unique(x,axis=0))/len(x),
        sample_rounded_shape_duplicate_fraction=1-len(np.unique(np.round(normalized,2),axis=0))/len(x),
        duplicate_scope='100000 seeded windows, not exhaustive; rounded L1-normalized difference shapes are a near-duplicate proxy',
        source_rejections='Accepted-curve artifact only; generation rejection counts remain in original generation metadata',
        library_metadata=json.loads((folder/'metadata.json').read_text())))
    return family, bins, codes, lookup, curves


def embed_new(curves):
    xs, ys, codes = [], [], []
    from model import valid_history_windows
    for i,c in enumerate(curves):
        windows = np.lib.stride_tricks.sliding_window_view(np.maximum(c.ts,1e-8),11)
        valid = valid_history_windows(windows[:,:6])
        diff = np.diff(windows[valid],axis=1)
        xs.append(diff[:,:5]);ys.append(diff[:,6:]);codes.extend([i]*len(diff))
    return np.vstack(xs).astype('<f8'),np.vstack(ys).astype('<f8'),np.array(codes,dtype='<u4')


def make_libraries(original, family, bins, config, out, forbidden_hashes, codes):
    cfg = config['library_study'];budget=cfg['window_budget'];seed=config['seed']
    if budget < original.closest:raise ValueError('Budget must retain m=4422')
    selections = dict(L0S=np.sort(np.random.default_rng(seed).choice(len(family),budget,replace=False)),
                      L1=balanced_rows(family,budget,seed), L2=balanced_rows(bins,budget,seed))
    # Phase B assets exist before adding the single new family.
    for name,rows in selections.items():np.save(out/(name+'_rows.npy'), rows)
    cp_curves=[];cp_records=[];used=set(forbidden_hashes)
    for i in range(cfg['change_point_curves']*10):
        key=[seed, 3691107317, 0, i]
        curve,states,parameters=change_point_sir(np.random.default_rng(np.random.SeedSequence(key)),return_state=True)
        digest=array_hash(curve.ts)
        if digest in used or eligible(curve) is None:continue
        used.add(digest);cp_curves.append(curve)
        cp_records.append(dict(curve_id=f'cp_reference_{i}',family='change_point_sir',seed_key=key,sha256=digest,parameters=parameters))
        if len(cp_curves)==cfg['change_point_curves']:break
    if len(cp_curves)!=cfg['change_point_curves']:raise RuntimeError('Insufficient eligible change-point references')
    save_packed(cp_curves,out/'change_point_reference.npz')
    cx,cy,cc=embed_new(cp_curves)
    np.savez_compressed(out/'change_point_embedding.npz',x=cx,y=cy,curve_codes=cc)
    atomic_json(out/'change_point_curves.json',cp_records)
    for fraction in cfg['augmentation_fractions']:
        n=int(round(budget*fraction))
        if n>len(cx):raise ValueError('Insufficient change-point windows for fixed augmentation budget')
        name=f'L3_{int(fraction*100)}'
        old=balanced_rows(family,budget-n,seed)
        new=np.sort(np.random.default_rng(seed).choice(len(cx),n,replace=False))
        np.save(out/(name+'_rows.npy'),old);np.save(out/(name+'_new_rows.npy'),new)
        selections[name]=(old,new)
    manifests={}
    for name,rows in selections.items():
        old,new=rows if isinstance(rows,tuple) else (rows,np.array([],int))
        manifests[name]=dict(windows=len(old)+len(new), original_windows=len(old),new_windows=len(new),seed=seed,
            original_curve_count=int(len(np.unique(codes[old]))),new_curve_count=int(len(np.unique(cc[new]))) if len(new) else 0,
            original_rows_sha256=file_hash(out/(name+'_rows.npy')),
            new_rows_sha256=file_hash(out/(name+'_new_rows.npy')) if len(new) else None,
            policy='uniform original' if name=='L0S' else 'balanced families' if name=='L1' else 'balanced fixed dynamics strata' if name=='L2' else 'balanced original families plus change-point SIR',
            family_windows={**{FAMILIES[i]:int((family[old]==i).sum()) for i in range(3)},'change_point_sir':len(new)},
            generator_sha256=file_hash(ROOT/'experiments/synthetic/generator.py'))
    atomic_json(out/'library_manifests.json',manifests)
    return selections,cx,cy,cc,cp_curves,manifests


def instantiate(original, selection, cx, cy):
    if selection is None:return original
    old,new=selection if isinstance(selection,tuple) else (selection,np.array([],int))
    x=original.x[old];y=original.y[old]
    if len(new):x=np.vstack([x,cx[new]]);y=np.vstack([y,cy[new]])
    return RobustSMOA(x,y,original.closest,original.smoothing)


def diagnostic_matches(engine,cases, row_families, name):
    rows=[]
    for case in cases:
        match=engine.match(case.history)
        counts=np.bincount(row_families[match.indices],minlength=4);p=counts/counts.sum()
        rows.append(dict(library=name,dataset=case.dataset,case_id=case.case_id,history_length=len(case.history),
            minimum_distance=float(match.distances.min()),median_distance=float(np.median(match.distances)),
            family_entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))),
            future_iqr_h4=float(np.ptp(np.quantile(np.cumsum(match.future,axis=1)[:,-1],[.25,.75]))),
            family_fractions=json.dumps(p.tolist())))
    return rows


def score(table, original):
    truth=table.actual.to_numpy();p=table.prediction.to_numpy();e=p-truth
    return dict(n=len(table),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),
        nmae=float(np.mean(np.abs(e)/np.maximum(table.scale,1))),
        relative_mae_improvement=1-float(np.abs(e).mean())/original if original else None,
        p95_absolute_error=float(np.quantile(np.abs(e),.95)),p99_absolute_error=float(np.quantile(np.abs(e),.99)),
        severe_underprediction=float(np.mean((truth>0)&(p<.5*truth))),severe_overprediction=float(np.mean(p>2*np.maximum(truth,1))),
        clusters=table.region.nunique())


def evaluate_tables(predictions,out,config):
    for suffix,groups in [('', ['dataset']),('_by_history',['dataset','history_length']),('_by_horizon',['dataset','horizon']),
                          ('_by_phase',['dataset','phase']),('_by_region',['dataset','region'])]:
        rows=[]
        for key,block in predictions.groupby(groups):
            key=key if isinstance(key,tuple) else (key,)
            base=block.query('library=="L0" and model=="M0"')
            base_loss=float(np.abs(base.prediction-base.actual).mean())
            for (lib,model),g in block.groupby(['library','model']):
                rows.append(dict(zip(groups,key),library=lib,model=model,**score(g,base_loss)))
        pd.DataFrame(rows).to_csv(out/f'metrics{suffix}.csv',index=False)
    pd.read_csv(out/'metrics.csv').to_csv(out/'library_ablation.csv',index=False)
    uq=[];paired=[]
    for dataset,block in predictions.groupby('dataset'):
        base=block.query('library=="L0" and model=="M0"').set_index(['case_id','horizon'])
        for (lib,model),g in block.groupby(['library','model']):
            g=g.set_index(['case_id','horizon']).reindex(base.index)
            if g.prediction.isna().any() or not np.array_equal(g.actual,base.actual):raise ValueError('Unpaired library comparison')
            delta=np.abs(g.prediction-g.actual)-np.abs(base.prediction-base.actual)
            sums=delta.groupby(g.region).sum();counts=delta.groupby(g.region).size()
            rng=np.random.default_rng(config['seed']);indices=rng.integers(len(sums),size=(config['evaluation']['bootstrap_draws'],len(sums)))
            bootstrap=sums.to_numpy()[indices].sum(axis=1)/counts.to_numpy()[indices].sum(axis=1)
            paired.append(dict(dataset=dataset,library=lib,model=model,n=len(g),paired_mae_difference=float(delta.mean()),lower=float(np.quantile(bootstrap,.025)),upper=float(np.quantile(bootstrap,.975))))
            from experiments.robust.pipeline import QCOLS
            uq.append(dict(dataset=dataset,library=lib,model=model,**interval_metrics(g.actual,g[QCOLS])))
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(uq).to_csv(out/'uncertainty_metrics.csv',index=False)


def run_library_study(config,smoke=False):
    start=time.perf_counter();root=resolve(config['output_root'])/'library_study'
    original,hashes,provenance=load_reference(config)
    synthetic,synthetic_folder=generate_independent(config,hashes,provenance,resolve(config['output_root']),smoke)
    identity=dict(config=config,smoke=smoke,provenance=provenance,source=file_hash(Path(__file__)),
        implementation_sha256={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},external_sha256=file_hash(resolve(config['evaluation']['database'])))
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=root/(('smoke_' if smoke else 'full_')+run_id);out.mkdir(parents=True,exist_ok=True)
    with acquire_run_lock(out/'run.lock'):
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status')=='complete':return out
        atomic_json(out/'run_config.json',identity)
        phase_start=time.perf_counter();timing={}
        family,bins,codes,lookup,curves=audit_reference(original,config,out)
        timing['audit_seconds']=time.perf_counter()-phase_start
        phase_start=time.perf_counter()
        forbidden=hashes|{r['sha256'] for role in ['validation','background','check'] for r in synthetic[role+'_records']}
        selections,cx,cy,cc,cp_curves,manifests=make_libraries(original,family,bins,config,out,forbidden,codes)
        timing['augmentation_generation_and_embedding_seconds']=time.perf_counter()-phase_start
        forbidden.update(array_hash(c.ts) for c in cp_curves)
        # Held-out mechanism/parameter stress curves have their own seed role.
        stress=[];records=[]
        for i in range(config['library_study']['stress_curves']):
            key=[config['seed'],3691107317,1,i]
            curve=change_point_sir(np.random.default_rng(np.random.SeedSequence(key)),reproduction=4.,recovery=.4)
            digest=array_hash(curve.ts)
            if digest in forbidden or digest in {r['sha256'] for r in records}:raise ValueError('Stress curve overlap')
            onset=eligible(curve)
            if onset is None:continue
            stress.append(curve);records.append(dict(curve_id=f'cp_stress_{i}',onset=onset,sha256=digest,seed_key=key))
        synthetic['stress']=stress;synthetic['stress_records']=records
        save_packed(stress,out/'stress_curves.npz');atomic_json(out/'stress_provenance.json',records)
        val_cases,val_truth=synthetic_cases(synthetic,'validation',config['history_lengths'])
        validation=[];validation_frames=[];diagnostics=[];phase_start=time.perf_counter()
        libraries={'L0':None,**selections}
        for name,selection in libraries.items():
            engine=instantiate(original,selection,cx,cy)
            issued,_=forecast_cases(engine,val_cases,dict(temperature=4.,beta=0.),config['seed'],config['uncertainty']['draws'])
            scored=attach_outcomes(issued.query('model in ["M0","M1"]'),val_truth)
            validation_frames.append(scored.assign(library=name))
            for model,g in scored.groupby('model'):validation.append(dict(library=name,model=model,**score(g,1)))
            localfamilies=family if selection is None else np.r_[family[selection[0]],np.full(len(selection[1]),3)] if isinstance(selection,tuple) else family[selection]
            diagnostics.extend(diagnostic_matches(engine,val_cases,localfamilies,name))
            print('Validated',name,flush=True)
            if engine is not original:del engine
        timing['validation_matching_and_inference_seconds']=time.perf_counter()-phase_start
        validation=pd.DataFrame(validation)
        vf=pd.concat(validation_frames,ignore_index=True)
        vf['family']=vf.region.map({r['curve_id']:r['family'] for r in synthetic['validation_records']})
        vf['seed']=vf.region.map({r['curve_id']:r['seed_key'][3] for r in synthetic['validation_records']})
        vf['count_scale']=pd.cut(vf.scale,[0,1000,100000,np.inf],labels=['low','medium','high'])
        vf.to_csv(out/'validation_predictions.csv.gz',index=False)
        strata=[]
        for dimension in ['history_length','horizon','phase','family','seed','count_scale']:
            for (name,model,stratum),g in vf.groupby(['library','model',dimension],observed=True):
                strata.append(dict(library=name,model=model,dimension=dimension,stratum=str(stratum),**score(g,None)))
        pd.DataFrame(strata).to_csv(out/'validation_by_stratum.csv',index=False)
        base_val=validation.query('library=="L0"').set_index('model').mae
        validation['relative_mae_improvement']=1-validation.mae/validation.model.map(base_val)
        validation.to_csv(out/'library_validation.csv',index=False)
        candidates=validation.query('model=="M0"')
        winner=candidates.sort_values(['mae','nmae','library']).iloc[0].library
        frozen=dict(library=winner,temperature=4.,beta=0.,selection='synthetic validation unweighted pooled MAE; NMAE reported',
                    manifest=manifests.get(winner,dict(windows=len(original.x),policy='unchanged full reference')),reference=provenance)
        atomic_json(out/'selected_library.json',frozen) # before real histories/outcomes
        pd.DataFrame(diagnostics).to_csv(out/'matching_diagnostics.csv',index=False)
        real=load_real_cases(config,smoke);check,ct=synthetic_cases(synthetic,'check',config['history_lengths']);st,stt=synthetic_cases(synthetic,'stress',config['history_lengths'])
        cases=real+check+st;all_predictions=[];selected_engine=None;selected_diagnostics=None
        phase_start=time.perf_counter()
        for name,selection in libraries.items():
            engine=instantiate(original,selection,cx,cy)
            issued,diag=forecast_cases(engine,cases,dict(temperature=4.,beta=0.),config['seed'],config['uncertainty']['draws'])
            issued=issued.query('model in ["M0","M1"]').assign(library=name)
            issued.to_csv(out/f'issued_{name}.csv.gz',index=False)
            all_predictions.append(issued)
            localfamilies=family if selection is None else np.r_[family[selection[0]],np.full(len(selection[1]),3)] if isinstance(selection,tuple) else family[selection]
            diagnostics.extend(diagnostic_matches(engine,cases,localfamilies,name))
            if name=='L0':original_diag={d['case_id']:d for d in diag}
            if name==winner:selected_engine=engine;selected_diagnostics=diag
            elif engine is not original:del engine
            print('Evaluated',name,flush=True)
        timing['evaluation_matching_and_inference_seconds']=time.perf_counter()-phase_start
        truth=pd.concat([load_real_outcomes(config,real),ct,stt],ignore_index=True)
        predictions=attach_outcomes(pd.concat(all_predictions,ignore_index=True),truth)
        predictions.to_csv(out/'predictions.csv.gz',index=False)
        evaluate_tables(predictions,out,config)
        pd.DataFrame(diagnostics).to_csv(out/'matching_diagnostics.csv',index=False)
        # Existing direct SHAP machinery explains chosen-library forecast versus L0.
        from experiments.robust.explain import explain_experiment
        parts=[]
        for lib,src,dst in [('L0','M0','M0'),('L0','M1','M1'),(winner,'M0','M2'),(winner,'M1','M3')]:
            parts.append(predictions.loc[predictions.library.eq(lib)&predictions.model.eq(src)].assign(model=dst))
        xai=out/'xai';xai.mkdir(exist_ok=True);xp=pd.concat(parts,ignore_index=True);xp.to_csv(xai/'predictions.csv.gz',index=False)
        for d in selected_diagnostics:d['original_analogue_paths']=original_diag[d['case_id']]['analogue_paths']
        atomic_json(xai/'matching_diagnostics.json',selected_diagnostics)
        explain_experiment(selected_engine,real,xp,dict(temperature=4.,beta=0.),synthetic,config,xai,reference_engine=original)
        atomic_json(out/'status.json',dict(status='complete',selected_library=winner,smoke=smoke,real_cases=len(real),
            seconds=time.perf_counter()-start,timing=timing,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            window_budget=config['library_study']['window_budget'],storage_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())))
    return out
