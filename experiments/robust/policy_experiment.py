"""Synthetic-only rank-policy preparation, selection, evaluation and XAI."""
from __future__ import annotations
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd

from experiments.runtime import atomic_json,file_hash,acquire_run_lock
from experiments.robust.data import (ROOT,resolve,array_hash,load_reference,load_real_cases,
                                     load_real_outcomes,attach_outcomes)
from experiments.robust.policy import FEATURES,RankPolicy,candidate_features,convex_forecast
from experiments.robust.tuning import generate_independent,synthetic_cases,eligible
from experiments.robust.pipeline import forecast_cases,save_tables,QCOLS
from experiments.robust.evaluation import point_metrics
from generator import change_point_sir
from bank import save_packed


def prepare_records(engine,cases,config):
    rows=[];diagnostics=[]
    for case in cases:
        base,paths=forecast_cases(engine,[case],dict(temperature=4.,beta=0.),config['seed'],config['uncertainty']['draws'])
        x,candidates,ranked,order=candidate_features(engine,case.history)
        a=base.query('model=="M0"').sort_values('horizon').reset_index(drop=True).copy()
        a['weighted']=base.query('model=="M1"').sort_values('horizon').prediction.to_numpy()
        for j,name in enumerate(FEATURES):a[name]=x[:,j]
        a['horizon']=a.horizon.astype(int);a['history_length']=a.history_length.astype(int)
        for j in range(3):
            a[f'candidate_{j}']=candidates[:,j];a[f'rank_{j}']=ranked[:,j];a[f'identity_{j}']=order[:,j]
        rows.append(a);diagnostics.extend(paths)
    return pd.concat(rows,ignore_index=True),diagnostics


def scores(frame,point):
    actual=frame.actual.to_numpy();point=np.asarray(point)
    return dict(**point_metrics(actual,point),nmae=float(np.mean(np.abs(point-actual)/np.maximum(frame.scale,1))),
        severe_underprediction=float(np.mean((actual>0)&(point<.5*actual))))


def select_policy(training,validation,config):
    if set(training.dataset)!= {'synthetic_training'} or set(validation.dataset)!= {'synthetic_validation'}:
        raise ValueError('Selection accepts only synthetic training and validation roles')
    cfg=config['policy'];base=scores(validation,validation.prediction)
    trials=[];fits={}
    for depth in cfg['depths']:
        fit=RankPolicy(depth,config['seed']).fit(training);fits[depth]=fit
        p=fit.probabilities(validation[FEATURES]);ranked=validation[['rank_0','rank_1','rank_2']].to_numpy()
        for rho in cfg['rhos']:
            values=convex_forecast(validation.prediction,ranked,p,rho)
            result=scores(validation,values)
            benefit=(result['mae']<base['mae']*(1-cfg['minimum_improvement']) and result['nmae']<base['nmae']*(1-cfg['minimum_improvement'])
                     and result['rmse']<=base['rmse']*(1+cfg['rmse_tolerance'])
                     and result['severe_underprediction']<=base['severe_underprediction']+cfg['underprediction_tolerance'])
            trials.append(dict(depth=depth,rho=rho,eligible=bool(benefit),objective=(result['mae']/max(base['mae'],np.finfo(float).eps)+result['nmae']/max(base['nmae'],np.finfo(float).eps))/2,**result))
    table=pd.DataFrame(trials);permitted=table.loc[table.eligible]
    if permitted.empty:
        # Select a candidate only on synthetic validation; preserve the original
        # point function by freezing rho=0 when promotion evidence is inadequate.
        chosen=table.query('rho==1').sort_values(['objective','depth']).iloc[0]
        rho=0.
    else:
        chosen=permitted.sort_values(['objective','rho','depth']).iloc[0];rho=float(chosen.rho)
    frozen=dict(depth=int(chosen.depth),rho=rho,seed=config['seed'],selected_on='synthetic_validation_only',
                baseline_validation=base,features=FEATURES,model='RandomForestClassifier',
                substitution='LightGBM unavailable; one shallow sklearn tree-ensemble architecture',policy=cfg,
                training_curves=int(training.region.nunique()),validation_curves=int(validation.region.nunique()))
    return fits[int(chosen.depth)],frozen,table


def predict_records(records,policy,rho,config):
    from experiments.forecasting.uncertainty import interval_quantiles,INITIAL_SIZES,QUANTILES
    probability=policy.probabilities(records[FEATURES]);ranked=records[['rank_0','rank_1','rank_2']].to_numpy()
    values=dict(M0=records.prediction.to_numpy(),M1=records.weighted.to_numpy(),
                M2=convex_forecast(records.prediction,ranked,probability,1.),
                M3=convex_forecast(records.prediction,ranked,probability,rho))
    parts=[]
    for model,point in values.items():
        frame=records.drop(columns=['actual'],errors='ignore').copy();frame['model']=model;frame['prediction']=point
        frame['original_prediction']=records.prediction.to_numpy()
        frame['forecast_adjustment']=point-records.prediction.to_numpy()
        for j in range(3):frame[f'probability_{j}']=probability[:,j]
        frame['rho']=rho if model=='M3' else 1. if model=='M2' else 0.
        if model not in ['M0']:
            quantiles=[]
            for row,value in zip(records.itertuples(),point):
                rng=np.random.default_rng(np.random.SeedSequence([config['seed'],int(row.history_sha256[:8],16),int(row.horizon)]))
                size=INITIAL_SIZES[int(row.horizon)-1]
                if value>=1:q=interval_quantiles(float(value),size,rng,draws=config['uncertainty']['draws'],initial=True)
                else:
                    # Same initial mean=point+1 and simulated median recentering,
                    # extended to valid nonnegative points below the legacy API floor.
                    draws=rng.negative_binomial(size,size/(size+value+1),size=config['uncertainty']['draws'])
                    q=np.quantile(draws,QUANTILES);q=value+q-q[3]
                quantiles.append(q)
            frame[QCOLS]=np.asarray(quantiles)
            frame['uncertainty']='modified_point_frozen_initial_nb'
        parts.append(frame)
    return pd.concat(parts,ignore_index=True)


def report(predictions,out,config):
    save_tables(predictions,out,config)
    # Augment shared paired metrics with policy-specific breakdowns and gains.
    extra=[]
    for dimension in ['family','count_scale','seed']:
        for (dataset,model,stratum),g in predictions.groupby(['dataset','model',dimension],dropna=False):
            extra.append(dict(dataset=dataset,model=model,dimension=dimension,stratum=str(stratum),**scores(g,g.prediction)))
    pd.DataFrame(extra).to_csv(out/'metrics_by_synthetic_stratum.csv',index=False)
    paired=[]
    for dataset,block in predictions.groupby('dataset'):
        base=block.query('model=="M0"').set_index(['case_id','horizon'])
        for model,g in block.groupby('model'):
            g=g.set_index(['case_id','horizon']).reindex(base.index)
            delta=np.abs(g.prediction-g.actual)-np.abs(base.prediction-base.actual)
            paired.append(dict(dataset=dataset,model=model,improved=int((delta<0).sum()),worsened=int((delta>0).sum()),
                               tied=int((delta==0).sum()),improved_fraction=float((delta<0).mean()),worsened_fraction=float((delta>0).mean()),mean_absolute_error_difference=float(delta.mean())))
    pd.DataFrame(paired).to_csv(out/'improved_worsened.csv',index=False)
    from experiments.robust.evaluation import paired_metrics
    paired_metrics(predictions,('dataset','phase')).to_csv(out/'metrics_by_phase.csv',index=False)
    paired_metrics(predictions,('dataset','region')).to_csv(out/'metrics_by_region.csv',index=False)


def policy_tree_shap(policy,training,validation,out,seed):
    import shap
    import matplotlib.pyplot as plt
    sample=validation.sample(min(200,len(validation)),random_state=seed)
    x=sample[FEATURES].to_numpy(float)
    explainer=shap.TreeExplainer(policy.model)
    explanation=explainer(x);explanation.feature_names=FEATURES
    p=policy.model.predict_proba(x)
    residual=float(np.max(np.abs(explanation.base_values+explanation.values.sum(axis=1)-p)))
    if residual>1e-6:raise ValueError('TreeSHAP additivity failed')
    rows=[]
    for column,rank in enumerate(policy.model.classes_):
        phi=explanation.values[:,:,column]
        for dim in ['global','horizon','history_length']:
            groups=[('all',np.arange(len(sample)))] if dim=='global' else [(key,np.flatnonzero(sample[dim].to_numpy()==key)) for key in sample[dim].unique()]
            for stratum,indices in groups:
                rows.extend(dict(rank=int(rank),dimension=dim,stratum=str(stratum),feature=f,mean_absolute_shap=float(np.abs(phi[indices,j]).mean())) for j,f in enumerate(FEATURES))
        e=shap.Explanation(values=phi,base_values=explanation.base_values[:,column],data=x,feature_names=FEATURES)
        shap.plots.beeswarm(e,show=False);plt.gcf().savefig(out/f'tree_shap_rank_{rank}.png',dpi=150,bbox_inches='tight');plt.close()
        shap.plots.waterfall(e[0],show=False);plt.gcf().savefig(out/f'tree_waterfall_rank_{rank}.png',dpi=150,bbox_inches='tight');plt.close()
    pd.DataFrame(rows).to_csv(out/'tree_shap_importance.csv',index=False)
    np.savez_compressed(out/'tree_shap_values.npz',values=explanation.values,base=explanation.base_values,features=x)
    atomic_json(out/'tree_shap_metadata.json',dict(target='Classifier probability for ordered forecast rank, not final forecast',
                classes=policy.model.classes_.tolist(),max_additivity_error=residual,sample='independent synthetic validation',seed=seed))


def run_policy(config,smoke=False):
    started=time.perf_counter();root=resolve(config['output_root'])/'context_policy'
    engine,reference_hashes,provenance=load_reference(config)
    # Earlier synthetic checks are development evidence. Exclude their whole
    # hashes, together with all previous generated roles, from the fresh roles.
    namespace=3931189327+int(smoke)
    forbidden=set(reference_hashes)
    for path in resolve(config['output_root']).rglob('metadata.json'):
        metadata=json.loads(path.read_text())
        if metadata.get('identity',{}).get('namespace')==namespace:continue
        for key,records in metadata.items():
            if key.endswith('_records') and isinstance(records,list):forbidden.update(r['sha256'] for r in records)
    counts={role:config['policy'][role+'_per_seed_family'] for role in ['training','validation','check','background']}
    # Namespace separated from original reference, prior robust and library studies.
    synthetic,folder=generate_independent(config,forbidden,provenance,root,smoke,role_counts=counts,namespace=namespace)
    identity=dict(config=config,smoke=smoke,reference=provenance,synthetic_identity=synthetic['identity'],synthetic_folder=str(folder),
                  source={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
                  python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn','shap']})
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=root/(('smoke_' if smoke else 'full_')+run_id);out.mkdir(parents=True,exist_ok=True)
    with acquire_run_lock(out/'run.lock'):
        atomic_json(out/'run_config.json',identity)
        atomic_json(out/'library_study_diagnosis.json',dict(full_windows=len(engine.x),sampled_windows=config['library_study']['window_budget'],
            m=engine.closest,full_neighbor_fraction=engine.closest/len(engine.x),
            sampled_neighbor_fraction=engine.closest/config['library_study']['window_budget'],
            conclusion='Full versus 100k libraries confounds composition with neighbourhood density; historical challenger-only selection excluded the better L0; future library selection now includes L0. Previous artifacts retained.'))
        frames={};case_sets={};diagnostics={};timings={}
        for role in ['training','validation']:
            cases,truth=synthetic_cases(synthetic,role,config['history_lengths']);case_sets[role]=cases
            tick=time.perf_counter();cache_id=hashlib.sha256(json.dumps([provenance,file_hash(Path(__file__).with_name("policy.py")),file_hash(Path(__file__)),config["uncertainty"]],sort_keys=True).encode()).hexdigest()[:16];path=folder/f'{role}_candidates_{cache_id}.pkl'
            if path.exists():
                frame=pd.read_pickle(path)
            else:
                issued,paths=prepare_records(engine,cases,config);frame=attach_outcomes(issued,truth)
                frame['family']=frame.region.map({r['curve_id']:r['family'] for r in synthetic[role+'_records']})
                frame['seed']=frame.region.map({r['curve_id']:r['seed_key'][3] for r in synthetic[role+'_records']})
                frame.to_pickle(path)
            frame['count_scale']=np.digitize(frame.scale,[1000,100000]);frames[role]=frame
            timings[role+'_preparation_seconds']=time.perf_counter()-tick
        tick=time.perf_counter();policy,frozen,trials=select_policy(frames['training'],frames['validation'],config)
        timings['training_and_selection_seconds']=time.perf_counter()-tick
        joblib.dump(policy,out/'policy.joblib');atomic_json(out/'selected_policy.json',frozen)
        trials.to_csv(out/'validation_selection.csv',index=False)
        # Fresh check candidates and labels remain separate until selection ends.
        check_cases,check_truth=synthetic_cases(synthetic,'check',config['history_lengths'])
        tick=time.perf_counter()
        check_issued,_=prepare_records(engine,check_cases,config)
        check_frame=attach_outcomes(check_issued,check_truth)
        check_frame['family']=check_frame.region.map({r['curve_id']:r['family'] for r in synthetic['check_records']})
        check_frame['seed']=check_frame.region.map({r['curve_id']:r['seed_key'][3] for r in synthetic['check_records']})
        check_frame['count_scale']=np.digitize(check_frame.scale,[1000,100000]);frames['check']=check_frame
        timings['check_preparation_seconds']=time.perf_counter()-tick
        # All choices frozen before opening any real case or scoring data.
        real=load_real_cases(config,smoke);tick=time.perf_counter()
        issued,paths=prepare_records(engine,real,config);diagnostics['real']=paths
        timings['real_matching_preparation_seconds']=time.perf_counter()-tick
        issued['family']='external';issued['seed']=-1;issued['count_scale']=np.digitize(issued.scale,[1000,100000])
        pred=predict_records(issued,policy,frozen['rho'],config)
        pred.to_csv(out/'issued_real_predictions.csv.gz',index=False)
        timings['real_end_to_end_issue_seconds']=time.perf_counter()-tick
        scored=attach_outcomes(pred,load_real_outcomes(config,real))
        results=[scored]
        for role in ['validation','check']:
            frame=frames[role];issued=predict_records(frame,policy,frozen['rho'],config)
            issued.to_csv(out/f'issued_{role}.csv.gz',index=False)
            results.append(attach_outcomes(issued,frame[['case_id','horizon','target_date','actual']]))
        # A fresh held-out generating mechanism, never used for policy selection.
        stress=[];records=[];all_hashes=forbidden|{r['sha256'] for role in counts for r in synthetic[role+'_records']}
        for attempt in range(config['policy']['stress_curves']*10):
            key=[config['seed'],3931189327,9,attempt]
            curve=change_point_sir(np.random.default_rng(np.random.SeedSequence(key)))
            onset=eligible(curve);digest=array_hash(curve.ts)
            if onset is None or digest in all_hashes:continue
            all_hashes.add(digest);stress.append(curve);records.append(dict(curve_id=f'policy_stress_{attempt}',sha256=digest,onset=onset,seed_key=key))
            if len(stress)==config['policy']['stress_curves']:break
        if len(stress)!=config['policy']['stress_curves']:raise RuntimeError('Insufficient independent stress curves')
        save_packed(stress,out/'stress.npz');atomic_json(out/'stress_provenance.json',records)
        synthetic['stress']=stress;synthetic['stress_records']=records
        cases,truth=synthetic_cases(synthetic,'stress',config['history_lengths'])
        stress_frame,_=prepare_records(engine,cases,config);stress_frame['family']='change_point_sir';stress_frame['seed']=-1
        stress_frame['count_scale']=np.digitize(stress_frame.scale,[1000,100000])
        stress_issued=predict_records(stress_frame,policy,frozen['rho'],config);stress_issued.to_csv(out/'issued_stress.csv.gz',index=False)
        results.append(attach_outcomes(stress_issued,truth))
        predictions=pd.concat(results,ignore_index=True);predictions.to_csv(out/'predictions.csv.gz',index=False)
        report(predictions,out,config)
        # Fixed alternate fitting seeds diagnose variance; check errors never
        # change selected depth, rho, or the primary seed.
        sensitivity=[]
        for seed in config['policy']['sensitivity_seeds']:
            fit=RankPolicy(frozen['depth'],seed).fit(frames['training'])
            for role in ['validation','check']:
                frame=frames[role];p=fit.probabilities(frame[FEATURES])
                values=convex_forecast(frame.prediction,frame[['rank_0','rank_1','rank_2']],p,1.)
                sensitivity.append(dict(seed=seed,role=role,**scores(frame,values)))
        pd.DataFrame(sensitivity).to_csv(out/'seed_sensitivity.csv',index=False)
        tick=time.perf_counter();policy_tree_shap(policy,frames['training'],frames['validation'],out,config['seed'])
        from experiments.robust.explain import explain_experiment
        atomic_json(out/'matching_diagnostics.json',diagnostics['real'])
        explain_experiment(engine,real,predictions,dict(temperature=4.,beta=0.),synthetic,config,out,
                           policy=policy,rho=frozen['rho'])
        timings['xai_seconds']=time.perf_counter()-tick
        atomic_json(out/'status.json',dict(status='complete',rho=frozen['rho'],smoke=smoke,
                    training_curves=frozen['training_curves'],check_curves=len(synthetic['check']),real_cases=len(real),
                    seconds=time.perf_counter()-started,timing=timings,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))
        print('Complete policy:',out,flush=True)
    return out
