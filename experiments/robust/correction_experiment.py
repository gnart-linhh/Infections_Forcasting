"""Fresh synthetic experiment for a single bounded conditional-median model."""
from __future__ import annotations
from pathlib import Path
import hashlib
import inspect
import json
import time
import resource
import platform
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from experiments.robust.data import resolve,array_hash,load_reference,load_real_cases,load_real_outcomes,attach_outcomes
from experiments.robust.policy import FEATURES,CORRECTION_MODELS,MedianCorrection,rank_labels
from experiments.robust.policy_experiment import prepare_records,scores,policy_tree_shap,point_quantiles
from experiments.robust.tuning import generate_independent,synthetic_cases,eligible
from experiments.robust.evaluation import paired_metrics,cluster_comparisons,uncertainty_metrics
from experiments.robust.explain import explain_experiment,plot_explanation
from experiments.runtime import atomic_json,file_hash,acquire_run_lock
from generator import change_point_sir
from bank import save_packed


def diagnose_previous(config,out):
    previous=resolve(config['median_correction']['previous_run'])
    data=pd.read_csv(previous/'predictions.csv.gz');rows=[];labels=[];confusions=[]
    expected={'synthetic_validation':(96659.261599,94440.804766),'synthetic_check':(112767.700231,106146.042269),
              'synthetic_stress':(829493.563195,818022.549242),'COVID19_early':(1436.100833,1510.445773)}
    for dataset,values in expected.items():
        for model,value in zip(['M0','M2'],values):
            g=data.loc[data.dataset.eq(dataset)&data.model.eq(model)]
            if not np.isclose(scores(g,g.prediction)['mae'],value,atol=.01,rtol=0):raise ValueError('Historical metric verification failed')
    for dataset,block in data.groupby('dataset'):
        base=block.query('model=="M0"').copy();policy=block.query('model=="M2"')
        truth_rank=rank_labels(base[['rank_0','rank_1','rank_2']].to_numpy(),base[['identity_0','identity_1','identity_2']].to_numpy(int),base.actual.to_numpy())
        probabilities=base[['probability_0','probability_1','probability_2']].to_numpy();predicted=np.argmax(probabilities,axis=1)
        matrix=confusion_matrix(truth_rank,predicted,labels=[0,1,2])
        for i in range(3):
            labels.append(dict(dataset=dataset,rank=i,label_frequency=float(np.mean(truth_rank==i)),predicted_frequency=float(np.mean(predicted==i)),mean_probability=float(probabilities[:,i].mean()),p10=float(np.quantile(probabilities[:,i],.1)),p90=float(np.quantile(probabilities[:,i],.9))))
            for j in range(3):confusions.append(dict(dataset=dataset,true_rank=i,predicted_rank=j,n=int(matrix[i,j])))
        for dim in ['overall','history_length','horizon','phase']:
            groups=[('all',policy)] if dim=='overall' else policy.groupby(dim)
            for key,g in groups:
                adjustment=g.prediction-g.candidate_1
                rows.append(dict(dataset=dataset,dimension=dim,stratum=str(key),**scores(g,g.prediction),
                    upward_adjustment=float(adjustment.clip(lower=0).mean()),downward_adjustment=float((-adjustment).clip(lower=0).mean()),
                    normalized_absolute_adjustment=float((abs(adjustment)/g.scale).mean()),
                    lower_gap=float((g.candidate_1-g.candidate_0).mean()),upper_gap=float((g.candidate_2-g.candidate_1).mean()),
                    candidate_spread=float((g.candidate_2-g.candidate_0).mean())))
    pd.DataFrame(rows).to_csv(out/'previous_policy_diagnosis.csv',index=False)
    pd.DataFrame(labels).to_csv(out/'rank_frequencies.csv',index=False)
    pd.DataFrame(confusions).to_csv(out/'rank_confusion.csv',index=False)
    pd.read_csv(previous/'validation_selection.csv').to_csv(out/'previous_selection.csv',index=False)
    atomic_json(out/'diagnosis.json',dict(source=str(previous),predictions_sha256=file_hash(previous/'predictions.csv.gz'),
        deployed_method='original_smoa',rho=0.,objective_mismatch='Argmin candidate classification followed by probability averaging does not minimize conditional absolute forecasting loss.',
        previous_gate='Raw and normalized gains were both required; rho=0 was a legitimate rejection, not a deployed learned improvement.'))


def cache_records(engine,cases,truth,synthetic,role,folder,config,provenance):
    signature=dict(reference=provenance,inputs=[(c.case_id,array_hash(c.history)) for c in cases],
        features=file_hash(Path(__file__).with_name('policy.py')),preparation=inspect.getsource(prepare_records),
        matching=file_hash(Path(__file__).with_name('method.py')),
        pipeline=file_hash(Path(__file__).with_name('pipeline.py')),
        truth=hashlib.sha256(pd.util.hash_pandas_object(truth,index=True).values.tobytes()).hexdigest(),point_only=True)
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    path=folder/f'{role}_point_records_{key}.pkl';metadata=path.with_suffix('.json')
    reused=path.exists() and metadata.exists()
    if reused:
        stored=json.load(open(metadata))
        if file_hash(path)!=stored['sha256'] or hashlib.sha256(json.dumps(stored['signature'],sort_keys=True).encode()).hexdigest()[:16]!=key:
            raise ValueError('Corrupt compact point cache')
        frame=pd.read_pickle(path)
    else:
        issued,_=prepare_records(engine,cases,config,point_only=True)
        frame=attach_outcomes(issued,truth)
        frame['family']=frame.region.map({r['curve_id']:r['family'] for r in synthetic[role+'_records']})
        frame['seed']=frame.region.map({r['curve_id']:r['seed_key'][3] for r in synthetic[role+'_records']})
        frame['count_scale']=np.digitize(frame.scale,[1000,100000]);frame.to_pickle(path)
        atomic_json(metadata,dict(signature=signature,sha256=file_hash(path)))
    return frame,reused


def choose_correction(training,validation,synthetic,config):
    if set(training.dataset)!={'synthetic_training'} or set(validation.dataset)!={'synthetic_validation'}:
        raise ValueError('Median selection is synthetic-only')
    cfg=config['median_correction'];gate=config['policy'];base=scores(validation,validation.candidate_1)
    records=pd.DataFrame(synthetic['training_records'])
    records['replicate']=records.seed_key.map(lambda k:k[3])
    small=set(records.groupby(['family','replicate'],sort=False).head(cfg['small_per_seed_family']).curve_id)
    cohorts={'small':training.loc[training.region.isin(small)],'large':training}
    fits={};rows=[]
    for cohort,frame in cohorts.items():
        fit=MedianCorrection(config['seed']).fit(frame);fits[cohort]=fit
        for rho in gate['rhos']:
            value=fit.predict(validation,rho);result=scores(validation,value)
            conditions=dict(raw_mae=result['mae']<base['mae']*(1-gate['minimum_improvement']),
                normalized_mae=result['nmae']<base['nmae']*(1-gate['minimum_improvement']),
                rmse=result['rmse']<=base['rmse']*(1+gate['rmse_tolerance']),
                underprediction=result['severe_underprediction']<=base['severe_underprediction']+gate['underprediction_tolerance'])
            rows.append(dict(training_size=cohort,training_curves=frame.region.nunique(),rho=rho,eligible=all(conditions.values()),
                rejection_reasons=';'.join(k for k,v in conditions.items() if not v),**result))
    table=pd.DataFrame(rows);valid=table.loc[table.eligible]
    if valid.empty:
        candidate=table.query('rho==1').sort_values(['mae','training_curves']).iloc[0];rho=0.
    else:candidate=valid.sort_values(['mae','training_curves','rho']).iloc[0];rho=float(candidate.rho)
    frozen=dict(selected_method='original_smoa' if rho==0 else 'bounded_median_blend',rho=rho,
        candidate_training_size=candidate.training_size,candidate_training_curves=int(candidate.training_curves),
        candidate_deployed=rho>0,primary_selection_metric='raw MAE among eligible configurations',
        acceptance_thresholds={k:gate[k] for k in ['minimum_improvement','rmse_tolerance','underprediction_tolerance']},
        secondary_acceptance_gates='Unchanged 1% normalized MAE gain, 5% RMSE and 2 percentage-point underprediction safeguards',
        baseline_validation=base,model='HistGradientBoostingRegressor(loss=absolute_error)',features=FEATURES,
        correction_envelope='convex hull of lower Q25, exact original, upper Q75; normalized adjustment clipped to [-1,1]',
        target='(synthetic_actual-original)/max(high-low,history_scale,1); no target clipping',seed=config['seed'])
    return fits[candidate.training_size],frozen,table,fits


def issue(records,rank,median,rho,config,uncertainty=False):
    # Use frozen rank inference and NB construction; only request quantiles for
    # external scoring, never for synthetic point-training preparations.
    base=records.candidate_1.to_numpy();weights=rank.probabilities(records[FEATURES])
    from experiments.robust.policy import convex_forecast
    values=dict(original_smoa=base,weighted_smoa=records.weighted.to_numpy(),
        historical_rank_policy=convex_forecast(base,records[['rank_0','rank_1','rank_2']],weights,1.),
        median_correction=median.predict(records),selected_forecast=median.predict(records,rho))
    parts=[]
    for name,point in values.items():
        frame=records.drop(columns=['actual'],errors='ignore').copy();frame['model']=name;frame['prediction']=point
        frame['forecast_adjustment']=point-base;frame['rho']=rho if name=='selected_forecast' else 1.
        if uncertainty:
            frame[QCOLS]=point_quantiles(records,point,config)
            frame['uncertainty']='frozen_initial_nb_point_comparator'
        parts.append(frame)
    return pd.concat(parts,ignore_index=True)


QCOLS=['q025','q100','q250','q500','q750','q900','q975']


def tables(predictions,out,config):
    for name,groups in [('metrics',['dataset']),('metrics_by_history_horizon',['dataset','history_length','horizon']),
                       ('metrics_by_phase',['dataset','phase']),('metrics_by_region',['dataset','region']),
                       ('metrics_by_synthetic_stratum',['dataset','family','count_scale'])]:
        frame=paired_metrics(predictions,groups,models=CORRECTION_MODELS,baseline_model='original_smoa')
        # NMAE is calculated on the same explicit paired keys.
        normal=predictions.assign(normalized_error=lambda d:abs(d.prediction-d.actual)/np.maximum(d.scale,1)).groupby(groups+['model'],as_index=False).normalized_error.mean()
        frame.merge(normal,on=groups+['model'],validate='one_to_one').rename(columns={'normalized_error':'nmae'}).to_csv(out/(name+'.csv'),index=False)
    cluster_comparisons(predictions,config['seed'],config['evaluation']['bootstrap_draws'],baseline_model='original_smoa').to_csv(out/'paired_comparisons.csv',index=False)
    gains=[]
    for dataset,block in predictions.groupby('dataset'):
        base=block.query('model=="original_smoa"').set_index(['case_id','horizon'])
        for model,g in block.groupby('model'):
            g=g.set_index(['case_id','horizon']).reindex(base.index)
            delta=abs(g.prediction-g.actual)-abs(base.prediction-base.actual)
            gains.append(dict(dataset=dataset,model=model,n=len(g),improved=int((delta<0).sum()),worsened=int((delta>0).sum()),ties=int((delta==0).sum())))
    pd.DataFrame(gains).to_csv(out/'improved_worsened.csv',index=False)
    metrics=pd.read_csv(out/'metrics.csv')
    metrics.query('dataset=="synthetic_check"').to_csv(out/'synthetic_check_metrics.csv',index=False)
    metrics.query('dataset=="synthetic_stress"').to_csv(out/'stress_metrics.csv',index=False)
    real=predictions.loc[predictions.dataset.eq('COVID19_early')]
    uncertainty_metrics(real).to_csv(out/'uncertainty_metrics.csv',index=False)


def feature_shift(frames,out):
    rows=[]
    train=frames['training']
    for dataset,frame in frames.items():
        for feature in FEATURES+['scale']:
            a=frame[feature].to_numpy();b=train[feature].to_numpy()
            rows.append(dict(dataset=dataset,feature=feature,n=len(a),median=float(np.median(a)),q10=float(np.quantile(a,.1)),q90=float(np.quantile(a,.9)),
                 fraction_outside_training_range=float(np.mean((a<b.min())|(a>b.max())))))
    pd.DataFrame(rows).to_csv(out/'feature_shift.csv',index=False)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(13,6))
    for ax,feature in zip(axes.flat,FEATURES+['scale']):
        for name in ['training','external']:
            values=np.sort(frames[name][feature]);ax.plot(values,np.arange(1,len(values)+1)/len(values),label=name)
        ax.set(title=feature);ax.set_xscale('symlog',linthresh=1)
    axes.flat[-1].axis('off');axes.flat[0].legend();fig.tight_layout();fig.savefig(out/'feature_shift.png',dpi=150);plt.close(fig)


class WeightedForecast:
    def forecast(self,engine,history,rho=1.):return engine.variants(history,4.,0.)[0]['M1'],{}


def correction_xai(engine,median,rank,synthetic,real,predictions,validation,config,out,rho):
    import shap
    import matplotlib.pyplot as plt
    sample=validation.sample(min(200,len(validation)),random_state=config['seed']);x=sample[FEATURES].to_numpy()
    explanation=shap.TreeExplainer(median.model)(x)
    residual=float(np.max(abs(explanation.base_values+explanation.values.sum(axis=1)-median.model.predict(x))))
    if residual>1e-6:raise ValueError('Median TreeSHAP additivity failed')
    np.savez_compressed(out/'correction_tree_values.npz',values=explanation.values,base=explanation.base_values,features=x)
    explanation.feature_names=FEATURES
    shap.plots.beeswarm(explanation,show=False);plt.gcf().savefig(out/'correction_tree_summary.png',dpi=150,bbox_inches='tight');plt.close()
    plot_explanation(explanation[0],'Candidate normalized correction before bounds/rescaling');plt.gcf().savefig(out/'correction_tree_waterfall.png',dpi=150,bbox_inches='tight');plt.close()
    pd.DataFrame(dict(feature=FEATURES,mean_absolute_shap=abs(explanation.values).mean(axis=0))).to_csv(out/'correction_feature_importance.csv',index=False)
    policy_tree_shap(rank,None,validation,out,config['seed'])
    cfg=json.loads(json.dumps(config));cfg['xai']['max_cases']=5
    metadata={}
    targets=[('weighted_smoa',WeightedForecast(),1.),('median_correction',median,1.)]
    if rho>0:targets.append(('selected_forecast',median,rho))
    for name,policy,blend in targets:
        for horizon in [1,4]:
            folder=out/f'xai_{name}_h{horizon}';folder.mkdir(exist_ok=True)
            cfg['xai']['horizon']=horizon
            explain_experiment(engine,real,predictions,dict(temperature=4.,beta=0.),synthetic,cfg,folder,
                policy=policy,rho=blend,baseline_model='original_smoa',forecast_model=name,artifact_directory='..')
            metadata[f'{name}_h{horizon}']=json.load(open(folder/'explanation_metadata.json'))
    atomic_json(out/'xai_metadata.json',dict(median_tree_target='Unbounded normalized residual median, before envelope and blending',tree_additivity_error=residual,
            old_rank_tree_target='Experimental candidate rank class probabilities, not deployed success',direct=metadata,
            deployed_zero_contrast='Not generated when selected rho=0; candidate correction is explicitly experimental',limitations='Background-dependent, correlated features, noncausal, sampled coalitions'))


def run_correction(config,smoke=False):
    started=time.perf_counter();cfg=config['median_correction'];root=resolve(config['output_root'])/'median_correction'
    engine,hashes,provenance=load_reference(config)
    namespace=3991872331+int(smoke);forbidden=set(hashes)
    for path in resolve(config['output_root']).rglob('metadata.json'):
        meta=json.load(open(path))
        if meta.get('identity',{}).get('namespace')==namespace:continue
        for name,records in meta.items():
            if name.endswith('_records') and isinstance(records,list):forbidden.update(r['sha256'] for r in records)
    for path in resolve(config['output_root']).rglob('stress_provenance.json'):
        if root in path.parents:continue
        forbidden.update(r['sha256'] for r in json.load(open(path)))
    counts={role:cfg[role+'_per_seed_family'] for role in ['training','validation','check','background']}
    synthetic,folder=generate_independent(config,forbidden,provenance,root,smoke,role_counts=counts,namespace=namespace)
    identity=dict(config=config,smoke=smoke,reference=provenance,synthetic_folder=str(folder),
        source={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        external_input_sha256=file_hash(resolve(config['evaluation']['database'])),
        packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn','shap']})
    ident=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=root/(('smoke_' if smoke else 'full_')+ident);out.mkdir(parents=True,exist_ok=True)
    with acquire_run_lock(out/'run.lock'),threadpool_limits(limits=1):
        if (out/'status.json').exists() and json.load(open(out/'status.json')).get('status')=='complete':
            print('Reusing immutable complete run:',out,flush=True);return out
        atomic_json(out/'run_config.json',identity)
        atomic_json(out/'status.json',dict(status='running',target_disease_training=False))
        diagnose_previous(config,out)
        frames={};timing={};cache={}
        for role in ['training','validation']:
            cases,truth=synthetic_cases(synthetic,role,config['history_lengths']);tick=time.perf_counter()
            frames[role],cache[role]=cache_records(engine,cases,truth,synthetic,role,folder,config,provenance)
            timing[role+'_preparation_seconds']=time.perf_counter()-tick
        tick=time.perf_counter();median,frozen,selection,fits=choose_correction(frames['training'],frames['validation'],synthetic,config)
        timing['training_selection_seconds']=time.perf_counter()-tick
        balance=frames['training'].groupby(['family','count_scale'],as_index=False).agg(records=('region','size'),curves=('region','nunique'))
        balance['total_training_weight']=1/len(balance)
        balance.to_csv(out/'training_balance.csv',index=False)
        joblib.dump(median,out/'median_model.joblib')
        frozen['candidate_model_sha256']=file_hash(out/'median_model.joblib')
        frozen['historical_classifier_sha256']=file_hash(resolve(cfg['previous_run'])/'policy.joblib')
        frozen['model_registry']={
            'original_smoa':'Exact original k5/m4422/GAM full-reference sMOA',
            'weighted_smoa':'Frozen temperature=4, beta=0 similarity-weighted median',
            'historical_rank_policy':'Frozen old depth-5 classifier; unblended experimental comparator',
            'median_correction':'New bounded conditional-median candidate, unblended',
            'selected_forecast':frozen['selected_method']}
        atomic_json(out/'selected_model.json',frozen);selection.to_csv(out/'validation_selection.csv',index=False)
        rank=joblib.load(resolve(cfg['previous_run'])/'policy.joblib')
        # Frozen model precedes check, stress, real data, and all external labels.
        cases,truth=synthetic_cases(synthetic,'check',config['history_lengths'])
        tick=time.perf_counter()
        frames['check'],cache['check']=cache_records(engine,cases,truth,synthetic,'check',folder,config,provenance)
        timing['check_preparation_seconds']=time.perf_counter()-tick
        stress_count=min(cfg['stress_curves'],3) if smoke else cfg['stress_curves']
        curves=[];records=[];used=forbidden|{r['sha256'] for role in counts for r in synthetic[role+'_records']}
        for attempt in range(stress_count*10):
            key=[config['seed'],namespace,9,attempt];curve=change_point_sir(np.random.default_rng(np.random.SeedSequence(key)))
            onset=eligible(curve);digest=array_hash(curve.ts)
            if onset is None or digest in used:continue
            used.add(digest);curves.append(curve);records.append(dict(curve_id=f'median_stress_{attempt}',sha256=digest,onset=onset,seed_key=key,family='change_point_sir'))
            if len(curves)==stress_count:break
        if len(curves)!=stress_count:raise RuntimeError('Insufficient stress curves')
        save_packed(curves,out/'stress.npz');atomic_json(out/'stress_provenance.json',records)
        synthetic['stress']=curves;synthetic['stress_records']=records
        cases,truth=synthetic_cases(synthetic,'stress',config['history_lengths'])
        tick=time.perf_counter()
        frames['stress'],cache['stress']=cache_records(engine,cases,truth,synthetic,'stress',folder,config,provenance)
        timing['stress_preparation_seconds']=time.perf_counter()-tick
        real=load_real_cases(config,smoke);tick=time.perf_counter()
        frame,diagnostics=prepare_records(engine,real,config,point_only=True)
        frame['family']='external';frame['count_scale']=np.digitize(frame.scale,[1000,100000]);frame['seed']=-1
        issued=issue(frame,rank,median,frozen['rho'],config,uncertainty=True)
        issued.to_csv(out/'issued_predictions.csv.gz',index=False)
        timing['real_end_to_end_issue_seconds']=time.perf_counter()-tick
        external=load_real_outcomes(config,real);scored=attach_outcomes(issued,external)
        frames['external']=attach_outcomes(frame,external)
        results=[scored]
        for role in ['validation','check','stress']:
            frame=frames[role];issued=issue(frame,rank,median,frozen['rho'],config)
            results.append(attach_outcomes(issued,frame[['case_id','horizon','target_date','actual']]))
        predictions=pd.concat(results,ignore_index=True);predictions.to_csv(out/'predictions.csv.gz',index=False)
        old=pd.read_csv(resolve(cfg['previous_run'])/'predictions.csv.gz')
        comparisons={}
        for name,oldname in [('original_smoa','M0'),('weighted_smoa','M1'),('historical_rank_policy','M2')]:
            a=scored.loc[scored.model.eq(name)].set_index(['case_id','horizon']).prediction
            b=old.loc[old.dataset.eq('COVID19_early')&old.model.eq(oldname)].set_index(['case_id','horizon']).prediction.reindex(a.index)
            comparisons[name]=float(np.max(abs(a-b)))
            if not np.allclose(a,b,rtol=1e-13,atol=1e-9):raise ValueError('Frozen historical baseline changed')
        atomic_json(out/'baseline_regression.json',comparisons)
        tables(predictions,out,config);feature_shift(frames,out)
        atomic_json(out/'matching_diagnostics.json',diagnostics)
        tick=time.perf_counter();correction_xai(engine,median,rank,synthetic,real,scored,frames['validation'],config,out,frozen['rho'])
        timing['xai_seconds']=time.perf_counter()-tick
        # Measure compact cache reads separately from fresh matching/preparation.
        tick=time.perf_counter()
        for role in ['training','validation','check']:
            cases,truth=synthetic_cases(synthetic,role,config['history_lengths'])
            replay,reused=cache_records(engine,cases,truth,synthetic,role,folder,config,provenance)
            if not reused:raise ValueError('Expected cache replay')
            pd.testing.assert_frame_equal(replay,frames[role],check_exact=True)
        timing['cached_point_record_replay_seconds']=time.perf_counter()-tick
        atomic_json(out/'runtime.json',dict(total_seconds=time.perf_counter()-started,phase_seconds=timing,cache_reused=cache,
                    build_kind='cached' if all(cache.values()) else 'fresh_or_partial',peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024)))
        atomic_json(out/'status.json',dict(status='complete',selected_method=frozen['selected_method'],rho=frozen['rho'],
                    target_disease_training=False,smoke=smoke,training_curves=len(synthetic['training']),check_curves=len(synthetic['check']),real_cases=len(real)))
        print('Complete correction:',out,flush=True)
    return out
