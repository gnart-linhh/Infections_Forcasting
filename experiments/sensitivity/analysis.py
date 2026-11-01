"""Paired sensitivity summaries, factorial contrasts, stability and scientific plots."""
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'infections-e2-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from design import configurations,REFERENCE,LEVELS

PARAMS=['N','k','m']
TARGETS=['series','forecast_origin','target_end_date','horizon']


def metrics(group):
    e=group.error.to_numpy(float)
    return dict(targets=len(group),MAE=float(np.mean(np.abs(e))),RMSE=float(np.sqrt(np.mean(e*e))),
                bias=float(np.mean(e)),normalized_MAE=float(np.mean(np.abs(e)/group.normalization)))


def factorial_effects(scores):
    rows=[]
    for seed,g in scores.groupby('seed'):
        g=g.loc[g.N.isin([1000,5000])&g.k.isin([3,7])&g.m.isin([500,2000])].copy()
        if len(g)!=8 or g.duplicated(PARAMS).any():raise ValueError('Factorial requires all eight unique corners per seed')
        signs={name:np.where(g[name].eq(low),-1,1) for name,low in [('N',1000),('k',3),('m',500)]}
        for order in [1,2,3]:
            for names in combinations(PARAMS,order):
                product=np.prod([signs[name] for name in names],axis=0)
                effect=float(2*np.mean(product*g.MAE))
                rows.append(dict(seed=seed,term=' × '.join(names),order=order,effect=effect,
                    difference_in_differences=2*effect if order==2 else np.nan))
    return pd.DataFrame(rows)


def conditional_interactions(scores):
    rows=[]
    levels={'N':[1000,5000],'k':[3,7],'m':[500,2000]}
    for seed,frame in scores.groupby('seed'):
        corners=frame.loc[frame.N.isin(levels['N'])&frame.k.isin(levels['k'])&frame.m.isin(levels['m'])]
        if len(corners)!=8 or corners.duplicated(PARAMS).any():raise ValueError('Incomplete conditional factorial')
        lookup=corners.set_index(PARAMS).MAE
        for condition,response,fixed in [('N','m','k'),('N','k','m'),('m','k','N')]:
            for fixed_value in levels[fixed]:
                changes=[]
                for condition_value in levels[condition]:
                    values=[]
                    for response_value in levels[response]:
                        params={condition:condition_value,response:response_value,fixed:fixed_value}
                        values.append(lookup[tuple(params[p] for p in PARAMS)])
                    changes.append(float(values[1]-values[0]))
                rows.append(dict(seed=seed,interaction=condition+' × '+response,fixed_parameter=fixed,fixed_value=fixed_value,
                    response_parameter=response,conditioning_parameter=condition,effect_at_low=changes[0],effect_at_high=changes[1],
                    difference_in_differences=changes[1]-changes[0]))
    return pd.DataFrame(rows)


def paired_deltas(data,draws=1000):
    reference=data.loc[data.N.eq(REFERENCE[0])&data.k.eq(REFERENCE[1])&data.m.eq(REFERENCE[2]),['seed',*TARGETS,'actual','error']]
    paired=data.merge(reference,on=['seed',*TARGETS],validate='many_to_one',suffixes=('','_reference'))
    if len(paired)!=len(data) or paired.error_reference.isna().any() or not paired.actual.eq(paired.actual_reference).all():raise ValueError('Paired targets/labels differ')
    paired['delta_absolute_error']=paired.error.abs()-paired.error_reference.abs()
    rows=[]
    for key,g in paired.groupby(PARAMS,sort=True):
        blocks=g.groupby('series').delta_absolute_error.agg(['sum','size'])
        rng=np.random.default_rng(20261031)
        sample=rng.integers(0,len(blocks),size=(draws,len(blocks)))
        distribution=blocks['sum'].to_numpy()[sample].sum(axis=1)/blocks['size'].to_numpy()[sample].sum(axis=1)
        lo,hi=np.quantile(distribution,[.025,.975])
        rows.append(dict(zip(PARAMS,key),delta_MAE=float(g.delta_absolute_error.mean()),delta_lo=float(lo),delta_hi=float(hi)))
    return pd.DataFrame(rows)


def segment_validity_audit(out,runtime):
    """Audit the recorded run's filter; never reinterpret old embeddings as v2."""
    from library import load_prefix
    from model import WINDOW_FILTER, valid_history_windows
    rows=[]
    runtime=runtime.copy()
    runtime['window_filter']=runtime.get('window_filter',pd.Series('numpy_variance_v1',index=runtime.index)).fillna('numpy_variance_v1')
    for seed in [0,1,2]:
        jobs=runtime.loc[runtime.seed.eq(seed),['N','k','M','window_filter']].drop_duplicates()
        if jobs.duplicated(['N','k']).any():raise ValueError('Mixed embedding filters in a sensitivity group')
        policies={(int(r.N),int(r.k)):r.window_filter for r in jobs.itertuples()}
        if not set(policies.values()) <= {WINDOW_FILTER,'numpy_variance_v1'}:raise ValueError('Unknown window filter')
        packed=out/f'packed/seed_{seed}'
        if jobs.N.max()>18387:packed=out/f'amendments/N{int(jobs.N.max())}/packed/seed_{seed}'
        curves,ids=load_prefix(packed,int(jobs.N.max()))
        with np.load(packed/'attempts.npz',allow_pickle=False) as source:attempts=dict(zip(source['curve_id'],source['index']))
        counters={(int(r.N),int(r.k)):[0,0] for r in jobs.itertuples()}
        for curve,ident in zip(curves,ids):
            values=np.maximum(curve.ts,1e-8);j=attempts[ident]
            for k in [3,5,7,9]:
                keys=[key for key in counters if key[1]==k and j<key[0]]
                if not keys or len(values)<k+6:continue
                history=np.lib.stride_tricks.sliding_window_view(values,k+6)[:,:k+1]
                for key in keys:
                    accepted=(valid_history_windows(history) if policies[key]==WINDOW_FILTER
                              else np.var(history,axis=1,ddof=1)>0)
                    count=int(accepted.sum());constant=int((accepted & (np.ptp(history,axis=1)==0)).sum())
                    counters[key][0]+=count;counters[key][1]+=constant
        for record in jobs.itertuples():
            m,constant=counters[(record.N,record.k)]
            if m!=record.M:raise ValueError('Validity audit disagrees with original embedding M')
            rows.append(dict(seed=seed,N=record.N,k=record.k,window_filter=record.window_filter,M_core=m,constant_history_rows=constant,
                constant_fraction=constant/m if m else 0.,nonconstant_rows=m-constant))
    return pd.DataFrame(rows)


def analyze(out):
    out=Path(out);groups=sorted((out/'groups').glob('seed_*'))
    from model import WINDOW_FILTER
    from runtime import file_hash
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['design'].get('window_filter')!=WINDOW_FILTER or protocol['design']['version']!=4:
        raise ValueError('Results use an outdated E2 protocol/window filter')
    frames=[];match_frames=[];runtime=[];early=[];availability=[]
    for folder in groups:
        if not (folder/'complete.json').exists():raise ValueError('Incomplete group')
        completion=json.loads((folder/'complete.json').read_text())
        for name,digest in completion['sha256'].items():
            if file_hash(folder/name)!=digest:raise ValueError('Modified E2 group artifact')
        info=json.loads((folder/'runtime.json').read_text())
        if info.get('window_filter')!=WINDOW_FILTER or info.get('constant_history_rows')!=0:
            raise ValueError('Results use an outdated filter or retain constant histories')
        if any(protocol['source_sha256'].get(name)!=digest for name,digest in info.get('source_sha256',{}).items()) or not info.get('source_sha256'):
            raise ValueError('E2 group numerical source differs from protocol')
        if info.get('context_sha256')!=file_hash(out/'contexts.npz'):raise ValueError('E2 group used different contexts')
        frames.append(pd.read_csv(folder/'forecasts.csv.gz',float_precision='round_trip'))
        match_frames.append(pd.read_csv(folder/'matches.csv.gz',float_precision='round_trip'))
        for m,details in info['per_m'].items():
            runtime.append(dict(seed=info['seed'],N=info['N'],k=info['k'],m=int(m),
                **{key:value for key,value in info.items() if key not in ['seed','N','k','per_m']},**details))
        if (folder/'early_forecasts.csv.gz').exists():early.append(pd.read_csv(folder/'early_forecasts.csv.gz'))
        if (folder/'early_availability.csv').exists():availability.append(pd.read_csv(folder/'early_availability.csv'))
    data=pd.concat(frames,ignore_index=True);matches=pd.concat(match_frames,ignore_index=True)
    if data.duplicated(['seed',*PARAMS,*TARGETS]).any():raise ValueError('Duplicate E2 targets')
    counts=data.groupby(['seed',*PARAMS]).size()
    expected={(seed,int(row.N),int(row.k),int(row.m)) for seed in [0,1,2] for row in configurations().itertuples()}
    if set(counts.index)!=expected or not counts.eq(10000).all():raise ValueError('Require all configured seeds/configurations on 10000 identical targets')
    scores=pd.DataFrame([dict(zip(['seed',*PARAMS],key),**metrics(g)) for key,g in data.groupby(['seed',*PARAMS])])
    horizon=pd.DataFrame([dict(zip(['seed',*PARAMS,'horizon'],key),**metrics(g)) for key,g in data.groupby(['seed',*PARAMS,'horizon'])])
    summary=scores.groupby(PARAMS,as_index=False).agg(MAE=('MAE','mean'),MAE_seed_sd=('MAE','std'),
        MAE_min=('MAE','min'),MAE_max=('MAE','max'),RMSE=('RMSE','mean'),bias=('bias','mean'),normalized_MAE=('normalized_MAE','mean'))
    summary=summary.merge(configurations(),on=PARAMS,validate='one_to_one').merge(paired_deltas(data),on=PARAMS,validate='one_to_one')
    stability=data.groupby(PARAMS+TARGETS).prediction.agg(['std','min','max']).reset_index()
    stable_summary=stability.groupby(PARAMS,as_index=False).agg(prediction_seed_sd_mean=('std','mean'),prediction_seed_range_mean=('max','mean'))
    # Range is the mean of per-target ranges, not a difference between pooled maxima.
    stability['seed_range']=stability['max']-stability['min']
    stable_summary['prediction_seed_range_mean']=stability.groupby(PARAMS).seed_range.mean().to_numpy()
    summary=summary.merge(stable_summary,on=PARAMS,validate='one_to_one')
    matching=matches.groupby(['seed',*PARAMS],as_index=False).agg(minimum_distance_mean=('minimum_distance','mean'),
        median_distance_mean=('median_distance','mean'),median_distance_per_difference_mean=('median_distance_per_difference','mean'),
        increment_iqr_mean=('increment_iqr_mean','mean'),mean_median_path_gap=('mean_median_path_gap','mean'))
    # The non-increase property concerns identical queries/k on truly nested banks.
    subset=matches.loc[matches.k.eq(5)&matches.m.eq(4422)&matches.N.isin(LEVELS['N'])]
    monotonic=subset.pivot(index=['seed','series','forecast_origin'],columns='N',values='minimum_distance').reindex(columns=LEVELS['N'])
    if monotonic.isna().any().any():raise ValueError('Missing nested-library matching observations')
    previous=monotonic.iloc[:,:-1].to_numpy();differences=np.diff(monotonic.to_numpy(),axis=1)
    violations=int(np.sum(differences>1e-8*(1+np.abs(previous))))
    if violations:raise ValueError('Nested minimum distance increased')
    monotonic.to_csv(out/'nested_minimum_distances.csv')
    m_subset=matches.loc[matches.N.eq(18387)&matches.k.eq(5)]
    median_by_m=m_subset.pivot(index=['seed','series','forecast_origin'],columns='m',values='median_distance').reindex(columns=LEVELS['m'])
    if median_by_m.isna().any().any():raise ValueError('Missing m sensitivity matching observations')
    median_decreases=int((np.diff(median_by_m.to_numpy(),axis=1)<-1e-8).sum())
    if median_decreases:raise ValueError('Increasing nearest-neighbor prefixes reduced median distance')
    performance=pd.DataFrame(runtime)
    validity=segment_validity_audit(out,performance)
    if validity.constant_history_rows.any():raise ValueError('Nonconstant filter retained constant windows')
    validity.to_csv(out/'segment_validity_audit.csv',index=False)
    context=json.loads((out/'context_metadata.json').read_text())
    performance['estimated_core_case_seconds']=performance.query_benchmark_mean_seconds+performance.aggregation_seconds/performance.cases+context['smoothing_seconds']/context['cases']
    effects=factorial_effects(scores)
    conditional=conditional_interactions(scores)
    conditional.to_csv(out/'conditional_interactions.csv',index=False)
    early_data=pd.concat(early,ignore_index=True)
    early_metrics=pd.DataFrame([dict(zip(['history_length','k'],key),**metrics(g)) for key,g in early_data.groupby(['history_length','k'])])
    availability_data=pd.concat(availability,ignore_index=True)
    availability_summary=availability_data.groupby(['history_length','k','status'],as_index=False).size()
    generation=[]
    for seed in [0,1,2]:
        meta=json.loads((out/f'packed/seed_{seed}/metadata.json').read_text())
        generation.append(dict(seed=seed,**meta))
    for name,frame in [('scores_by_seed',scores),('scores_by_horizon',horizon),('summary',summary),
        ('stability_by_target',stability),('matching_summary',matching),('runtime',performance),('factorial_effects',effects),
        ('early_metrics',early_metrics),('early_availability',availability_summary),('generation',pd.DataFrame(generation))]:
        frame.to_csv(out/f'{name}.csv',index=False)
    # Keep original child artifacts as auditable per-target and per-case records.
    validation=dict(protocol_version=4,window_filter=WINDOW_FILTER,constant_history_rows=int(validity.constant_history_rows.sum()),
        protocol_sha256=file_hash(out/'protocol.json'),source_sha256=protocol['source_sha256'],
        main_targets=len(data),configurations=len(summary),seeds=3,nested_queries=len(monotonic),
        minimum_distance_increase_violations=violations,matching_M_min=int(performance.M.min()),
        median_distance_decrease_with_m_violations=median_decreases,
        max_requested_m=8000,early_targets=len(early_data),early_unavailable_cases=int(availability_data.status.ne('ok').sum()))
    validation['constant_history_rows_reference_seed0']=int(validity.loc[validity.seed.eq(0)&validity.N.eq(18387)&validity.k.eq(5),'constant_history_rows'].iloc[0])
    # Seed zero reference must reproduce the audited Experiment 4 point forecasts.
    reference_path=out.parent/'experiment_4_omit/instances.csv'
    reference_meta=reference_path.with_name('protocol.json')
    reference_compatible=(reference_meta.exists() and json.loads(reference_meta.read_text()).get('source_sha256',{}).get('experiments/forecasting/model.py')
                          ==protocol['source_sha256']['experiments/forecasting/model.py'])
    if reference_path.exists() and reference_compatible:
        original=pd.read_csv(reference_path)
        reference=data.loc[data.seed.eq(0)&data.N.eq(18387)&data.k.eq(5)&data.m.eq(4422)]
        joined=reference.merge(original,on=TARGETS,validate='one_to_one',suffixes=('_e2','_e4'))
        if len(joined)!=10000 or not np.allclose(joined.prediction_e2,joined.prediction_e4,rtol=0,atol=1e-8) or not joined.actual_e2.eq(joined.actual_e4).all():
            raise ValueError('Paper seed0 reference does not reproduce Experiment 4')
        validation['reference_experiment4_equal_targets']=len(joined)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    plots(out,summary,horizon,matching,performance,effects,early_metrics)
    selected=validity.loc[validity.N.eq(18387)].groupby('k')[['nonconstant_rows','constant_history_rows']].mean()
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(selected.index,selected.nonconstant_rows/1e6,label='Nonconstant history')
    ax.bar(selected.index,selected.constant_history_rows/1e6,bottom=selected.nonconstant_rows/1e6,label='Constant history retained by float variance')
    ax.set(xticks=selected.index,xlabel='k',ylabel='Million rows; mean across seeds',title='nonconstant_v2 validity audit; N=18387');ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/'figures'/f'10_segment_validity.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    write_report(out,summary,performance,matching,effects,validation)
    return summary


def oat(frame,parameter):
    mask=np.ones(len(frame),dtype=bool)
    for name,value in zip(PARAMS,REFERENCE):
        if name!=parameter:mask &= frame[name].eq(value)
    return frame.loc[mask].sort_values(parameter)


def plots(out,summary,horizon,matching,runtime,effects,early):
    directory=out/'figures';directory.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(directory/f'{name}.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig)
    for metric,error,name in [('MAE','MAE_seed_sd','01_individual_mae'),('MAE_seed_sd',None,'05_seed_stability')]:
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,parameter in zip(axes,PARAMS):
            group=oat(summary,parameter)
            ax.errorbar(group[parameter],group[metric],yerr=group[error] if error else None,fmt='o-',capsize=4)
            if parameter in ['N','m']:ax.set_xscale('log')
            ax.set(xticks=group[parameter],xticklabels=group[parameter],xlabel=parameter,ylabel=metric,
                title=f'Vary {parameter}; other parameters fixed');ax.tick_params(axis='x',rotation=60 if parameter=='N' else 25,labelsize=8 if parameter=='N' else 10)
        save(fig,name)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,parameter in zip(axes,PARAMS):
        subset=oat(horizon,parameter)
        for h,g in subset.groupby('horizon'):
            grouped=g.groupby(parameter).MAE.mean();ax.plot(grouped.index,grouped,'o-',label=f'h={h}')
        if parameter in ['N','m']:ax.set_xscale('log')
        ax.set(xticks=sorted(subset[parameter].unique()),xticklabels=sorted(subset[parameter].unique()),xlabel=parameter,ylabel='MAE',title='Mean over 3 training seeds');ax.legend()
        ax.tick_params(axis='x',rotation=60 if parameter=='N' else 0,labelsize=8 if parameter=='N' else 10)
    save(fig,'02_horizon_sensitivity')
    factorial=summary.loc[summary.sections.str.contains('factorial')]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,k in zip(axes,[3,7]):
        grid=factorial.loc[factorial.k.eq(k)].pivot(index='m',columns='N',values='MAE').sort_index().sort_index(axis=1)
        image=ax.imshow(grid,cmap='viridis_r',vmin=factorial.MAE.min(),vmax=factorial.MAE.max(),aspect='auto')
        for i in range(len(grid)):
            for j in range(len(grid.columns)):ax.text(j,i,f'{grid.iloc[i,j]:,.0f}',ha='center',va='center',color='white',bbox=dict(facecolor='black',alpha=.3,edgecolor='none'))
        ax.set(xticks=range(len(grid.columns)),xticklabels=grid.columns,yticks=range(len(grid)),yticklabels=grid.index,xlabel='N attempts/family',ylabel='m neighbors',title=f'k={k}; MAE, 3-seed mean')
        fig.colorbar(image,ax=ax,label='MAE')
    save(fig,'03_factorial_heatmaps')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    cm=plt.colormaps['viridis_r'].copy();cm.set_bad('#dddddd')
    for ax,k in zip(axes.flat,[3,5,7,9]):
        grid=summary.loc[summary.k.eq(k)].pivot(index='m',columns='N',values='MAE').sort_index().sort_index(axis=1)
        image=ax.imshow(np.ma.masked_invalid(grid.to_numpy()),cmap=cm,vmin=summary.MAE.min(),vmax=summary.MAE.max(),aspect='auto')
        for i in range(len(grid)):
            for j in range(len(grid.columns)):
                value=grid.iloc[i,j]
                if pd.notna(value):ax.text(j,i,f'{value:,.0f}',ha='center',va='center',color='white',fontsize=8,bbox=dict(facecolor='black',alpha=.35,edgecolor='none'))
        ax.set(xticks=range(len(grid.columns)),xticklabels=grid.columns,yticks=range(len(grid)),yticklabels=grid.index,xlabel='N',ylabel='m',title=f'k={k}; gray cells were not evaluated')
        ax.tick_params(axis='x',rotation=25);fig.colorbar(image,ax=ax,label='MAE (same scale)')
    save(fig,'11_observed_grid_by_k')
    fig,ax=plt.subplots(figsize=(11,4))
    order=effects.term.drop_duplicates().tolist()
    for seed,g in effects.groupby('seed'):
        g=g.set_index('term').reindex(order);ax.plot(range(len(order)),g.effect,'o-',label=f'seed {seed}')
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(len(order)),xticklabels=order,ylabel='Coded factorial effect on MAE',title='Main and interaction contrasts; descriptive, not significance tests');ax.legend()
    save(fig,'04_interaction_effects')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,parameter in zip(axes,PARAMS):
        group=oat(matching,parameter).groupby(parameter)[['minimum_distance_mean','median_distance_mean']].mean()
        for field in group:ax.plot(group.index,group[field],'o-',label=field.replace('_distance_mean',''))
        if parameter in ['N','m']:ax.set_xscale('log')
        ax.set(xticks=group.index,xticklabels=group.index,xlabel=parameter,ylabel='Raw L1 distance',yscale='symlog',title='Same targets; k changes distance geometry');ax.legend(fontsize=8)
        ax.tick_params(axis='x',rotation=60 if parameter=='N' else 0,labelsize=8 if parameter=='N' else 10)
    save(fig,'06_matching_quality')
    speed=runtime.groupby(PARAMS,as_index=False).agg(seconds=('estimated_core_case_seconds','mean'),peak_rss=('peak_rss_bytes','max')).merge(summary,on=PARAMS)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for k,g in speed.groupby('k'):
        axes[0].scatter(1000*g.seconds,g.MAE,label=f'k={k}')
        axes[1].scatter(g.peak_rss/1024**3,g.MAE,label=f'k={k}')
    axes[0].set(xlabel='Estimated warmed core ms/case (separate query benchmark)',ylabel='MAE',title='Setup and diagnostic I/O excluded')
    axes[1].set(xlabel='Isolated group peak RSS (GiB)',ylabel='MAE',title='Includes shared maximum-m query and loaded bank')
    axes[0].legend();axes[1].legend();save(fig,'07_accuracy_cost')
    fig,ax=plt.subplots(figsize=(8,4))
    grid=early.pivot(index='history_length',columns='k',values='MAE').reindex(index=[4,6,8,10],columns=[3,5,7,9])
    masked=np.ma.masked_invalid(grid.to_numpy());cm=plt.colormaps['viridis_r'].copy();cm.set_bad('#dddddd')
    image=ax.imshow(masked,aspect='auto',cmap=cm)
    for i in range(4):
        for j in range(4):ax.text(j,i,'unavailable' if pd.isna(grid.iloc[i,j]) else f'{grid.iloc[i,j]:,.0f}',ha='center',va='center',fontsize=9,color='black',bbox=dict(facecolor='white',alpha=.65,edgecolor='none'))
    ax.set(xticks=range(4),xticklabels=[3,5,7,9],yticks=range(4),yticklabels=[4,6,8,10],xlabel='k',ylabel='Observed last L weeks',title='Controlled short-history MAE; seed 0, first 16 origins, same cases per L')
    fig.colorbar(image,ax=ax,label='MAE');save(fig,'08_short_history')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    merged=oat(matching,'m').groupby('m',as_index=False)[['increment_iqr_mean','mean_median_path_gap']].mean()
    for ax,field in zip(axes,['increment_iqr_mean','mean_median_path_gap']):
        ax.plot(merged.m,merged[field],'o-');ax.set(xlabel='m',ylabel=field,title='Selected analogue futures; N=18387, k=5')
    save(fig,'09_aggregation_behavior')


def write_report(out,summary,runtime,matching,effects,validation):
    def markdown(df):
        df=df.copy()
        for col in df.select_dtypes('number'):
            df[col]=df[col].map(lambda v:f'{int(v):,}' if pd.notna(v) else 'NA') if col in ['N','k','m','order'] else df[col].map(lambda v:f'{v:,.3f}' if pd.notna(v) else 'NA')
        return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))
    reference=summary.loc[summary.N.eq(18387)&summary.k.eq(5)&summary.m.eq(4422)].iloc[0]
    lines=['# Experiment 2 — Individual sensitivity, interactions and trade-offs','',
        f'{len(summary)} configurations ×3 independent training banks ×10.000 common COVID targets = {len(summary)*30000:,} point forecasts. GAM, exact L1, omit bridge; no BO or UQ rerun. The calendar is unchanged from Experiment 4.',
        '', 'Protocol v4, nonconstant_v2: exactly constant histories retained = 0. N=22000 and N=30000 remain the user-requested levels from the earlier study. Existing banks/prefixes are hash-verified and reused; forecasts are recomputed, not imported from v1. No levels are chosen again from COVID errors.',
        '',f"Reference (18387,5,4422): mean MAE {reference.MAE:,.2f}, seed SD {reference.MAE_seed_sd:,.2f}. E4 equality is checked only when its numerical source matches; a legacy E4 run is not a correctness oracle for the new filter.",
        '', '## E2.A — Individual sensitivity','',
        'Each row changes one parameter; all other reference parameters remain fixed. Error bars show SD across three training seeds, not a confidence interval. COVID results are descriptive, not selection feedback.','']
    for parameter in PARAMS:
        group=oat(summary,parameter)
        lines.extend(['',f'### {parameter}','',markdown(group[[parameter,'MAE','MAE_seed_sd','RMSE','delta_MAE','delta_lo','delta_hi']]),''])
        lo,hi=group.iloc[0],group.iloc[-1]
        lines.append(f"Endpoint comparison for {parameter}: MAE {lo.MAE:,.2f} → {hi.MAE:,.2f}. Check intermediate rows; accuracy need not be monotone.")
    n_values=oat(summary,'N')
    increases=int((np.diff(n_values.MAE)>0).sum())
    lines += ['',f'Mean MAE increases in {increases} of the {len(n_values)-1} adjacent N expansions, although nearest matching distance never increases on these nested banks. This separates past matching quality from future forecasting quality.']
    lines += ['',f"Nested N check: {validation['nested_queries']:,} seed/state/origin queries, zero minimum-distance increase violations. This verifies matching coverage, not monotonic forecast accuracy.",
        '', 'Paired delta = configuration absolute error minus same-seed reference error. 95% percentile intervals resample whole states (1000 draws), keeping origins/horizons/seeds within state. They describe this panel; they do not establish nationwide independence or account for selecting a best row after viewing test.',
        '', '## E2.B — Joint interactions','',markdown(summary.loc[summary.sections.str.contains('factorial|center')][['N','k','m','MAE','MAE_seed_sd']]),
        '',markdown(effects.groupby(['term','order'],as_index=False).agg(effect_mean=('effect','mean'),effect_seed_sd=('effect','std'))),
        '', 'conditional_interactions.csv also reports each pairwise difference-in-differences separately at both levels of the third factor, so opposite conditional effects are not hidden by an average contrast.',
        '', 'Effects use coded factors ±1: 2×mean(sign product×MAE). A pairwise difference-in-differences is twice the listed coded interaction effect and is exported separately. Center/reference are not factorial corners. Three-way effects are exploratory; there are only eight corners and three seeds.',
        '', '## E2.C — Stability, matching and computational cost','',
        'summary.csv includes seed SD of MAE and average per-target prediction SD/range. matching_summary.csv includes minimum/median L1, median L1/k, incremental IQR and mean-versus-median path differences. L1/k does not eliminate cross-k geometry differences.',
        '', 'runtime.csv reports accepted curves and valid segments M; prefix loading, embedding and tree setup; shared maximum-m query wall time; independent warmed 32-query benchmarks for each m; aggregation time and isolated group peak RSS. The estimated core time adds smoothing + per-m query benchmark + median aggregation; it is not measured end-to-end job time.',
        '', 'Peak RSS includes bank loading, embedding, KD tree, shared maximum-m work and diagnostics, so it is a group peak, not an independent per-m memory benchmark. Generation process/worker RSS is excluded. Disk scratch is removed after every group.',
        '', 'Prefix loading decompresses the full largest packed bank before selecting a nested subset; smaller-N peak RSS includes this common backing bank. Group wall time also includes diagnostic I/O and the auxiliary short-history study in seed0 full-N jobs. Use stage timings and separate per-m benchmarks rather than ranking seeds by total job time.',
        '', 'generation.csv reports actual largest-bank generation+CSV export duration using file timestamps at lock acquisition and metadata completion (imports excluded), plus packing time. Smaller N uses exact prefixes; no standalone smaller-N generation wall time is invented.',
        '', 'Generation timing for seed0 is a fresh complete replay, not the original historical duration. Seeds 1/2 were generated concurrently and seed0 replay alone; these wall times reflect different system load.',
        '', 'extension_generation.csv separately records each incremental extension and its generation duration (18387→22000→30000). Extended banks append new streams; original scalar family seeds are never shifted by changing N.',
        '', f"Minimum M across jobs: {validation['matching_M_min']:,}; every requested m (including 8000) is validated against its own library M before querying.",
        '', '## Controlled short-history availability','',
        'Separate seed-0 analysis at first 16 genuine panel origins, exposing only last L=4/6/8/10 weekly observations. This is controlled data availability, not an earlier vintage. Unavailable k/L combinations remain explicit; no invalid forecast or fallback is forced. GAM core requires at least 5 observations, so even k=3 cannot run at L=4 without changing software.',
        '', '## Limits','',
        '- N is attempts per family; seasonal rejections are retained and never replaced. All seed-level banks have the same largest-N family namespace; nested subsets select attempt indices, not first N accepted curves.',
        '- k changes both required history and the number of valid segments M. No equal-M ablation is performed; both N and M are exported.',
        f"- Numerical core audit: seed0 reference retains {validation['constant_history_rows_reference_seed0']:,} exactly constant-history windows. segment_validity_audit.csv records the filter for each N/k/seed: legacy numpy_variance_v1 may retain constant rows due to rounding; nonconstant_v2 excludes them exactly. Existing predictions are never relabelled as corrected runs. k effects also include the recorded filter's behavior.",
        '- Common main origins have >=10 observations. Short-history results are separate and must only compare supported k within the same L.',
        '- Population, wave phase and reporting zeros remain as in the fixed panel. This study does not repair data, infer causal parameter effects, or prove numerical equivalence with R/mgcv.',
        '- RMSE in summary.csv is the mean of seed-specific RMSE; per-seed/horizon metrics are available separately. Three seeds give limited stability evidence.','']
    for name in ['01_individual_mae','02_horizon_sensitivity','03_factorial_heatmaps','11_observed_grid_by_k','04_interaction_effects','05_seed_stability','06_matching_quality','07_accuracy_cost','08_short_history','09_aggregation_behavior','10_segment_validity']:
        lines += [f'![{name}](figures/{name}.png)','']
    (out/'report.md').write_text('\n'.join(lines))
    from runtime import file_hash,atomic_json
    atomic_json(out/'report_metadata.json',dict(analysis_source_sha256=file_hash(Path(__file__)),
        artifacts_sha256={str(p.relative_to(out)):file_hash(p) for p in [*out.glob('*.csv'),out/'validation.json',out/'report.md',*out.joinpath('figures').glob('*')]}))
