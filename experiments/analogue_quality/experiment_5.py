"""E5: exact analogue distances, conditional error analysis and held-out risk diagnostics."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'infections-e5-matplotlib'))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from runtime import atomic_json,file_hash,acquire_run_lock
from neighbors import ExactNeighbors
from experiments.error_analysis.conditions import features
from experiments.error_analysis.library import embeddings

PARAMETERS=dict(N=18387,k=5,m=4422,h=4,seed=0,smoothing='gam',bridge='omit')
KEYS=['series','forecast_origin']
DISTANCES=['minimum_distance','mean_distance','median_distance']
CUTOFF='2020-12-31'
TEST_START='2021-01-01'
RULES=dict(parameters=PARAMETERS,cohort='unchanged E2/E4: 50 states x50 origins x4 horizons',
    primary_distance='mean_distance_scaled',
    scale='max(mean(last 6 origin-available raw weekly values floored at 1),1)',
    scaled_distance='raw top-m L1 distance / (5 * level_scale); not a different neighbor search',
    error_scale='E4 max(mean(last 4 origin-available floored values),1)',
    extreme='absolute forecast error / error_scale >1',
    bins='training-case distance quintiles, unique boundaries, ties stay together; no test quantiles',
    good_match='mean_distance_scaled <= training-case 20th percentile',
    split=f'train: all four targets <= {CUTOFF}; test origins >= {TEST_START}; intervening cases embargoed',
    risk_training_labels='historical truth in first test-origin JHU vintage; never later revised/future labels',
    risk_test_labels='final truth, used only to evaluate',
    risk_models='distance only; origin context only; context plus distance; separate logistic fit per horizon',
    context='log1p(level_scale),log1p(recent_cases),asinh(growth),volatility,log1p(history_weeks),phase indicators',
    diagnostic_fit='train-only standardization; intercept; L2 penalty=1 on coefficients; no tuning on test',
    bootstrap='500 whole-state resamples, fixed seed 20261101; preserve origins/horizons within state',
    secondary_seeds='E2 seeds 1/2: stored median-distance correlation only; mean distance not recorded there',
    counterexamples='all good-match extreme-error instances; plot worst four distinct states, selected post hoc')


def spearman(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return float('nan')
    return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])


def context_matrix(frame):
    return np.column_stack([np.log1p(frame.level_scale),np.log1p(frame.recent_cases),
        np.arcsinh(frame.growth),frame.volatility,np.log1p(frame.history_weeks),
        frame.phase.eq('rising'),frame.phase.eq('declining')]).astype(float)


def partial_spearman(frame,x='mean_distance_scaled',y='normalized_absolute_error'):
    controls=context_matrix(frame)
    controls=np.column_stack([np.ones(len(frame)),rankdata(controls,axis=0)])
    rx=rankdata(frame[x]);ry=rankdata(frame[y])
    rx=rx-controls@np.linalg.lstsq(controls,rx,rcond=None)[0]
    ry=ry-controls@np.linalg.lstsq(controls,ry,rcond=None)[0]
    if np.std(rx)<1e-10 or np.std(ry)<1e-10:return float('nan')
    return float(np.corrcoef(rx,ry)[0,1])


def distance_bins(training_values,values):
    training_values=np.asarray(training_values,float)
    if not len(training_values) or not np.isfinite(training_values).all():raise ValueError('Invalid training distances')
    cuts=np.unique(np.quantile(training_values,[.2,.4,.6,.8]))
    # Keep coincident quantiles together instead of inventing ranks within ties.
    return 1+np.searchsorted(cuts,np.asarray(values,float),side='left'),cuts


def auc(y,score):
    y=np.asarray(y,bool);score=np.asarray(score,float)
    positive=int(y.sum());negative=len(y)-positive
    if not positive or not negative:return float('nan')
    return float((rankdata(score)[y].sum()-positive*(positive+1)/2)/(positive*negative))


def average_precision(y,score):
    y=np.asarray(y,int);score=np.asarray(score,float)
    if not y.sum():return float('nan')
    order=np.argsort(-score,kind='stable');s=score[order];labels=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(labels)[ends];precision=tp/(ends+1);recall=tp/labels.sum()
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def fit_risk(x,y,test_x):
    x=np.asarray(x,float);test_x=np.asarray(test_x,float);y=np.asarray(y,float)
    center=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
    design=np.column_stack([np.ones(len(x)),(x-center)/scale])
    test=np.column_stack([np.ones(len(test_x)),(test_x-center)/scale])
    if len(np.unique(y))<2:
        return np.full(len(test),y.mean()),dict(status='constant_training_labels',coefficients=None)
    def objective(beta):
        z=design@beta
        loss=np.mean(np.logaddexp(0,z)-y*z)+.5*np.sum(beta[1:]**2)/len(y)
        gradient=design.T@(expit(z)-y)/len(y);gradient[1:]+=beta[1:]/len(y)
        return loss,gradient
    fit=minimize(objective,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'gtol':1e-8})
    if not fit.success:raise ValueError('Risk diagnostic fit failed: '+fit.message)
    return expit(test@fit.x),dict(status='ok',coefficients=fit.x.tolist(),train_center=center.tolist(),train_scale=scale.tolist())


def state_samples(frame,draws=500):
    groups=[np.flatnonzero(frame.series.to_numpy()==state) for state in sorted(frame.series.unique())]
    rng=np.random.default_rng(20261101)
    for _ in range(draws):yield np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])


def rho_interval(frame):
    values=[]
    x=frame.mean_distance_scaled.to_numpy();y=frame.normalized_absolute_error.to_numpy()
    for ids in state_samples(frame):values.append(spearman(x[ids],y[ids]))
    finite=np.asarray(values)[np.isfinite(values)]
    return np.quantile(finite,[.025,.975]).tolist() if len(finite) else [np.nan,np.nan]


def prepare_cases(panel,contexts):
    rows=[];histories={key:g.sort_values('target_end_date') for key,g in panel.groupby(KEYS,sort=False)}
    for i,(state,origin) in enumerate(zip(contexts['series'],contexts['origin'])):
        g=histories[(state,origin)]
        if g.target_end_date.iloc[-1]!=origin:raise ValueError('History extends beyond/differs from origin')
        if not np.isfinite(g.value).all():raise ValueError('Nonfinite origin history')
        f=features(g.value.to_numpy())
        if f['normalization']!=contexts['normalization'][i]:raise ValueError('History scale differs from E2')
        rows.append(dict(series=state,forecast_origin=origin,case_index=i,
            level_scale=float(np.mean(np.maximum(g.value.to_numpy()[-6:],1))),**f))
    return pd.DataFrame(rows)


def replay_distances(cases,contexts,x,y,stored_matches,stored_forecasts,m=4422):
    if m>len(x):raise ValueError('Requested m exceeds valid segments')
    index=ExactNeighbors(x,workers=1);rows=[]
    expected=stored_forecasts.set_index(KEYS+['horizon'])
    matching=stored_matches.set_index(KEYS)
    for start in range(0,len(cases),32):
        ids,distances=index.query(contexts['query_k5'][start:start+32],m)
        for j,(neighbors,distance) in enumerate(zip(ids,distances),start):
            state=str(contexts['series'][j]);origin=str(contexts['origin'][j]);known=matching.loc[(state,origin)]
            point=np.maximum(1,contexts['last'][j]+np.cumsum(np.median(y[neighbors],axis=0)))
            values=dict(minimum_distance=float(distance[0]),mean_distance=float(distance.mean()),
                median_distance=float(np.median(distance)),maximum_distance=float(distance[-1]))
            for field in ['minimum_distance','median_distance','maximum_distance']:
                if not np.isclose(values[field],known[field],rtol=1e-10,atol=1e-8):raise ValueError('Replayed distance differs from E2')
            for h,pred in enumerate(point,1):
                if float(expected.loc[(state,origin,h),'prediction'])!=pred:raise ValueError('Replayed point differs from E2')
            rows.append(dict(series=state,forecast_origin=origin,**values))
    return cases.merge(pd.DataFrame(rows),on=KEYS,validate='one_to_one')


def join_instances(cases,forecasts):
    if cases.duplicated(KEYS).any() or forecasts.duplicated(KEYS+['horizon']).any():raise ValueError('Duplicate cases or targets')
    result=forecasts.drop(columns=['normalization']).merge(cases,on=KEYS,how='left',validate='many_to_one',indicator=True)
    if not result._merge.eq('both').all():raise ValueError('Forecast missing matching diagnostics')
    result=result.drop(columns='_merge')
    days=(pd.to_datetime(result.target_end_date)-pd.to_datetime(result.forecast_origin)).dt.days
    if not days.eq(result.horizon*7).all():raise ValueError('Target dates/horizons differ')
    result['absolute_error']=(result.prediction-result.actual).abs()
    result['normalized_absolute_error']=result.absolute_error/result.normalization
    result['extreme']=result.normalized_absolute_error>1
    return result


def correlations(data):
    rows=[]
    for scope in ['all','test']:
        selected=data if scope=='all' else data.loc[data.split.eq('test')]
        for h,g in selected.groupby('horizon'):
            for field in DISTANCES:
                for scaled in [False,True]:
                    distance=field+('_scaled' if scaled else '')
                    for error in ['absolute_error','normalized_absolute_error']:
                        lo=hi=np.nan
                        if distance=='mean_distance_scaled' and error=='normalized_absolute_error':lo,hi=rho_interval(g)
                        rows.append(dict(seed=0,scope=scope,stratum='all',horizon=h,distance=distance,error=error,
                            cases=len(g),states=g.series.nunique(),rho=spearman(g[distance],g[error]),lo=lo,hi=hi))
            rows.append(dict(seed=0,scope=scope,stratum='context_adjusted',horizon=h,distance='mean_distance_scaled',
                error='normalized_absolute_error',cases=len(g),states=g.series.nunique(),rho=partial_spearman(g),lo=np.nan,hi=np.nan))
            if scope=='test':
                for dimension in ['phase','case_level']:
                    for level,part in g.groupby(dimension):
                        rows.append(dict(seed=0,scope=scope,stratum=dimension+'='+level,horizon=h,
                            distance='mean_distance_scaled',error='normalized_absolute_error',cases=len(part),states=part.series.nunique(),
                            rho=spearman(part.mean_distance_scaled,part.normalized_absolute_error) if len(part)>=20 else np.nan,lo=np.nan,hi=np.nan))
    # A pooled case-level score does not count the four correlated horizons as independent cases.
    cases=data.groupby(KEYS,as_index=False).agg(mean_distance_scaled=('mean_distance_scaled','first'),
        normalized_absolute_error=('normalized_absolute_error','mean'))
    lo,hi=rho_interval(cases)
    rows.append(dict(seed=0,scope='all',stratum='case_mean_h1_h4',horizon=0,distance='mean_distance_scaled',
        error='normalized_absolute_error',cases=len(cases),states=cases.series.nunique(),
        rho=spearman(cases.mean_distance_scaled,cases.normalized_absolute_error),lo=lo,hi=hi))
    return pd.DataFrame(rows)


def risk_diagnostics(data,panel):
    train=data.loc[data.split.eq('train')].copy();test=data.loc[data.split.eq('test')].copy()
    first=test.forecast_origin.min()
    vintage=panel.loc[panel.forecast_origin.eq(first),['series','target_end_date','value']]
    train=train.merge(vintage,on=['series','target_end_date'],how='left',validate='many_to_one')
    if train.value.isna().any() or not train.target_end_date.le(CUTOFF).all():raise ValueError('Missing or future training outcome')
    train['risk_label']=((train.prediction-train.value).abs()/train.normalization>1).astype(int)
    scores=[];predictions=[];fits={}
    for h in [1,2,3,4]:
        a=train.loc[train.horizon.eq(h)].sort_values(KEYS);b=test.loc[test.horizon.eq(h)].sort_values(KEYS).reset_index(drop=True)
        xa=context_matrix(a);xb=context_matrix(b)
        da=np.log1p(a.mean_distance_scaled.to_numpy())[:,None];db=np.log1p(b.mean_distance_scaled.to_numpy())[:,None]
        probabilities={}
        for name,xx,yy in [('distance_only',da,db),('context_only',xa,xb),('context_plus_distance',np.column_stack([xa,da]),np.column_stack([xb,db]))]:
            p,fit=fit_risk(xx,a.risk_label.to_numpy(),yy);probabilities[name]=p
            fits[f'h{h}_{name}']=dict(fit,train_cases=len(a),training_label_vintage=first,max_training_target=a.target_end_date.max())
        target=b.extreme.to_numpy(bool)
        bootstrap={name:[] for name in probabilities};deltas=[];brier_deltas=[]
        for ids in state_samples(b):
            measures={name:auc(target[ids],p[ids]) for name,p in probabilities.items()}
            for name,value in measures.items():bootstrap[name].append(value)
            deltas.append(measures['context_plus_distance']-measures['context_only'])
            brier_deltas.append(np.mean((probabilities['context_plus_distance'][ids]-target[ids])**2)-np.mean((probabilities['context_only'][ids]-target[ids])**2))
        for name,p in probabilities.items():
            finite=np.asarray(bootstrap[name]);finite=finite[np.isfinite(finite)]
            ci=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
            delta=np.asarray(deltas);delta=delta[np.isfinite(delta)]
            scores.append(dict(horizon=h,diagnostic=name,train_cases=len(a),test_cases=len(b),test_extreme_rate=target.mean(),
                AUROC=auc(target,p),AUROC_lo=ci[0],AUROC_hi=ci[1],average_precision=average_precision(target,p),
                Brier=float(np.mean((p-target)**2)),fit_status=fits[f'h{h}_{name}']['status'],
                delta_AUROC_vs_context=auc(target,p)-auc(target,probabilities['context_only']) if name=='context_plus_distance' else np.nan,
                delta_AUROC_lo=np.quantile(delta,.025) if name=='context_plus_distance' and len(delta) else np.nan,
                delta_AUROC_hi=np.quantile(delta,.975) if name=='context_plus_distance' and len(delta) else np.nan,
                delta_Brier_vs_context=float(np.mean((p-target)**2)-np.mean((probabilities['context_only']-target)**2)) if name=='context_plus_distance' else np.nan,
                delta_Brier_lo=np.quantile(brier_deltas,.025) if name=='context_plus_distance' else np.nan,
                delta_Brier_hi=np.quantile(brier_deltas,.975) if name=='context_plus_distance' else np.nan))
            for j,r in enumerate(b.itertuples()):predictions.append(dict(series=r.series,forecast_origin=r.forecast_origin,
                target_end_date=r.target_end_date,horizon=h,diagnostic=name,risk_probability=float(p[j]),extreme=bool(target[j])))
    return pd.DataFrame(scores),pd.DataFrame(predictions),fits


def error_groups(data,bin_count):
    rows=[]
    for scope in ['all','test']:
        selected=data if scope=='all' else data.loc[data.split.eq('test')]
        for h in [1,2,3,4]:
            for distance_bin in range(1,bin_count+1):
                g=selected.loc[selected.horizon.eq(h)&selected.distance_bin.eq(distance_bin)]
                rows.append(dict(scope=scope,horizon=h,distance_bin=distance_bin,cases=len(g),states=g.series.nunique(),
                    MAE=float(g.absolute_error.mean()),normalized_MAE=float(g.normalized_absolute_error.mean()),
                    extreme_rate=float(g.extreme.mean()),median_scaled_distance=float(g.mean_distance_scaled.median())))
    return pd.DataFrame(rows)


def make_plots(out,data,corr,risk,groups,panel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=out/'figures';directory.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(directory/f'{name}.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,h in zip(axes.flat,[1,2,3,4]):
        g=data.loc[data.horizon.eq(h)]
        for phase,color in [('rising','#c94f3e'),('stable','#537ca6'),('declining','#4a9569')]:
            part=g.loc[g.phase.eq(phase)]
            ax.scatter(np.log1p(part.mean_distance_scaled),np.log1p(part.normalized_absolute_error),s=9,alpha=.3,color=color,label=phase,rasterized=True)
        ax.set(xlabel='log1p(mean L1 / (k x past level))',ylabel='log1p(absolute error / recent cases)',title=f'h={h}; all 2500 cases, descriptive');ax.legend(fontsize=8)
    save(fig,'01_distance_error')
    fig,axes=plt.subplots(2,2,figsize=(12,8));test=data.loc[data.split.eq('test')];bins=sorted(data.distance_bin.unique())
    for ax,h in zip(axes.flat,[1,2,3,4]):
        values=[test.loc[test.horizon.eq(h)&test.distance_bin.eq(b),'normalized_absolute_error'].to_numpy() for b in bins]
        ax.boxplot(values,tick_labels=[f'D{b}\nn={len(v)}' for b,v in zip(bins,values)],showfliers=False)
        ax.axhline(1,color='gray',ls='--');ax.set(xlabel='Distance bins fitted on train cases (low to high)',ylabel='Normalized absolute error',title=f'h={h}; test period (fliers hidden)')
    save(fig,'02_error_by_distance_bin')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,error in zip(axes,['absolute_error','normalized_absolute_error']):
        subset=corr.loc[corr.scope.eq('all')&corr.stratum.eq('all')&corr.error.eq(error)]
        matrix=subset.pivot(index='distance',columns='horizon',values='rho').reindex(DISTANCES+[d+'_scaled' for d in DISTANCES])
        img=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
        ax.set(xticks=range(4),xticklabels=[1,2,3,4],yticks=range(len(matrix)),yticklabels=matrix.index,xlabel='Horizon',title='Spearman vs '+error)
        for i in range(len(matrix)):
            for j in range(4):ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
        fig.colorbar(img,ax=ax)
    save(fig,'03_correlations')
    fig,ax=plt.subplots(figsize=(9,5))
    for h,g in groups.loc[groups.scope.eq('test')].groupby('horizon'):ax.plot(g.distance_bin,g.extreme_rate,'o-',label=f'h={h}')
    ax.set(xticks=bins,xlabel='Training-defined distance bin (low to high)',ylabel='Extreme error rate (>1)',title='Held-out period; low distance does not imply guaranteed accuracy');ax.legend()
    save(fig,'04_extreme_error_rates')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,g in risk.groupby('diagnostic'):
        for ax,field in zip(axes,['AUROC','Brier']):ax.plot(g.horizon,g[field],'o-',label=name)
    axes[0].axhline(.5,color='gray',ls='--')
    for ax,field in zip(axes,['AUROC','Brier']):ax.set(xticks=[1,2,3,4],xlabel='Horizon',ylabel=field,title='Held-out risk of sMOA error; frozen diagnostic fits');ax.legend(fontsize=7)
    save(fig,'05_risk_diagnostics')
    failed=data.loc[data.good_match & data.extreme].sort_values('normalized_absolute_error',ascending=False)
    chosen=failed.drop_duplicates('series').head(4)
    fig,axes=plt.subplots(2,2,figsize=(12,8));fig.suptitle('Good distance, large error: worst cases selected after scoring, not representative')
    for ax,case in zip(axes.flat,chosen.itertuples()):
        history=panel.loc[panel.series.eq(case.series)&panel.forecast_origin.eq(case.forecast_origin)].sort_values('target_end_date').tail(12)
        forecast=data.loc[data.series.eq(case.series)&data.forecast_origin.eq(case.forecast_origin)].sort_values('horizon')
        ax.plot(pd.to_datetime(history.target_end_date),history.value,'o-',color='gray',label='As-of input')
        ax.plot(pd.to_datetime(forecast.target_end_date),forecast.actual,'o-',color='black',label='Final actual')
        ax.plot(pd.to_datetime(forecast.target_end_date),forecast.prediction,'o--',color='#c94f3e',label='sMOA')
        ax.axvline(pd.Timestamp(case.forecast_origin),color='gray',ls=':')
        ax.set(title=f'{case.series}, {case.forecast_origin}\nscaled mean distance={case.mean_distance_scaled:.4f}',ylabel='Weekly cases');ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=7)
    for ax in axes.flat[len(chosen):]:ax.set_visible(False)
    save(fig,'06_good_match_failures')
    fig,ax=plt.subplots(figsize=(10,5))
    subset=corr.loc[corr.scope.eq('test')&~corr.stratum.isin(['all','context_adjusted'])]
    matrix=subset.pivot(index='stratum',columns='horizon',values='rho')
    image=ax.imshow(np.ma.masked_invalid(matrix.to_numpy()),vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
    ax.set(xticks=range(4),xticklabels=[1,2,3,4],yticks=range(len(matrix)),yticklabels=matrix.index,xlabel='Horizon',title='Held-out Spearman within origin-known strata (n<20 omitted)')
    for i in range(len(matrix)):
        for j in range(4):
            value=matrix.iloc[i,j]
            if pd.notna(value):ax.text(j,i,f'{value:.2f}',ha='center',va='center')
    fig.colorbar(image,ax=ax);save(fig,'07_conditional_correlations')


def markdown(frame):
    frame=frame.copy()
    for column in frame.select_dtypes('number'):frame[column]=frame[column].map(lambda x:f'{x:,.3f}' if pd.notna(x) else 'NA')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))


def write_report(out,data,corr,risk,groups,metadata):
    main=corr.loc[corr.seed.eq(0)&corr.scope.eq('all')&corr.stratum.eq('all')&corr.distance.eq('mean_distance_scaled')&corr.error.eq('normalized_absolute_error')]
    adjusted=corr.loc[corr.scope.eq('all')&corr.stratum.eq('context_adjusted')]
    good=data.loc[data.good_match].groupby('horizon').agg(cases=('extreme','size'),extreme_rate=('extreme','mean'),normalized_MAE=('normalized_absolute_error','mean')).reset_index()
    good_test=data.loc[data.good_match & data.split.eq('test')].groupby('horizon').agg(cases=('extreme','size'),extreme_instances=('extreme','sum'),extreme_rate=('extreme','mean')).reset_index()
    lines=['# Experiment 5 — Analogue quality vs forecast accuracy','',
        'Primary: paper parameters N=18.387 attempts/family, k=5, m=4.422, GAM, omit bridge, seed0. 50 states ×50 fixed origins ×4 horizons =10.000 forecast instances. No BO, synthetic generation or sMOA forecast change.',
        '',f"Exact replay verified {metadata['verified_forecasts']:,} point forecasts and all stored min/median/max distances. Mean distance was filled by exact top-m replay using the existing memmap bank. Other E2 seeds are secondary median-distance sensitivity only.",
        '', '## Definitions and chronology','',
        '- L1 search uses unchanged GAM history differences. Scale normalization divides the resulting distance by k×mean(last 6 floored raw values); it does not change which analogues are selected.',
        '- Errors normalize by mean(last 4 floored observations), as Experiment 4. Distances/errors use related origin scales: ratio correlations can be induced by scale. Compare raw/scaled metrics, strata, adjusted rank association and held-out diagnostic performance together.',
        '- One observation at each horizon per state/origin. Spearman is reported separately per horizon; pooled horizon analysis uses one mean-error record per case, never 10.000 independent cases. No naive independent-row p-values.',
        f"- Training cases have all targets ≤{CUTOFF}; test origins ≥{TEST_START}. Training labels come only from the first test-origin historical JHU vintage, not future final revisions. The diagnostic is frozen for all later origins; final truth is used to evaluate test risk.",
        f"- First test origin: {metadata['first_test_origin']}; {metadata['train_cases']} training, {metadata['test_cases']} test, {metadata['embargo_cases']} embargo cases. Bin boundaries fitted only to training cases: {metadata['distance_bin_cuts']}.",
        '', '## Does closer matching imply smaller error?','',markdown(main[['horizon','cases','rho','lo','hi']]),
        '', 'Context-adjusted rank association (controls: both past levels, growth, volatility, history length and phase):','',markdown(adjusted[['horizon','rho']]),
        '',f"Scaled-mean Spearman spans {main.rho.min():.3f}..{main.rho.max():.3f}; after context adjustment {adjusted.rho.min():.3f}..{adjusted.rho.max():.3f}. This separates unconditional association from remaining rank association; it does not establish causality or rule out nonlinear signals.",
        '', 'Full raw/scaled minimum/mean/median and error-scale combinations, train/test scopes, strata and secondary seeds are in correlations.csv. Their comparisons are descriptive; no post-hoc best metric is promoted to a selected risk signal.',
        '', '## Can a good match still fail?','',
        f"Good matching is fixed at training-case q20={metadata['good_match_threshold']:.6f} for scaled mean distance; a large error is absolute error/origin scale >1. All matching-small/error-large cases remain in counterexamples.csv; the figure picks four extreme distinct states after scoring.",
        '', 'All cases (descriptive):','',markdown(good),
        '', 'Held-out period only:','',markdown(good_test),
        '', '## Does distance help diagnose risk on a later period?','',markdown(risk[['horizon','diagnostic','train_cases','test_cases','AUROC','average_precision','Brier']]),
        '', 'Increment over origin-context diagnostic, using exactly the same held-out targets (positive delta AUROC / negative delta Brier favors adding distance):','',
        markdown(risk.loc[risk.diagnostic.eq('context_plus_distance'),['horizon','delta_AUROC_vs_context','delta_AUROC_lo','delta_AUROC_hi','delta_Brier_vs_context','delta_Brier_lo','delta_Brier_hi']]),
        '', 'Risk scores are diagnostics for forecast failure, not additional case-count forecasting models. Logistic standardization is fitted on train only; L2 penalty fixed at 1; no test tuning. AP should be compared with test extreme prevalence in risk_scores.csv; Brier measures overall probability accuracy, not calibration alone; AUROC measures ranking.',
        '', 'Small positive/negative incremental scores are horizon-dependent: a distance-only association is not proof of additional value beyond context. Read paired intervals and both ranking/probability metrics. No best horizon or diagnostic is selected for a production alert from this test.',
        '', '## Limits','',
        '- Distance similarity represents past trajectories, not future mechanisms. Novel waves, phase changes and reporting zeros can break that link; observations here do not establish causality.',
        '- State-cluster percentile intervals use 500 resamples retaining origins within state. They do not remove national temporal dependence; one fixed time split is not external validation. Strata with <20 cases or constant inputs have NA correlation.',
        '- The primary mean-distance result uses one training bank. Two other E2 banks support only a median-distance check; no mean distance was invented for them.',
        '- This preserves the existing numerical core, including its floating-point variance filter, to match E2/E4. Python/R numerical parity is not established.',
        '- Quartile/quantile cuts are frozen from early-period cases and may have imbalanced/empty groups on test. Hidden boxplot outlier markers affect display only; all instances remain in scores and scatter.',
        '- Confidence intervals are descriptive, without multiple-comparison adjustment. Conditional associations and one diagnostic family do not prove actionable risk prediction or absence of every possible distance-based signal. Read incremental held-out AUROC/Brier before treating distance as a warning signal.','']
    for name in ['01_distance_error','02_error_by_distance_bin','03_correlations','04_extreme_error_rates','05_risk_diagnostics','06_good_match_failures','07_conditional_correlations']:lines.extend([f'![{name}](figures/{name}.png)',''])
    (out/'report.md').write_text('\n'.join(lines))


def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True);lock=acquire_run_lock(out/'run.lock')
    start=time.perf_counter()
    original=json.loads((args.e2_dir/'protocol.json').read_text())
    for name in ['experiments/forecasting/model.py','experiments/forecasting/neighbors.py','experiments/forecasting/smoothing.py']:
        if file_hash(ROOT/name)!=original['source_sha256'][name]:raise ValueError('E2 numerical core changed')
    if file_hash(args.e2_dir/'contexts.npz')!=json.loads((args.e2_dir/'context_metadata.json').read_text())['contexts_sha256']:raise ValueError('E2 contexts modified')
    bank_meta=args.e2_dir/'packed/seed_0/metadata.json'
    if json.loads((args.cache/'metadata.json').read_text())['library_sha256']!=json.loads(bank_meta.read_text())['source_sha256']:
        raise ValueError('Memmap cache is not the original E2 reference bank')
    inputs=[args.e2_dir/'contexts.npz',args.e2_dir/'context_metadata.json',args.panel_dir/'panel.csv',args.panel_dir/'provenance.json',
        args.e2_dir/'protocol.json',args.library.with_name('metadata.json'),args.cache/'metadata.json',bank_meta]
    for seed in [0,1,2]:
        folder=args.e2_dir/f'groups/seed_{seed}_N18387_k5'
        completion=json.loads((folder/'complete.json').read_text())
        for name in ['forecasts.csv.gz','matches.csv.gz']:
            if file_hash(folder/name)!=completion['sha256'][name]:raise ValueError('Modified E2 forecast/match table')
            inputs.append(folder/name)
    signature=hashlib.sha256(ast.dump(ast.parse(inspect.getsource(prepare_cases)+'\n'+inspect.getsource(replay_distances))).encode()).hexdigest()
    identity=dict(rules=RULES,input_sha256={str(p.resolve()):file_hash(p) for p in inputs},extraction_signature=signature,
        dependencies_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'experiments/error_analysis/conditions.py',ROOT/'experiments/forecasting/neighbors.py',ROOT/'experiments/error_analysis/library.py']})
    proto=out/'protocol.json'
    if proto.exists():
        saved=json.loads(proto.read_text())
        if any(saved[key]!=value for key,value in identity.items()):raise ValueError('Changed inputs/extraction protocol; use a new output directory')
    else:atomic_json(proto,dict(identity,source_sha256=file_hash(Path(__file__)),source_text=Path(__file__).read_text(),packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','matplotlib']}))
    previous=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    atomic_json(out/'progress.json',dict(status='matching',started_utc=datetime.now(timezone.utc).isoformat()))
    panel=pd.read_csv(args.panel_dir/'panel.csv')
    if file_hash(args.panel_dir/'panel.csv')!=json.loads((args.panel_dir/'provenance.json').read_text())['panel_sha256']:raise ValueError('Panel checksum differs')
    with np.load(args.e2_dir/'contexts.npz',allow_pickle=False) as source:contexts={name:source[name] for name in source.files}
    folder=args.e2_dir/'groups/seed_0_N18387_k5'
    forecasts=pd.read_csv(folder/'forecasts.csv.gz',float_precision='round_trip').query('m==4422').copy()
    matches=pd.read_csv(folder/'matches.csv.gz',float_precision='round_trip').query('m==4422').copy()
    if len(forecasts)!=10000 or len(matches)!=2500:raise ValueError('Require full E2 reference cohort')
    if (out/'cases.csv').exists() and previous.get('cases_sha256')==file_hash(out/'cases.csv'):
        cases=pd.read_csv(out/'cases.csv',float_precision='round_trip')
    else:
        cases=prepare_cases(panel,contexts)
        if not (args.cache/'metadata.json').exists():raise ValueError('Existing paper memmap cache required; do not regenerate a new bank')
        x,y=embeddings(args.library,args.cache,k=5,horizon=5)
        cases=replay_distances(cases,contexts,x,y[:,1:],matches,forecasts)
        del x,y
        for field in DISTANCES:cases[field+'_scaled']=cases[field]/(5*cases.level_scale)
        cases['split']=np.select([
            (pd.to_datetime(cases.forecast_origin)+pd.Timedelta(weeks=4)).le(pd.Timestamp(CUTOFF)),
            cases.forecast_origin.ge(TEST_START)],['train','test'],default='embargo')
        cases['distance_bin'],cuts=distance_bins(cases.loc[cases.split.eq('train'),'mean_distance_scaled'],cases.mean_distance_scaled)
        threshold=float(cases.loc[cases.split.eq('train'),'mean_distance_scaled'].quantile(.2))
        cases['good_match']=cases.mean_distance_scaled.le(threshold)
        cases.to_csv(out/'cases.csv',index=False)
    training=cases.loc[cases.split.eq('train')]
    _,cuts=distance_bins(training.mean_distance_scaled,cases.mean_distance_scaled)
    threshold=float(training.mean_distance_scaled.quantile(.2))
    data=join_instances(cases,forecasts);data.to_csv(out/'instances.csv.gz',index=False)
    atomic_json(out/'progress.json',dict(status='analyzing',cases_sha256=file_hash(out/'cases.csv'),verified_forecasts=10000))
    corr=correlations(data)
    # Reuse other seeds' recorded distances; do not claim missing means are known.
    for seed in [1,2]:
        folder=args.e2_dir/f'groups/seed_{seed}_N18387_k5'
        other_matches=pd.read_csv(folder/'matches.csv.gz').query('m==4422')[KEYS+['median_distance']]
        other=pd.read_csv(folder/'forecasts.csv.gz').query('m==4422')
        other=other.merge(cases[KEYS+['level_scale','split']],on=KEYS,validate='many_to_one').merge(other_matches,on=KEYS,validate='many_to_one')
        other['median_distance_scaled']=other.median_distance/(5*other.level_scale)
        other['normalized_absolute_error']=(other.prediction-other.actual).abs()/other.normalization
        extra=[]
        for h,g in other.groupby('horizon'):
            extra.append(dict(seed=seed,scope='all',stratum='secondary_seed',horizon=h,distance='median_distance_scaled',error='normalized_absolute_error',cases=len(g),states=g.series.nunique(),rho=spearman(g.median_distance_scaled,g.normalized_absolute_error),lo=np.nan,hi=np.nan))
        corr=pd.concat([corr,pd.DataFrame(extra)],ignore_index=True)
    scores,predictions,fits=risk_diagnostics(data,panel)
    groups=error_groups(data,len(cuts)+1)
    corr.to_csv(out/'correlations.csv',index=False);scores.to_csv(out/'risk_scores.csv',index=False)
    predictions.to_csv(out/'risk_predictions.csv.gz',index=False);groups.to_csv(out/'grouped_errors.csv',index=False)
    data.loc[data.good_match & data.extreme].sort_values('normalized_absolute_error',ascending=False).to_csv(out/'counterexamples.csv',index=False)
    metadata=dict(verified_forecasts=10000,cases=2500,train_cases=int(cases.split.eq('train').sum()),test_cases=int(cases.split.eq('test').sum()),
        embargo_cases=int(cases.split.eq('embargo').sum()),first_test_origin=cases.loc[cases.split.eq('test'),'forecast_origin'].min(),
        good_match_threshold=threshold,distance_bin_cuts=cuts.tolist())
    make_plots(out,data,corr,scores,groups,panel);write_report(out,data,corr,scores,groups,metadata)
    atomic_json(out/'progress.json',dict(status='complete',**metadata,cases_sha256=file_hash(out/'cases.csv'),
        analysis_source_sha256=file_hash(Path(__file__)),risk_fits=fits,wall_seconds=time.perf_counter()-start,
        completed_utc=datetime.now(timezone.utc).isoformat(),artifacts_sha256={str(p.relative_to(out)):file_hash(p) for p in [*out.glob('*.csv'),*out.glob('*.csv.gz'),out/'report.md',*out.joinpath('figures').glob('*')]}))
    print('Complete:',out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/experiment_5')
    parser.add_argument('--e2-dir',type=Path,default=ROOT/'outputs/experiment_2')
    parser.add_argument('--panel-dir',type=Path,default=ROOT/'outputs/epidemic_panel')
    parser.add_argument('--library',type=Path,default=ROOT/'outputs/reproduction_full/01_synthetic/curves.csv')
    parser.add_argument('--cache',type=Path,default=ROOT/'outputs/paper_embedding_k5_h4_omit')
    parser.add_argument('--execute-notebook',action='store_true')
    args=parser.parse_args()
    if args.execute_notebook:
        from notebooks import execute_notebook
        execute_notebook(Path(__file__).with_suffix('.ipynb'), working_dir=ROOT, timeout=600)
    else:run(args)
