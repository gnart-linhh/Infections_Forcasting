"""Stratified metrics, diagnostic figures and data-driven interpretation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from conditions import summarize

SMOA='Original_sMOA_Python'
ORDERS={'phase':['rising','stable','declining'], 'case_level':['low','medium','high'],
        'volatility_level':['low','medium','high'],'history_stage':['early','later'],'horizon':[1,2,3,4]}


def markdown(frame):
    values=frame.copy()
    for col in values.select_dtypes('number'):
        values[col]=values[col].map(lambda v:f'{v:,.3f}' if pd.notna(v) else 'NA')
    return '| '+' | '.join(values.columns)+' |\n| '+' | '.join(['---']*len(values.columns))+' |\n'+ '\n'.join('| '+' | '.join(map(str,row))+' |' for row in values.itertuples(index=False,name=None))


def case_quantiles(data):
    """Post-hoc examples at fixed case-error quantiles, using all four horizons."""
    cases=data.groupby(['series','forecast_origin'],as_index=False).normalized_absolute_error.mean()
    cases=cases.sort_values(['normalized_absolute_error','series','forecast_origin']).reset_index(drop=True)
    rows=[]
    for q in [.10,.50,.90,.99]:
        row=cases.iloc[int(round(q*(len(cases)-1)))].to_dict()
        rows.append(dict(error_quantile=q,**row))
    return pd.DataFrame(rows)


def report(data,status,panel,out,config,truths=None):
    if set(data.model)!={SMOA} or config['rules']['models']!=[SMOA]:
        raise ValueError('Experiment 4 accepts only sMOA forecasts')
    out=Path(out)
    figures=out/'figures'
    figures.mkdir(exist_ok=True)
    tables={'overall':summarize(data,[])}
    for dim in ORDERS:
        tables[dim]=summarize(data,[dim])
    for dims in [('phase','horizon'),('phase','case_level'),('history_stage','horizon')]:
        tables['_'.join(dims)]=summarize(data,list(dims))
    for name,table in tables.items(): table.to_csv(out/f'{name}.csv',index=False)
    # Cases, rather than four correlated horizons, define the extreme-case catalogue.
    smoa=data.loc[data.model==SMOA].copy()
    extreme=smoa.sort_values('normalized_absolute_error',ascending=False).drop_duplicates(['series','forecast_origin']).head(30)
    extreme.to_csv(out/'extreme_cases.csv',index=False)
    quantiles=case_quantiles(smoa)
    quantiles.to_csv(out/'error_quantile_cases.csv',index=False)
    data.loc[data.extreme].groupby(['model','phase','case_level','volatility_level','history_stage'],observed=True).size().rename('extreme_count').reset_index().to_csv(out/'extreme_profiles.csv',index=False)
    case_features=status.drop_duplicates(['series','forecast_origin'])
    for dim in ['phase','case_level','volatility_level','history_stage']:
        case_features.groupby(dim).agg(cases=('series','size'),states=('series','nunique'),origins=('forecast_origin','nunique')).reindex(ORDERS[dim],fill_value=0).to_csv(out/f'group_counts_{dim}.csv')

    def save(fig,name):
        fig.tight_layout()
        for suffix in ['png','svg']: fig.savefig(figures/f'{name}.{suffix}',dpi=150)
        plt.close(fig)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    phase=tables['phase']
    for ax,metric in zip(axes,['MAE','normalized_MAE']):
        subset=phase.set_index('phase').reindex(ORDERS['phase'])
        xpos=np.arange(3)
        ax.bar(xpos,subset[metric],width=.6,color=['#c94f3e','#537ca6','#4a9569'])
        if metric+'_lo' in subset:
            vals=subset[metric].to_numpy()
            err=np.maximum(0,np.array([vals-subset[metric+'_lo'].to_numpy(),subset[metric+'_hi'].to_numpy()-vals]))
            ax.errorbar(xpos,vals,yerr=err,fmt='none',color='black',capsize=3)
        ax.set(xticks=np.arange(3),xticklabels=ORDERS['phase'],ylabel=metric,title=f'sMOA {metric} by origin-known phase')
    save(fig,'phase_mae')

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['absolute_error','normalized_absolute_error']):
        groups=[smoa.loc[smoa.case_level==g,metric].to_numpy() for g in ORDERS['case_level']]
        ax.boxplot(groups,tick_labels=ORDERS['case_level'],showfliers=False)
        ax.set(xlabel='Case-count level at origin',ylabel=metric,title='sMOA; outliers omitted from display only')
    save(fig,'case_count_boxplot')

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['error','normalized_error']):
        groups=[smoa.loc[smoa.horizon==h,metric].to_numpy() for h in range(1,5)]
        ax.boxplot(groups,tick_labels=['1','2','3','4'],showfliers=False)
        ax.axhline(0,color='black',lw=.8)
        ax.set(xlabel='Weeks ahead',ylabel=metric,title='Signed error; negative = underprediction')
    save(fig,'horizon_distribution')

    fig,ax=plt.subplots(figsize=(8,5))
    for phase,color in [('rising','#c94f3e'),('stable','#537ca6'),('declining','#4a9569')]:
        group=smoa.loc[smoa.phase==phase]
        ax.scatter(group.volatility,group.normalized_absolute_error,s=9,alpha=.25,label=phase,color=color,rasterized=True)
    ax.set(xlabel='SD of recent weekly log changes (known at origin)',ylabel='Absolute error / recent case level',yscale='symlog',title='sMOA volatility vs forecast error (all instances)')
    ax.legend()
    save(fig,'volatility_scatter')

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,['MAE','normalized_bias','extreme_rate']):
        group=tables['horizon'].sort_values('horizon')
        ax.plot(group.horizon,group[metric],marker='o')
        ax.set(xlabel='Weeks ahead',ylabel=metric,xticks=range(1,5),title=f'sMOA {metric}')
    save(fig,'horizon_metrics')

    def trajectories(catalog,name,title):
        # Both catalogues use outcomes for illustration only, never for features.
        fig,axes=plt.subplots(2,2,figsize=(12,8),squeeze=False)
        fig.suptitle(title,fontsize=12)
        for ax,case in zip(axes.flat,catalog.head(4).itertuples()):
            history=panel.loc[(panel.series==case.series)&(panel.forecast_origin==case.forecast_origin)].sort_values('target_end_date').tail(12)
            predicted=smoa.loc[(smoa.series==case.series)&(smoa.forecast_origin==case.forecast_origin)].sort_values('horizon')
            if truths is not None:
                final=truths[case.series]
                final=final.loc[(final.index>=history.target_end_date.min())&(final.index<=predicted.target_end_date.max())]
                ax.plot(pd.to_datetime(final.index),final.to_numpy(),'o-',label='Final actual (retrospective)',color='black',ms=3,lw=1)
            else:
                ax.plot(pd.to_datetime(predicted.target_end_date),predicted.actual,'o-',label='Final actual',color='black')
            ax.plot(pd.to_datetime(history.target_end_date),history.value,'o-',label='As-of history',color='gray',ms=4)
            ax.plot(pd.to_datetime(predicted.target_end_date),predicted.prediction,'o--',label='sMOA',color='#c94f3e')
            ax.axvline(pd.Timestamp(case.forecast_origin),ls=':',color='gray')
            suffix=f'error percentile {case.error_quantile:.0%}' if hasattr(case,'error_quantile') else 'worst-case selection'
            ax.set_title(f'{case.series}, {case.forecast_origin}: {suffix}',fontsize=10)
            ax.tick_params(axis='x',rotation=30);ax.set_ylabel('Weekly cases')
            ax.legend(fontsize=7)
        save(fig,name)
    trajectories(extreme,'extreme_trajectories','Four largest relative-error cases — selected after scoring; not typical forecasts')
    trajectories(quantiles,'error_quantile_trajectories','Case-error quantiles 10/50/90/99% — post-hoc examples; average over 4 horizons')

    fig,axes=plt.subplots(2,2,figsize=(13,8))
    # State list and horizons are fixed, independent of model performance.
    fixed_states=['California','Texas','New York','Florida']
    for ax,state in zip(axes.flat,fixed_states):
        subset=smoa.loc[smoa.series==state]
        if subset.empty:ax.set_visible(False);continue
        if truths is not None:
            final=truths[state]
            final=final.loc[(final.index>=subset.target_end_date.min())&(final.index<=subset.target_end_date.max())]
            ax.plot(pd.to_datetime(final.index),final.to_numpy(),color='black',lw=1,label='Final actual (weekly)')
        else:
            final=subset[['target_end_date','actual']].drop_duplicates().sort_values('target_end_date')
            ax.plot(pd.to_datetime(final.target_end_date),final.actual,color='black',lw=1,label='Final actual')
        for horizon,color in [(1,'#537ca6'),(4,'#c94f3e')]:
            group=subset.loc[subset.horizon==horizon].sort_values('target_end_date')
            ax.plot(pd.to_datetime(group.target_end_date),group.prediction,'o--',ms=3,lw=.8,color=color,label=f'sMOA {horizon}-week ahead')
        ax.set(title=state,ylabel='Weekly cases');ax.legend(fontsize=7);ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Fixed states, all scheduled origins; predictions plotted at target dates')
    save(fig,'forecast_overview')

    overall=tables['overall']
    main=tables['phase'].loc[tables['phase'].model==SMOA].set_index('phase')
    horizons=tables['horizon'].loc[tables['horizon'].model==SMOA].set_index('horizon')
    messages=[]
    if {'rising','stable'}.issubset(main.index):
        rising,stable=main.loc['rising'],main.loc['stable']
        messages.append(f"Rising vs stable: MAE {rising.MAE:,.2f} vs {stable.MAE:,.2f}; normalized MAE {rising.normalized_MAE:.3f} vs {stable.normalized_MAE:.3f}. Compare both because raw case scales differ.")
        direction='underprediction' if rising.bias<0 else 'overprediction' if rising.bias>0 else 'zero mean bias'
        messages.append(f"Rising phase: bias {rising.bias:,.2f} cases, normalized bias {rising.normalized_bias:.3f}, underprediction in {rising.underprediction_rate:.1%} of instances. The mean signed error indicates {direction} in this group; individual outbreaks can differ.")
    if {1,4}.issubset(horizons.index):
        h1,h4=horizons.loc[1],horizons.loc[4]
        ratio=h4.MAE/h1.MAE if h1.MAE else float('nan')
        messages.append(f"Horizon 1 -> 4: MAE {h1.MAE:,.2f} -> {h4.MAE:,.2f} ({ratio:.2f}x); extreme rate {h1.extreme_rate:.1%} -> {h4.extreme_rate:.1%}.")
    all_extreme=smoa.loc[smoa.extreme]
    if len(all_extreme):
        for dim in ['phase','case_level','volatility_level','history_stage']:
            common=all_extreme[dim].value_counts().idxmax()
            rate=all_extreme[dim].eq(common).mean()
            conditional=smoa.loc[smoa[dim]==common,'extreme'].mean()
            messages.append(f"Extreme errors: {rate:.1%} occur in {dim}={common}; that group's conditional extreme rate is {conditional:.1%}. Composition alone does not establish higher risk.")
    else: messages.append('No extreme errors under the preregistered >100% threshold.')
    text=['# Experiment 4 — Forecast error across epidemic conditions','',
          f"Panel: {'complete' if config['panel_complete'] else 'INCOMPLETE'}; {status.series.nunique()} states, {status.forecast_origin.nunique()} origins; {len(status)} forecast cases; {sum(status.status!='ok')} failed cases.",
          '', f"sMOA: N=18,387 attempts/family; k=5; m=4,422; bridge={config['rules']['smoa']['bridge_mode']}; full Python synthetic library; GAM/GCV; exact L1 matching. These are paper parameters, not the BO winner.",
          '',f"Synthetic: {config['library_attempted']:,} attempts; {config['library_accepted']:,} accepted curves. Four horizons share the same origin history. This experiment analyzes only sMOA errors across epidemic conditions.",
          '', '## Overall', '', markdown(overall[['model','n','MAE','RMSE','normalized_MAE','bias','extreme_rate']]),
          '', 'Normalized errors and extreme rates use recent origin-known cases as denominator, not future actuals. A rate of 0.10 means 10%. WIS is not computed in this point-error experiment; existing UQ/WIS experiments are separate.',
          '', '## Findings from this run','']
    text += ['- '+message for message in messages]
    text += ['', '## Strata and horizon', '',markdown(tables['phase'][['model','phase','n','states','origins','MAE','normalized_MAE','bias','extreme_rate']]),
             '',markdown(tables['horizon'][['model','horizon','n','MAE','RMSE','normalized_bias','extreme_rate']]),
             '', '![MAE by phase](figures/phase_mae.png)', '', '![Case level](figures/case_count_boxplot.png)',
             '', '![Horizon distributions](figures/horizon_distribution.png)', '', '![Volatility](figures/volatility_scatter.png)',
             '', '![Horizon metrics](figures/horizon_metrics.png)',
             '', '## Forecast overview and selection of examples','',
             'Overview: four fixed states, all scheduled origins, horizons 1/4; points are plotted at target dates. The later origin schedule is every four weeks, so prediction lines connect sparse points. Final weekly data are retrospective reference only.',
             '', '![Overview](figures/forecast_overview.png)',
             '', 'The next examples are selected at fixed percentiles of case mean normalized absolute error across four horizons. They show several error levels, not a random sample or a claim of typical accuracy in every state.',
             '', '![Error quantiles](figures/error_quantile_trajectories.png)',
             '', 'The original extreme plot deliberately selects the four largest relative-error cases out of all forecast cases. It diagnoses failures and must not be presented as representative overall performance. Historical final values are shown alongside actual as-of inputs to reveal revisions; only as-of inputs enter forecasts.',
             '', '![Extreme cases](figures/extreme_trajectories.png)',
             '', '## Rules fixed before scoring', '', *['- '+key+': '+str(value) for key,value in config['rules'].items()],
             '', '## Interpretation limits', '',
             '- This is an added diagnostic experiment, not a figure or accuracy claim reproduced from the original paper. No new BO is performed.',
             '- Historical inputs are JHU snapshots at the LANL Sunday cutoff, before Monday issuance. Final raw truth is used only for labels. Negative weekly revisions remain in evaluation; model history is floored at one with revision flags recorded.',
             '- Absolute case bins partly reflect state population. Phase, volatility, case level and horizon are correlated; joint CSV tables support checking this confounding. Findings are descriptive.',
             '- Early means at most 26 observed weeks, not low case counts. The calendar intentionally samples the early period more densely: pooled averages are sample averages, not uniformly weighted pandemic-year estimates.',
             '- Confidence intervals resample whole states, retaining origins/horizons within state. They do not remove nationwide temporal dependence. Empty groups are listed with zero counts; one-state groups have no interval.',
             '- Synthetic RNG and Python GAM numerical details differ from R/mgcv; exact numerical parity has not been established.',
             '- Boxplots omit outlier markers for legibility only. Metrics, scatter data, extreme catalogues and CSV exports retain every valid instance.',
             '- Runtime is recorded per forecast case; the concurrently running BO can affect timings. Setup is separate in progress.json.',
             '', 'Files: instances.csv (auditable per-target errors/features), case_status.csv, overall.csv, stratified/joint CSVs (state-cluster confidence intervals), group_counts_*.csv, extreme_cases.csv and extreme_profiles.csv.', '']
    (out/'report.md').write_text('\n'.join(text))
