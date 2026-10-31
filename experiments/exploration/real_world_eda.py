"""Raw weekly case distributions, epidemic changes and vintage/final discrepancies."""
from pathlib import Path
import numpy as np
import pandas as pd
from synthetic_eda import peaks

REGIONS={1:'Northeast',2:'Midwest',3:'South',4:'West'}
RULES=dict(sharp_change='abs(delta)>=1000 AND abs(delta/previous)>=0.50; previous>0,current>=0; consecutive weeks only',
    count_levels='low: 0..100 cases/week; high: >=10000; negative values classified as revisions',
    rate_levels='low: 0..1 /100k/week; high: >=100 /100k/week, fixed 2019 population',
    sustained_period='at least 3 consecutive weeks (all runs retained in CSV)',
    peaks='median-3 smoothing; prominence 10% of state range; minimum distance 8 weeks; descriptive full-series detection',
    revision='final minus as_of; positive means upward discrepancy',
    symmetric_revision='2*abs(final-asof)/(abs(final)+abs(asof)); both zero ->0',
    revision_units='origin-week pairs once per state/target; all-history pairs by lag; balanced state/target cohort over lags 0..12')


def load_truth(manifest,population):
    manifest=Path(manifest);entries=pd.read_csv(manifest)
    if entries.series.duplicated().any():raise ValueError('Duplicate states in manifest')
    census=pd.read_csv(population)
    census=census.loc[census.NAME.isin(entries.series),['NAME','REGION','POPESTIMATE2019']].rename(
        columns={'NAME':'series','POPESTIMATE2019':'population_2019'})
    if census.series.duplicated().any() or set(census.series)!=set(entries.series) or (census.population_2019<=0).any():
        raise ValueError('Population/region table must match each state exactly once')
    census['REGION']=pd.to_numeric(census.REGION,errors='raise')
    census['region']=census.REGION.map(REGIONS)
    if census.region.isna().any():raise ValueError('Unknown Census region code')
    frames=[]
    for entry in entries.itertuples():
        frame=pd.read_csv(manifest.parent/entry.truth)
        frame['series']=entry.series;frame['target_end_date']=pd.to_datetime(frame.target_end_date)
        if not np.isfinite(frame.value).all() or frame.target_end_date.duplicated().any():
            raise ValueError(f'Invalid final truth: {entry.series}')
        frame=frame.sort_values('target_end_date')
        if not frame.target_end_date.diff().dropna().eq(pd.Timedelta(weeks=1)).all():
            raise ValueError(f'Missing weekly truth: {entry.series}')
        frames.append(frame)
    all_truth=pd.concat(frames,ignore_index=True).merge(census,on='series',validate='many_to_one')
    all_truth['cases_per_100k']=all_truth.value/all_truth.population_2019*100000
    ranges=all_truth.groupby('series').target_end_date.agg(['min','max'])
    start,end=ranges['min'].max(),ranges['max'].min()
    common=all_truth.loc[all_truth.target_end_date.between(start,end)].copy()
    return all_truth,common,census,entries


def distributions(frame,groupby):
    rows=[]
    for key,g in frame.groupby(groupby,sort=True):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(groupby,key),weeks=len(g),states=g.series.nunique(),zeros=int(g.value.eq(0).sum()),negatives=int(g.value.lt(0).sum()))
        for field in ['value','cases_per_100k']:
            row.update({field+'_mean':float(g[field].mean()),field+'_median':float(g[field].median()),
                        field+'_p05':float(g[field].quantile(.05)),field+'_p95':float(g[field].quantile(.95)),
                        field+'_min':float(g[field].min()),field+'_max':float(g[field].max())})
        peak=g.loc[g.value.idxmax()]
        row['peak_date']=peak.target_end_date.date().isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def weekly_changes(frame):
    result=frame.sort_values(['series','target_end_date']).copy()
    result['previous_value']=result.groupby('series').value.shift()
    result['previous_date']=result.groupby('series').target_end_date.shift()
    consecutive=(result.target_end_date-result.previous_date).eq(pd.Timedelta(weeks=1))
    result['change']=(result.value-result.previous_value).where(consecutive)
    result['change_per_100k']=result.change/result.population_2019*100000
    valid=consecutive & result.previous_value.gt(0) & result.value.ge(0)
    result['relative_change']=(result.change/result.previous_value.where(result.previous_value.gt(0))).where(valid)
    result['from_zero']=consecutive & result.previous_value.eq(0) & result.value.gt(0)
    result['sharp_change']=valid & result.change.abs().ge(1000) & result.relative_change.abs().ge(.50)
    result['direction']=np.select([result.value.lt(0),result.change.gt(0),result.change.lt(0)],['negative_revision','rising','declining'],default='stable')
    return result


def periods(frame):
    rows=[]
    for state,g in frame.groupby('series'):
        g=g.sort_values('target_end_date')
        for field,low,high in [('value',100,10000),('cases_per_100k',1,100)]:
            for level,mask in [('low',g[field].between(0,low)),('high',g[field].ge(high))]:
                selected=g.loc[mask].copy()
                if selected.empty:continue
                run=selected.target_end_date.diff().ne(pd.Timedelta(weeks=1)).cumsum()
                for _,part in selected.groupby(run):
                    rows.append(dict(series=state,region=part.region.iloc[0],unit=field,level=level,
                        start=part.target_end_date.min().date().isoformat(),end=part.target_end_date.max().date().isoformat(),
                        weeks=len(part),sustained=len(part)>=3,min_value=float(part[field].min()),
                        median_value=float(part[field].median()),max_value=float(part[field].max())))
    return pd.DataFrame(rows)


def real_peaks(frame):
    rows=[]
    for state,g in frame.groupby('series'):
        g=g.sort_values('target_end_date').reset_index(drop=True)
        for index in peaks(g.value.to_numpy(),distance=8):
            r=g.iloc[index]
            rows.append(dict(series=state,region=r.region,target_end_date=r.target_end_date.date().isoformat(),
                             value=float(r.value),cases_per_100k=float(r.cases_per_100k)))
    return pd.DataFrame(rows)


def revision_summary(group):
    denominator=(np.abs(group.final_value)+np.abs(group.asof_value)).to_numpy(dtype=float)
    sym=np.divide(2*group.revision.abs().to_numpy(),denominator,out=np.zeros(len(group)),where=denominator!=0)
    return dict(pairs=len(group),states=group.series.nunique(),target_weeks=group.target_end_date.nunique(),
        signed_revision_mean=float(group.revision.mean()),absolute_revision_mean=float(group.revision.abs().mean()),
        absolute_revision_median=float(group.revision.abs().median()),absolute_revision_p90=float(group.revision.abs().quantile(.9)),
        symmetric_relative_revision_mean=float(np.mean(sym)),changed_fraction=float(group.revision.ne(0).mean()),
        upward_fraction=float(group.revision.gt(0).mean()),downward_fraction=float(group.revision.lt(0).mean()))


def revision_tables(pairs,max_lag=12):
    latest=pairs.loc[pairs.lag_weeks==0].copy()
    by_lag=pd.DataFrame([dict(lag_weeks=int(lag),**revision_summary(g)) for lag,g in pairs.groupby('lag_weeks')])
    by_state=pd.DataFrame([dict(series=s,**revision_summary(g)) for s,g in latest.groupby('series')])
    selected=pairs.loc[pairs.lag_weeks.between(0,max_lag)]
    keys=['series','target_end_date']
    complete=selected.groupby(keys).lag_weeks.nunique()
    cohort=complete.loc[complete==max_lag+1].reset_index()[keys]
    balanced=selected.merge(cohort,on=keys,validate='many_to_one')
    balanced_lag=pd.DataFrame([dict(lag_weeks=int(lag),**revision_summary(g)) for lag,g in balanced.groupby('lag_weeks')])
    return latest,by_lag,by_state,balanced,balanced_lag


def compare_vintages(archive,truth):
    archive=Path(archive);manifest=pd.read_csv(archive/'manifest.csv')
    if set(manifest.series)!=set(truth.series):raise ValueError('Archive/final states differ')
    records=[]
    for row in manifest.itertuples():
        frame=pd.read_csv(archive/row.as_of_truth)
        frame['series']=row.series
        for field in ['as_of','target_end_date']:frame[field]=pd.to_datetime(frame[field])
        if frame.duplicated(['as_of','target_end_date']).any() or (frame.target_end_date>frame.as_of).any():
            raise ValueError('Invalid or duplicate vintage records')
        records.append(frame)
    values=pd.concat(records,ignore_index=True).rename(columns={'value':'asof_value'})
    if not np.isfinite(values.asof_value).all():raise ValueError('Nonfinite vintage values')
    final=truth[['series','target_end_date','value','population_2019','region']].rename(columns={'value':'final_value'})
    joined=values.merge(final,on=['series','target_end_date'],how='left',validate='many_to_one',indicator=True)
    if not joined._merge.eq('both').all():raise ValueError('Missing final labels for historical observations')
    joined=joined.drop(columns='_merge')
    age=(joined.as_of-joined.target_end_date).dt.days
    if (age%7).ne(0).any():raise ValueError('Vintage ages must be whole weeks')
    joined['lag_weeks']=age//7
    joined['revision']=joined.final_value-joined.asof_value
    joined['absolute_revision']=joined.revision.abs()
    return joined
