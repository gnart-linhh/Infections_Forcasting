"""Full synthetic and real-world EDA, including matched historical/final case data."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'infections-eda-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
import numpy as np
import pandas as pd
import scipy

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from runtime import acquire_run_lock,atomic_json,file_hash
from synthetic_eda import FAMILIES,PEAK_RULE,scan_library
from real_world_eda import RULES,load_truth,distributions,weekly_changes,periods,real_peaks,compare_vintages,revision_tables,revision_summary

COLORS=['#2563a6','#df8430','#309077','#9957a5']
LABELS={'sir_rollercoaster':'Rollercoaster','sir_rollercoaster_wiggle':'Wiggle','seasonal':'Seasonal'}


def markdown(frame):
    frame=frame.copy()
    for col in frame.select_dtypes('number'):
        frame[col]=frame[col].map(lambda v:f'{v:,.3f}' if pd.notna(v) else 'NA')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))


def figure_writer(directory):
    directory.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(directory/f'{name}.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig)
    return save


def synthetic_plots(curves,examples,out):
    save=figure_writer(out)
    fig,axes=plt.subplots(4,2,figsize=(12,13))
    for i,(family,cadence) in enumerate([(f,'weekly') for f in FAMILIES]+[('seasonal','monthly')]):
        for j,scale in enumerate(['counts','proportion']):
            ax=axes[i,j];sample=examples.loc[(examples.curve_type==family)&(examples.scale==scale)&(examples.time_cadence==cadence)]
            for n,(curve,g) in enumerate(sample.groupby('curve_id',sort=False)):
                if n>=6:break
                ax.plot(g.step,g.value/g.value.max(),alpha=.8,ls='--' if g.time_cadence.iloc[0]=='monthly' else '-',
                        label=f'{curve}, {g.time_cadence.iloc[0]}')
            ax.set(title=f'{LABELS[family]} / {cadence} / {scale}',xlabel=f'Original {cadence} steps',ylabel='Value / own peak')
            ax.legend(fontsize=7,ncol=2)
    save(fig,'01_shapes')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for i,cadence in enumerate(['weekly','monthly']):
        for j,field in enumerate(['observations','epidemic_peaks','peaks_per_100_steps']):
            ax=axes[i,j];labels=[];groups=[]
            for family in FAMILIES:
                group=curves.loc[(curves.curve_type==family)&(curves.time_cadence==cadence),field]
                if len(group):groups.append(group.to_numpy());labels.append(LABELS[family])
            if groups:ax.boxplot(groups,tick_labels=labels,showfliers=False)
            ax.set(title=f'{cadence}: {field}',ylabel=field)
    save(fig,'02_lengths_and_peaks')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,cadence in enumerate(['weekly','monthly']):
        for j,scale in enumerate(['counts','proportion']):
            ax=axes[i,j];subset=curves.loc[(curves.scale==scale)&(curves.time_cadence==cadence)]
            bins=np.histogram_bin_edges(np.log10(subset.max_value),bins=35)
            for family,color in zip(FAMILIES,COLORS):
                group=subset.loc[subset.curve_type==family,'max_value']
                if len(group):ax.hist(np.log10(group),bins=bins,density=True,histtype='step',lw=1.5,label=LABELS[family],color=color)
            ax.set(title=f'Peak magnitude / {cadence} / {scale}',xlabel='log10(curve maximum)',ylabel='Density (each family normalized)');ax.legend(fontsize=8)
    save(fig,'03_scale_distributions')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,cadence in enumerate(['weekly','monthly']):
        for j,scale in enumerate(['counts','proportion']):
            ax=axes[i,j]
            for family,color in zip(FAMILIES,COLORS):
                vals=np.sort(curves.loc[(curves.curve_type==family)&(curves.time_cadence==cadence)&(curves.scale==scale),'mean_absolute_change_over_mean'].to_numpy())
                if len(vals):ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=LABELS[family],color=color)
            ax.set(title=f'{cadence} / {scale}',xlabel='Mean |step change| / curve mean',ylabel='ECDF',xscale='symlog');ax.legend(fontsize=8)
    save(fig,'04_volatility')


def real_plots(common,states,regions,changes,latest,balanced_lag,pairs,out):
    save=figure_writer(out)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    transform=np.sign(common.value)*np.log10(1+common.value.abs())
    axes[0].hist(transform,bins=60,color=COLORS[0]);axes[0].set(xlabel='signed log10(1 + |weekly cases|)',ylabel='State-weeks',title='Raw weekly counts: zeros/negatives retained')
    rates=np.sort(common.cases_per_100k)
    axes[1].plot(rates,np.arange(1,len(rates)+1)/len(rates));axes[1].set(xscale='symlog',xlabel='Weekly cases /100k (2019 population)',ylabel='ECDF',title='Population-scaled distribution')
    save(fig,'01_case_distribution')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,state in zip(axes.flat,['California','Texas','New York','Florida']):
        group=changes.loc[changes.series==state]
        ax.plot(group.target_end_date,group.value,color='#253950',lw=1)
        for direction,color in [('rising','#c94f3e'),('declining','#2563a6')]:
            event=group.loc[group.sharp_change & group.direction.eq(direction)]
            ax.scatter(event.target_end_date,event.value,s=15,color=color,label=direction)
        ax.set(title=state,ylabel='Raw weekly cases');ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=8)
    save(fig,'02_sharp_changes')
    fig,ax=plt.subplots(figsize=(12,4.5))
    for (region,g),color in zip(regions.groupby('region'),COLORS):
        ax.plot(g.target_end_date,g.cases_per_100k,color=color,label=region)
    ax.set(title='Census regions: sum(cases) / sum(2019 population)',ylabel='Weekly cases /100k',xlabel='Week ending');ax.legend(ncol=4)
    save(fig,'03_regions')
    order=states.sort_values('cases_per_100k_median').series.tolist()
    fig,ax=plt.subplots(figsize=(11,14))
    ax.boxplot([common.loc[common.series==s,'cases_per_100k'] for s in order],orientation='horizontal',tick_labels=order,showfliers=False)
    ax.set(xscale='symlog',xlabel='Weekly cases /100k (fixed 2019 population)',title='States on the same calendar; outlier markers omitted only');ax.tick_params(axis='y',labelsize=8)
    save(fig,'04_state_distributions')
    matrix=common.pivot(index='series',columns='target_end_date',values='cases_per_100k')
    order=common[['series','region']].drop_duplicates().sort_values(['region','series']).series
    matrix=matrix.loc[order]
    levels=np.select([matrix.to_numpy()<0,matrix.to_numpy()<=1,matrix.to_numpy()>=100],[-1,0,2],default=1)
    fig,ax=plt.subplots(figsize=(14,12))
    img=ax.imshow(levels,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#a43a58','#eeeeee','#9cc6dd','#204f7a']),norm=BoundaryNorm([-1.5,-.5,.5,1.5,2.5],4))
    ax.set_yticks(np.arange(len(matrix)),matrix.index,fontsize=7)
    ticks=np.linspace(0,len(matrix.columns)-1,8).astype(int);ax.set_xticks(ticks,[matrix.columns[i].strftime('%Y-%m-%d') for i in ticks],rotation=25,ha='right')
    bar=fig.colorbar(img,ax=ax,ticks=[-1,0,1,2],fraction=.02,pad=.02);bar.ax.set_yticklabels(['Negative','Low: <=1','Middle','High: >=100'])
    ax.set(title='Weekly case-rate regimes (/100k); states grouped by region',xlabel='Week ending')
    save(fig,'05_low_high_periods')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for (region,g),color in zip(latest.groupby('region'),COLORS):
        axes[0].scatter(g.asof_value,g.final_value,s=9,alpha=.3,color=color,label=region,rasterized=True)
    high=max(latest.asof_value.max(),latest.final_value.max());low=min(0,latest.asof_value.min(),latest.final_value.min())
    axes[0].plot([low,high],[low,high],color='black',lw=1,ls='--')
    axes[0].set(xscale='symlog',yscale='symlog',xlabel='As-of count at origin',ylabel='Final count',title='Each state/target appears once (lag 0)');axes[0].legend(fontsize=7)
    revision_rate=latest.revision/latest.population_2019*100000
    axes[1].hist(np.arcsinh(revision_rate),bins=60,color=COLORS[1]);axes[1].set(xlabel='asinh((final - as-of) /100k)',ylabel='Origin-week pairs',title='Signed discrepancies; positive = final higher')
    save(fig,'06_asof_vs_final')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    if len(balanced_lag):
        axes[0].plot(balanced_lag.lag_weeks,balanced_lag.absolute_revision_mean,'o-',label='Mean')
        axes[0].plot(balanced_lag.lag_weeks,balanced_lag.absolute_revision_median,'o--',label='Median');axes[0].legend()
        axes[1].plot(balanced_lag.lag_weeks,100*balanced_lag.symmetric_relative_revision_mean,'o-',color=COLORS[1])
    axes[0].set(xlabel='Weeks after target week',ylabel='Absolute final/as-of difference',title='Balanced target cohort: mean and median')
    axes[1].set(xlabel='Weeks after target week',ylabel='Mean symmetric relative difference (%)',title='Same state/target pairs at every lag')
    save(fig,'07_revision_lag')
    chosen=latest.sort_values('absolute_revision',ascending=False).drop_duplicates('series').head(4)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,r in zip(axes.flat,chosen.itertuples()):
        history=pairs.loc[(pairs.series==r.series)&(pairs.target_end_date==r.target_end_date)&pairs.lag_weeks.le(12)].sort_values('lag_weeks')
        ax.plot(history.lag_weeks,history.asof_value,'o-',label='Historical vintages')
        ax.axhline(r.final_value,color='black',ls='--',label='Final truth')
        ax.set(title=f'{r.series}: target {r.target_end_date:%Y-%m-%d}',xlabel='Weeks after target',ylabel='Cases');ax.legend(fontsize=8)
    save(fig,'08_revision_examples')


def run_exploration(synthetic_path,manifest_path,output_dir,k=5,h=4,archive_dir=None,population=None):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    lock=acquire_run_lock(output_dir/'run.lock')
    archive_dir=Path(archive_dir or ROOT/'outputs/paper_weekly_archive')
    population=Path(population or Path(__file__).parent/'reference/state_census_2019.csv')
    if k<1 or h<1:raise ValueError('k/h must be positive')
    atomic_json(output_dir/'progress.json',dict(status='synthetic_statistics'))
    curves,groups,examples=scan_library(Path(synthetic_path),k,h)
    syn=output_dir/'synthetic';real=output_dir/'real_world'
    syn.mkdir(exist_ok=True);real.mkdir(exist_ok=True)
    for name,frame in [('curves',curves),('groups',groups),('examples',examples)]:frame.to_csv(syn/f'{name}.csv',index=False)
    generation=Path(synthetic_path).with_name('metadata.json')
    gen=json.loads(generation.read_text()) if generation.exists() else {}
    library_hash=file_hash(synthetic_path)
    if gen.get('curves_sha256') and gen['curves_sha256']!=library_hash:raise ValueError('Synthetic input checksum mismatch')
    if gen.get('accepted_total') and gen['accepted_total']!=len(curves):raise ValueError('Synthetic accepted count mismatch')
    synthetic_plots(curves,examples,syn/'figures')
    atomic_json(output_dir/'progress.json',dict(status='real_world_statistics'))
    truth,common,census,manifest=load_truth(Path(manifest_path),population)
    states=distributions(common,['series','region'])
    for field,lo,hi in [('value',100,10000),('cases_per_100k',1,100)]:
        levels=common.assign(low=common[field].between(0,lo),high=common[field].ge(hi)).groupby('series')[['low','high']].mean()
        states=states.merge(levels.rename(columns={c:field+'_'+c+'_fraction' for c in levels}),on='series',validate='one_to_one')
    regions=common.groupby(['region','target_end_date'],as_index=False).agg(value=('value','sum'),population_2019=('population_2019','sum'),states=('series','nunique'))
    regions['cases_per_100k']=regions.value/regions.population_2019*100000
    region_summary=pd.DataFrame([dict(region=r,states=int(g.states.iloc[0]),population_2019=int(g.population_2019.iloc[0]),
        mean_weekly_cases=float(g.value.mean()),mean_cases_per_100k=float(g.cases_per_100k.mean()),
        median_cases_per_100k=float(g.cases_per_100k.median()),peak_cases_per_100k=float(g.cases_per_100k.max()),
        peak_date=g.loc[g.cases_per_100k.idxmax(),'target_end_date'].date().isoformat()) for r,g in regions.groupby('region')])
    changes=weekly_changes(common);runs=periods(common);waves=real_peaks(common)
    pairs=compare_vintages(archive_dir,truth)
    latest,by_lag,by_state,balanced,balanced_lag=revision_tables(pairs)
    sharp=changes.loc[changes.sharp_change].sort_values('change',key=lambda s:s.abs(),ascending=False)
    all_distribution=distributions(common.assign(group='all_state_weeks'),['group'])
    for name,frame in [('final_all_records',truth),('common_weekly_counts',common),('state_summary',states),
        ('case_distribution',all_distribution),('region_weekly',regions),('region_summary',region_summary),
        ('weekly_changes',changes),('sharp_changes',sharp),('low_high_periods',runs),('epidemic_peaks',waves),
        ('revision_pairs',pairs),('revision_latest',latest),('revision_by_lag',by_lag),('revision_by_state',by_state),
        ('revision_balanced_cohort',balanced),('revision_balanced_lag',balanced_lag),
        ('largest_revisions',latest.nlargest(100,'absolute_revision'))]:frame.to_csv(real/f'{name}.csv',index=False)
    real_plots(common,states,regions,changes,latest,balanced_lag,pairs,real/'figures')
    summary=dict(synthetic=dict(curves=len(curves),observations=int(curves.observations.sum()),
        families=curves.curve_type.value_counts().to_dict(),attempts=gen.get('attempted_total'),accepted=gen.get('accepted_total'),
        embedding_windows_keep=int(curves.embedding_windows_keep.sum()),embedding_windows_omit=int(curves.embedding_windows_omit.sum()),
        peak_rule=PEAK_RULE,k=k,h=h),
        real_world=dict(states=truth.series.nunique(),source_records=len(truth),common_records=len(common),
        common_start=str(common.target_end_date.min().date()),common_end=str(common.target_end_date.max().date()),
        negative_values=int(common.value.lt(0).sum()),zeros=int(common.value.eq(0).sum()),sharp_events=len(sharp),
        sustained_low_high_periods=int(runs.sustained.sum()),historical_pairs=len(pairs),origin_week_pairs=len(latest),
        latest_revision=revision_summary(latest),balanced_target_pairs=balanced[['series','target_end_date']].drop_duplicates().shape[0],
        balanced_records=len(balanced),rules=RULES))
    atomic_json(output_dir/'summary.json',summary)
    source_paths=[Path(synthetic_path),Path(manifest_path),population,generation,archive_dir/'manifest.csv',archive_dir/'provenance.json',
                  *[Path(manifest_path).parent/r.truth for r in manifest.itertuples()],
                  *[archive_dir/r.as_of_truth for r in pd.read_csv(archive_dir/'manifest.csv').itertuples()],
                  *Path(__file__).parent.glob('*.py')]
    atomic_json(output_dir/'metadata.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),packages=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
        sha256={str(p.resolve()):file_hash(p) for p in source_paths if p.exists()},
        population_source='LANL/precog smoa/data/state_census.csv, POPESTIMATE2019; fixed denominator, not annual population',
        scope='Full library statistics; only example curves are sampled. Common calendar for state/region comparisons. No forecasting or tuning.'))
    write_reports(syn,real,curves,groups,states,region_summary,sharp,runs,latest,balanced_lag,summary)
    (output_dir/'report.md').write_text('# Dataset Exploration\n\n[1. Synthetic Dataset Exploration](synthetic/report.md)\n\n[2. Real-World Dataset Exploration](real_world/report.md)\n\nFull input/code checksums: metadata.json; completion: progress.json.\n')
    atomic_json(output_dir/'progress.json',dict(status='complete',synthetic_curves=len(curves),states=truth.series.nunique(),figures=12))
    return summary


def write_reports(syn,real,curves,groups,states,regions,sharp,runs,latest,balanced_lag,summary):
    s=summary['synthetic'];r=summary['real_world']
    group_cols=['curve_type','scale','time_cadence','curves','observations_median','epidemic_peaks_median','mean_absolute_change_over_mean_median']
    lines=['# Synthetic Dataset Exploration','',
        f"{s['curves']:,} accepted curves, {s['observations']:,} observations; {s['attempts']} attempts. Full CSV scanned, no statistical subsampling.",'',markdown(groups[group_cols]),
        '', '## Định nghĩa và cách đọc', '',
        '- Epidemic peak: median-3 smoothing, prominence ≥10% curve range, distance ≥4 original steps. Endpoints không được tính; cột boundary_maximum ghi riêng. Sensitivity 5%/20% có trong curves.csv.',
        '- Peak counts là số đỉnh phát hiện trên observations, không phải số SIR components; waves có thể chồng lấn và bị ảnh hưởng bởi noise/rounding.',
        '- Volatility chính: mean(abs(diff(values)))/mean(values). Không phụ thuộc nhân scale; so sánh trong cùng cadence/scale. CV và SD(diff)/mean cũng được xuất.',
        '- Counts và proportion được giữ riêng; weekly và monthly không phải cùng đơn vị thời gian. Peak density /100 steps không được gộp qua cadence.',
        '- Độ dài khác nhau ảnh hưởng số đỉnh và số embedding windows. Đây là mô tả các họ dữ liệu, không phải paired causal effect của wiggle.',
        '', '## Khác biệt theo cùng đơn vị weekly/counts','']
    subset=groups.loc[groups.scale.eq('counts')&groups.time_cadence.eq('weekly')]
    for row in subset.itertuples():
        lines.append(f'- {LABELS[row.curve_type]}: median length {row.observations_median:.0f} steps; detected peaks {row.epidemic_peaks_median:.1f}; relative step volatility {row.mean_absolute_change_over_mean_median:.4f}.')
    lines += ['',f"Embedding windows k={s['k']}/h={s['h']}: giữ bridge {s['embedding_windows_keep']:,}; bỏ bridge {s['embedding_windows_omit']:,}. Đây là số windows, không phải số curves.",'']
    for name in ['01_shapes','02_lengths_and_peaks','03_scale_distributions','04_volatility']:lines += [f'![{name}](figures/{name}.png)','']
    (syn/'report.md').write_text('\n'.join(lines))
    revision=r['latest_revision']
    distribution=pd.read_csv(real/'case_distribution.csv').iloc[0]
    lines=['# Real-World Dataset Exploration','',
        f"{r['states']} bang; {r['source_records']:,} raw final records. So sánh khu vực dùng lịch chung {r['common_start']}–{r['common_end']} ({r['common_records']:,} state-weeks), không trộn phần Arizona dài hơn.",
        '',f"Raw common data: {r['zeros']} zeros, {r['negative_values']} negative revisions. Không floor/impute/drop các giá trị này trong thống kê.",
        '', '## Phân phối và khác biệt khu vực','',
        f"Weekly cases/state: median {distribution.value_median:,.0f}, mean {distribution.value_mean:,.2f}, p95 {distribution.value_p95:,.2f}, max {distribution.value_max:,.0f}. Mean cao hơn median cho thấy đuôi phải dài; raw values có cả revisions âm.",
        '',markdown(regions),
        '', 'Rates dùng population 2019 cố định từ state_census nguồn; không phải annual population-adjusted risk. Region rate = sum(cases)/sum(population), không phải mean các state rates.',
        '', '## Những tuần tăng/giảm mạnh','',
        f"{len(sharp):,} events thỏa đồng thời |weekly change| ≥1.000 và |relative change| ≥50%, previous>0/current>=0. Zero→positive được đánh dấu riêng; không chia cho zero hoặc tự thêm epsilon.",
        '',markdown(sharp[['series','target_end_date','previous_value','value','change','relative_change']].head(12)),
        '', 'Biến động báo cáo, backlog và revisions có thể gây nhảy số; không coi tất cả sharp changes là thay đổi sinh học của dịch.',
        '', '## Khoảng số ca rất thấp/cao','',
        f"{int(runs.sustained.sum()):,} runs kéo dài ≥3 tuần theo các ngưỡng đã ghi (counts hoặc /100k). CSV giữ cả runs ngắn hơn.",
        '', '- Raw counts: low 0–100/tuần, high ≥10.000. Case rate: low 0–1/100k/tuần, high ≥100/100k/tuần. Negative revisions không gán là low.',
        '- Counts/rate runs có thể trùng nhau; không cộng chúng thành số đợt dịch độc lập.',
        '',markdown(runs.loc[runs.unit.eq('cases_per_100k') & runs.sustained].sort_values('weeks',ascending=False)[['series','level','start','end','weeks']].head(10)),
        '', '## As-of so với final đã hiệu chỉnh','',
        f"Origin-week unit: {len(latest):,} pairs, mỗi state/target một lần. {revision['changed_fraction']:.1%} pairs khác final; mean signed(final−asof) {revision['signed_revision_mean']:,.2f}; mean absolute difference {revision['absolute_revision_mean']:,.2f} cases.",
        '', f"All-history comparison: {r['historical_pairs']:,} state/target/vintage pairs. Cùng target xuất hiện nhiều vintage; không coi đó là những targets độc lập.",
        '', f"Balanced lag cohort: {r['balanced_target_pairs']:,} state/target pairs có đủ mọi lag 0–12 tuần, tổng {r['balanced_records']:,} records. Đường lag dùng đúng cùng cohort để tránh thay đổi composition.",
        '',markdown(balanced_lag[['lag_weeks','pairs','absolute_revision_mean','absolute_revision_median','symmetric_relative_revision_mean']]),
        '', 'Symmetric relative difference = 2|final−asof|/(|final|+|asof|), bằng 0 nếu cả hai bằng 0. Bị chặn 0–200%; đây là discrepancy metric của EDA, không phải forecasting error.',
        '', 'As-of được dựng lại từ JHU county snapshots; final đến từ LANL state_truths. Chênh lệch có thể gồm revisions và khác biệt upstream/aggregation, không quy toàn bộ thành reporting delay. Sunday cutoff sau Saturday label, trước Monday issuance, theo nguồn.',
        '', 'Revision examples chọn hậu nghiệm từ các origin-week có absolute discrepancies lớn nhất ở bốn bang khác nhau; không đại diện toàn bộ dữ liệu. EDA hồi cứu không được đưa full-data statistics vào forecast-origin features hoặc tune theo test.','']
    for name in ['01_case_distribution','02_sharp_changes','03_regions','04_state_distributions','05_low_high_periods','06_asof_vs_final','07_revision_lag','08_revision_examples']:lines += [f'![{name}](figures/{name}.png)','']
    (real/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',type=Path,default=ROOT/'outputs/reproduction_full/01_synthetic/curves.csv')
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/truth_states/manifest.csv')
    p.add_argument('--archive-dir',type=Path,default=ROOT/'outputs/paper_weekly_archive')
    p.add_argument('--population',type=Path,default=Path(__file__).parent/'reference/state_census_2019.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/dataset_exploration')
    p.add_argument('--k',type=int,default=5);p.add_argument('--h',type=int,default=4)
    a=p.parse_args()
    print(json.dumps(run_exploration(a.synthetic,a.manifest,a.output_dir,a.k,a.h,a.archive_dir,a.population),indent=2))
