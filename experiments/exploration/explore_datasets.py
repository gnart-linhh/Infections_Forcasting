"""Read-only exploration of synthetic curves and weekly state truth CSVs."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'infections-forecasting-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ['sir_rollercoaster', 'sir_rollercoaster_wiggle', 'seasonal']
COLORS = ['#2563a6', '#df8430', '#309077']


def load_inputs(synthetic_path, manifest_path):
    synthetic = pd.read_csv(synthetic_path, dtype={'curve_id': str})
    required = {'curve_id','curve_type','time_cadence','scale','step','value'}
    if not required.issubset(synthetic.columns):
        raise ValueError(f'Missing synthetic columns: {required-set(synthetic.columns)}')
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    if manifest.empty or manifest['series'].duplicated().any() or (manifest['series'].str.strip() == '').any():
        raise ValueError('Manifest must list unique nonempty series')
    frames, sources = [], [Path(synthetic_path), Path(manifest_path)]
    for entry in manifest.to_dict('records'):
        path = (Path(manifest_path).parent/entry['truth']).resolve()
        frame = pd.read_csv(path)
        frame = frame[['target_end_date', 'value']].copy()
        frame['series'] = entry['series']; frame['source_file'] = path.name
        frames.append(frame); sources.append(path)
    truth = pd.concat(frames, ignore_index=True)
    truth['target_end_date'] = pd.to_datetime(truth['target_end_date'], errors='coerce')
    for frame in [synthetic, truth]:
        frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    synthetic['step'] = pd.to_numeric(synthetic['step'], errors='coerce')
    if synthetic.empty:
        raise ValueError('Synthetic CSV is empty')
    return synthetic, truth, sources


def summarize_synthetic(data, k=5, h=4):
    rows = []
    for curve_id, group in data.groupby('curve_id', sort=False, dropna=False):
        group = group.sort_values('step')
        values = group['value'].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        steps = group['step'].to_numpy(dtype=float)
        valid_steps = np.array_equal(steps, np.arange(1,len(group)+1))
        metadata_ok = all(group[c].nunique(dropna=False)==1 for c in ['curve_type','scale','time_cadence'])
        embeddings = 0
        if len(values) >= k+h+1 and np.all(np.isfinite(values)) and valid_steps and metadata_ok:
            windows = np.lib.stride_tricks.sliding_window_view(np.maximum(values,1e-8),k+h+1)
            embeddings = int((np.var(windows[:,:k+1], axis=1, ddof=1)>0).sum())
        rows.append(dict(curve_id=curve_id, curve_type=group['curve_type'].iloc[0],
            scale=group['scale'].iloc[0], time_cadence=group['time_cadence'].iloc[0],
            observations=len(group), finite_values=len(finite),
            min_value=float(finite.min()) if len(finite) else None,
            median_value=float(np.median(finite)) if len(finite) else None,
            max_value=float(finite.max()) if len(finite) else None,
            floor_values=int((values<=1e-10).sum()), negative_values=int((values<0).sum()),
            step_sequence_valid=valid_steps, metadata_consistent=metadata_ok,
            embedding_windows=embeddings))
    return pd.DataFrame(rows)


def summarize_truth(data):
    rows = []
    for series, group in data.groupby('series', sort=True):
        group = group.sort_values('target_end_date')
        valid = group[np.isfinite(group['value']) & group['target_end_date'].notna()]
        dates = group['target_end_date'].dropna().drop_duplicates().sort_values()
        expected = len(pd.date_range(dates.min(),dates.max(),freq='7D')) if len(dates) else 0
        peak = valid.loc[valid['value'].idxmax()] if len(valid) else None
        changes = group['value'].diff()
        rows.append(dict(series=series, observations=len(group),
            first_date=dates.min().date().isoformat() if len(dates) else None,
            last_date=dates.max().date().isoformat() if len(dates) else None,
            missing_values=int((~np.isfinite(group['value'])).sum()), invalid_dates=int(group['target_end_date'].isna().sum()),
            duplicate_date_rows=int(group.duplicated('target_end_date',keep=False).sum()),
            nonweekly_intervals=int((dates.diff().dropna()!=pd.Timedelta(days=7)).sum()),
            missing_week_slots=max(0, expected-len(dates)),
            negative_values=int((group['value']<0).sum()), zeros=int((group['value']==0).sum()),
            values_changed_by_floor_1=int((group['value']<1).sum()),
            min_value=float(valid['value'].min()) if len(valid) else None,
            median_value=float(valid['value'].median()) if len(valid) else None,
            p95_value=float(valid['value'].quantile(.95)) if len(valid) else None,
            peak_value=float(peak['value']) if peak is not None else None,
            peak_date=peak['target_end_date'].date().isoformat() if peak is not None else None,
            largest_absolute_weekly_change=float(changes.abs().max()) if changes.notna().any() else None))
    return pd.DataFrame(rows)


def truth_anomalies(data):
    frame = data.copy()
    frame['flags'] = ''
    for mask, flag in [(~np.isfinite(frame['value']),'nonfinite_value'),
                       (frame['value']<0,'negative_value'), (frame['value']==0,'zero'),
                       (frame['target_end_date'].isna(),'invalid_date'),
                       (frame.duplicated(['series','target_end_date'],keep=False),'duplicate_date')]:
        frame.loc[mask,'flags'] += flag + ';'
    frame['flags'] = frame['flags'].str.rstrip(';')
    return frame.loc[frame['flags']!=''].copy()


def largest_changes(data):
    frame = data.sort_values(['series','target_end_date']).copy()
    frame['previous_value'] = frame.groupby('series')['value'].shift()
    frame['change'] = frame['value']-frame['previous_value']
    frame['absolute_change'] = frame['change'].abs()
    return frame.nlargest(20, 'absolute_change')


def draw_plots(synthetic, truth, synstats, truthstats, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'white', 'axes.titlepad':10})
    plots = []
    def save(fig, name):
        fig.savefig(directory/f'{name}.png', dpi=150, bbox_inches='tight')
        fig.savefig(directory/f'{name}.svg', bbox_inches='tight')
        plt.close(fig); plots.append(directory/f'{name}.png')

    fig, axes = plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for r, family in enumerate(FAMILIES):
        for c, scale in enumerate(['counts','proportion']):
            ax = axes[r,c]
            selected = synstats[(synstats['curve_type']==family)&(synstats['scale']==scale)].head(3)
            for idx, row in enumerate(selected.itertuples()):
                curve = synthetic[synthetic['curve_id']==row.curve_id].sort_values('step')
                peak = curve['value'].max()
                ax.plot(curve['step'],curve['value']/peak if peak>0 else curve['value'],
                    color=COLORS[idx], linestyle='--' if row.time_cadence=='monthly' else '-',
                    label=f'ID {row.curve_id}, {row.time_cadence}')
            ax.set(title=f'{family.replace("_", " ")} / {scale}',xlabel='Step (original cadence)',ylabel='Value / own peak')
            if len(selected): ax.legend(fontsize=8)
            else: ax.text(.5,.5,'No curves in this group',transform=ax.transAxes,ha='center')
    fig.suptitle('Synthetic shapes: up to 3 curves per family and scale',fontsize=15)
    save(fig,'01_synthetic_examples')

    fig, axes = plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for scale, ax in zip(['counts','proportion'],axes[:2]):
        pooled_peaks = synstats.loc[(synstats['scale']==scale)&(synstats['max_value']>0),'max_value']
        bins = np.histogram_bin_edges(np.log10(pooled_peaks),bins=8) if len(pooled_peaks) else 8
        for color, family in zip(COLORS,FAMILIES):
            peaks = synstats.loc[(synstats['scale']==scale)&(synstats['curve_type']==family),'max_value']
            peaks = peaks[peaks>0]
            if len(peaks): ax.hist(np.log10(peaks),bins=bins,alpha=.5,label=family.replace('sir_',''),color=color)
        ax.set(title=f'Peak magnitude: {scale}',xlabel='log10(curve peak)',ylabel='Curves'); ax.legend(fontsize=8)
    length_bins = np.histogram_bin_edges(synstats['observations'], bins=10)
    for cadence, color in zip(['weekly','monthly'],COLORS):
        lengths=synstats.loc[synstats['time_cadence']==cadence,'observations']
        if len(lengths): axes[2].hist(lengths,bins=length_bins,alpha=.6,color=color,label=cadence)
    axes[2].set(title='Curve lengths by original cadence',xlabel='Steps (not comparable durations)',ylabel='Curves')
    axes[2].legend()
    save(fig,'02_synthetic_distributions')

    grid = truth.pivot_table(index='series',columns='target_end_date',values='value',aggfunc='size').fillna(0)
    fig, ax = plt.subplots(figsize=(14,11),layout='constrained')
    ax.imshow(grid.to_numpy()>0,aspect='auto',interpolation='nearest',cmap='Blues',vmin=0,vmax=1)
    ax.set_yticks(range(len(grid.index)),grid.index,fontsize=7)
    ticks = np.linspace(0,len(grid.columns)-1,min(9,len(grid.columns))).astype(int)
    ax.set_xticks(ticks,[grid.columns[i].date().isoformat() for i in ticks],rotation=25,ha='right')
    ax.set(title='Raw truth calendar: blue = row present, white = absent',xlabel='Week ending')
    save(fig,'03_truth_calendar')

    fig, axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    chosen = [s for s in ['California','Texas','New York','Florida'] if s in set(truth['series'])]
    chosen += [s for s in sorted(truth['series'].unique()) if s not in chosen][:4-len(chosen)]
    for ax, series in zip(axes.flat,chosen):
        group = truth[truth['series']==series].sort_values('target_end_date')
        ax.plot(group['target_end_date'],group['value'],color=COLORS[0],linewidth=1.2,label='Raw truth')
        mask = group['value']<0
        if mask.any():
            ax.scatter(group.loc[mask,'target_end_date'],group.loc[mask,'value'],marker='x',color='#c93e43',s=40,label='Negative value',zorder=3)
        ax.axhline(0,color='#888888',linewidth=.6)
        ax.set(title=series,ylabel='Weekly cases (raw)'); ax.tick_params(axis='x',rotation=20); ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(chosen):]: ax.set_visible(False)
    save(fig,'04_truth_examples')

    common = truth.pivot_table(index='target_end_date',columns='series',values='value',aggfunc='first')
    common = common.loc[common.notna().all(axis=1)]
    normalized = common.clip(lower=0).div(common.clip(lower=0).max(axis=0).replace(0,np.nan),axis=1)
    fig, axes = plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    if len(normalized):
        q = normalized.quantile([.1,.5,.9],axis=1).T
        axes[0].fill_between(q.index,q[.1],q[.9],alpha=.2,color=COLORS[0],label='10th–90th percentile across states')
        axes[0].plot(q.index,q[.5],color=COLORS[0],label='Median across states')
    axes[0].set(title='States normalized on shared observed weeks',ylabel='Cases / state peak on shared weeks')
    axes[0].tick_params(axis='x',rotation=20);axes[0].legend(fontsize=8)
    axes[1].hist(np.log10(truthstats['peak_value'].clip(lower=1)),bins=15,color=COLORS[1],alpha=.8)
    axes[1].set(title='Raw peak magnitude differs across states',xlabel='log10(state weekly peak)',ylabel='States')
    save(fig,'05_truth_scale_and_timing')
    return plots


def run_exploration(synthetic_path, manifest_path, output_dir, k=5, h=4):
    if k<1 or h<1: raise ValueError('k and h must be positive')
    synthetic, truth, sources = load_inputs(synthetic_path,manifest_path)
    synstats=summarize_synthetic(synthetic,k,h); truthstats=summarize_truth(truth)
    anomalies=truth_anomalies(truth); changes=largest_changes(truth)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    groups = synstats.groupby(['curve_type','scale','time_cadence'],dropna=False).agg(
        curves=('curve_id','size'),median_steps=('observations','median'),min_steps=('observations','min'),
        max_steps=('observations','max'),embedding_windows=('embedding_windows','sum')).reset_index()
    for name, frame in [('synthetic_curves.csv',synstats),('synthetic_groups.csv',groups),
                        ('truth_series.csv',truthstats),('truth_anomalies.csv',anomalies),('largest_weekly_changes.csv',changes)]:
        frame.to_csv(output_dir/name,index=False)
    summary=dict(synthetic=dict(curves=len(synstats),observations=len(synthetic),
        families={str(k):int(v) for k,v in synstats['curve_type'].value_counts().items()},
        cadence={str(k):int(v) for k,v in synstats['time_cadence'].value_counts().items()},
        scale={str(k):int(v) for k,v in synstats['scale'].value_counts().items()},
        floor_values=int(synstats['floor_values'].sum()),embedding_windows=int(synstats['embedding_windows'].sum()),k=k,h=h,
        nonfinite_values=int((~np.isfinite(synthetic['value'])).sum()),
        invalid_step_sequences=int((~synstats['step_sequence_valid']).sum()),
        inconsistent_metadata=int((~synstats['metadata_consistent']).sum())),
        truth=dict(series=len(truthstats),observations=len(truth),
        first_date=truth['target_end_date'].min().date().isoformat(),last_date=truth['target_end_date'].max().date().isoformat(),
        common_start=truthstats['first_date'].max(),common_end=truthstats['last_date'].min(),
        negative_values=int(truthstats['negative_values'].sum()),series_with_negatives=int((truthstats['negative_values']>0).sum()),
        zero_values=int(truthstats['zeros'].sum()),values_changed_by_floor_1=int(truthstats['values_changed_by_floor_1'].sum()),
        missing_values=int(truthstats['missing_values'].sum()),invalid_dates=int(truthstats['invalid_dates'].sum()),
        duplicate_date_rows=int(truthstats['duplicate_date_rows'].sum()),nonweekly_intervals=int(truthstats['nonweekly_intervals'].sum())))
    plots=draw_plots(synthetic,truth,synstats,truthstats,output_dir/'figures')
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    metadata=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+[Path(__file__)]},
        notes=['Read-only raw inputs; no modifications or imputation',
               'Synthetic floor <=1e-10 reflects generator clipping, not missing observations',
               'Peak normalization is descriptive using the full dataset, not forecasting preprocessing',
               'Raw counts/proportions and weekly/monthly steps are not pooled as comparable units',
               'Truth plots keep negative revisions, except the explicitly labelled nonnegative normalized panel'])
    (output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['# Dataset exploration','',f"- Synthetic: {len(synstats)} curves, {len(synthetic):,} observations, {summary['synthetic']['embedding_windows']:,} embedding windows (k={k}, h={h}).",
        f"- Truth: {len(truthstats)} states, {len(truth):,} weekly records; {summary['truth']['common_start']} to {summary['truth']['common_end']} is the shared calendar range.",
        f"- Negative truth: {summary['truth']['negative_values']} values in {summary['truth']['series_with_negatives']} states. Zero values: {summary['truth']['zero_values']}.",
        '', 'See CSV tables for per-curve/per-state statistics and flags. Zero values are listed for inspection, not declared data errors.',
        'Monthly steps are not converted into weekly steps. Largest changes are candidates for inspection, not automatically errors.',
        'No observations are dropped from the source datasets.','']
    for image in plots: lines += [f'![{image.stem}](figures/{image.name})','']
    (output_dir/'report.md').write_text('\n'.join(lines))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',type=Path,default=ROOT/'outputs/synthetic_demo/curves.csv')
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/truth_states/manifest.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/dataset_exploration')
    p.add_argument('--k',type=int,default=5);p.add_argument('--h',type=int,default=4)
    a=p.parse_args()
    print(json.dumps(run_exploration(a.synthetic,a.manifest,a.output_dir,a.k,a.h),indent=2))


if __name__=='__main__':
    main()
