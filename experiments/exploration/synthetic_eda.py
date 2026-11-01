"""Stream every synthetic curve and measure morphology without mixing units."""
import csv
from itertools import groupby
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'forecasting'))
from model import WINDOW_FILTER, valid_history_windows

FAMILIES=['sir_rollercoaster','sir_rollercoaster_wiggle','seasonal']
PEAK_RULE=dict(smoothing='centered median of 3 original steps, nearest edge padding',
               prominence_fraction=0.10,distance_steps=4,
               sensitivity_prominences=[0.05,0.20],endpoints='not counted; boundary_maximum recorded')


def peaks(values,prominence=.10,distance=4,smooth=True):
    values=np.asarray(values,dtype=float)
    signal=median_filter(values,size=3,mode='nearest') if smooth else values.copy()
    amplitude=float(np.ptp(signal))
    if len(signal)<3 or amplitude<=0:return np.array([],dtype=int)
    normalized=(signal-signal.min())/amplitude
    return find_peaks(normalized,prominence=prominence,distance=distance)[0]


def curve_metrics(values):
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Synthetic values must be finite and nonnegative')
    mean=float(values.mean());denominator=mean if mean>0 else 1.
    diff=np.diff(values)
    return dict(observations=len(values),mean_value=mean,min_value=float(values.min()),
        median_value=float(np.median(values)),max_value=float(values.max()),p95_value=float(np.quantile(values,.95)),
        cv=float(values.std(ddof=1)/denominator) if len(values)>1 and np.ptp(values)>0 else 0.,
        mean_absolute_change_over_mean=float(np.abs(diff).mean()/denominator) if len(diff) else 0.,
        change_std_over_mean=float(diff.std(ddof=1)/denominator) if len(diff)>1 else 0.,
        epidemic_peaks=len(peaks(values)),peaks_prominence_05=len(peaks(values,.05)),
        peaks_prominence_20=len(peaks(values,.20)),raw_prominent_peaks=len(peaks(values,smooth=False)),
        boundary_maximum=int(np.argmax(values) in [0,len(values)-1]),floor_values=int((values<=1e-10).sum()))


def scan_library(path,k=5,h=4):
    rows=[];examples=[];seen=set();example_counts={}
    with path.open() as handle:
        for curve,records in groupby(csv.DictReader(handle),key=lambda r:r['curve_id']):
            if curve in seen:raise ValueError('Synthetic rows must be grouped by curve_id')
            seen.add(curve);records=list(records)
            for field in ['curve_type','scale','time_cadence']:
                if len({r[field] for r in records})!=1:raise ValueError(f'Inconsistent {field}: {curve}')
            family=records[0]['curve_type'];scale=records[0]['scale'];cadence=records[0]['time_cadence']
            if family not in FAMILIES or scale not in ['counts','proportion'] or cadence not in ['weekly','monthly']:
                raise ValueError(f'Unknown synthetic metadata: {curve}')
            records.sort(key=lambda r:int(r['step']))
            if [int(r['step']) for r in records]!=list(range(1,len(records)+1)):
                raise ValueError(f'Invalid synthetic steps: {curve}')
            values=np.asarray([float(r['value']) for r in records])
            metrics=curve_metrics(values)
            metrics['peaks_per_100_steps']=100*metrics['epidemic_peaks']/max(len(values)-1,1)
            for mode,width in [('keep',k+h+1),('omit',k+h+2)]:
                count=0
                if len(values)>=width:
                    windows=np.lib.stride_tricks.sliding_window_view(np.maximum(values,1e-8),width)
                    count=int(valid_history_windows(windows[:,:k+1]).sum())
                metrics['embedding_windows_'+mode]=count
            rows.append(dict(curve_id=curve,curve_type=family,scale=scale,time_cadence=cadence,**metrics))
            key=(family,scale,cadence)
            if example_counts.get(key,0)<3:
                examples.extend(dict(curve_id=curve,curve_type=family,scale=scale,time_cadence=cadence,step=i,value=float(v))
                                for i,v in enumerate(values,1))
                example_counts[key]=example_counts.get(key,0)+1
            if len(rows)%10000==0:print(f'Synthetic EDA: {len(rows):,} curves scanned',flush=True)
    if not rows:raise ValueError('Synthetic library is empty')
    frame=pd.DataFrame(rows)
    groups=[]
    for key,group in frame.groupby(['curve_type','scale','time_cadence'],sort=False):
        row=dict(zip(['curve_type','scale','time_cadence'],key),curves=len(group),observations=int(group.observations.sum()),
                 floor_values=int(group.floor_values.sum()),boundary_maxima=int(group.boundary_maximum.sum()))
        row['embedding_windows_keep']=int(group.embedding_windows_keep.sum())
        row['embedding_windows_omit']=int(group.embedding_windows_omit.sum())
        for field in ['observations','max_value','epidemic_peaks','raw_prominent_peaks','mean_absolute_change_over_mean','change_std_over_mean','cv']:
            row.update({field+'_mean':float(group[field].mean()),field+'_median':float(group[field].median()),
                        field+'_p10':float(group[field].quantile(.10)),field+'_p90':float(group[field].quantile(.90))})
        groups.append(row)
    return frame,pd.DataFrame(groups),pd.DataFrame(examples)
