"""Preregistered origin-only strata and forecast errors for Experiment 4."""
from __future__ import annotations
import numpy as np
import pandas as pd

RULES = {
    'version':3,
    'phase':'(mean(last 2)+1)/(mean(previous 2)+1)-1: rising >0.10, declining <-0.10, stable otherwise',
    'case_level':'mean(last 4): low <1000, medium [1000,10000), high >=10000 weekly state cases',
    'volatility':'sample SD of differences of log1p(last 8 weekly counts): low <0.15, medium [0.15,0.40), high >=0.40',
    'history_stage':'early <=26 observed weekly values; later >26 (full available history, no artificial truncation)',
    'normalization':'max(mean(last 4 origin-available model-input weekly counts),1)',
    'extreme_error':'absolute error / origin normalization >1; sensitivity >0.5 and >2',
    'bias':'prediction minus final raw truth: negative means underprediction',
    'history_preprocessing':'floor raw weekly counts at 1, exactly as the sMOA forecast input; retain raw revision flags',
    'horizons':[1,2,3,4],
    'bootstrap':'state-cluster percentile bootstrap, 1000 replicates, seed 20261030; descriptive, not causal',
    'sampling':'weekly 2020-04-04..2020-09-05; every 4 weeks 2020-10-03..2022-10-01; all 50 states',
    'models':['Original_sMOA_Python'],
    'smoa':{'num_curves_per_family':18387,'k':5,'closest':4422,'smoothing':'gam','horizon':4,'bridge_mode':'omit'},
}


def features(raw):
    raw = np.asarray(raw,dtype=float)
    if len(raw)<8 or not np.isfinite(raw).all():
        raise ValueError('Need at least eight finite origin-available weeks')
    values = np.maximum(raw,1)
    recent = float(np.mean(values[-4:]))
    growth = float((np.mean(values[-2:])+1)/(np.mean(values[-4:-2])+1)-1)
    volatility = float(np.std(np.diff(np.log1p(values[-8:])),ddof=1))
    return dict(phase='rising' if growth>0.10 else 'declining' if growth < -0.10 else 'stable',
                case_level='low' if recent<1000 else 'medium' if recent<10000 else 'high',
                volatility_level='low' if volatility<0.15 else 'medium' if volatility<0.4 else 'high',
                history_stage='early' if len(raw)<=26 else 'later',
                history_weeks=len(raw), recent_cases=recent, growth=growth, volatility=volatility,
                normalization=max(recent,1), negative_history_weeks=int((raw<0).sum()),
                floored_history_weeks=int((raw<1).sum()))


def errors(frame):
    result = frame.copy()
    if (result.normalization<1).any() or not np.isfinite(result[['prediction','actual','normalization']]).all().all():
        raise ValueError('Nonfinite predictions, labels or normalization')
    result['error'] = result.prediction-result.actual
    result['absolute_error'] = result.error.abs()
    result['squared_error'] = result.error**2
    result['normalized_error'] = result.error/result.normalization
    result['normalized_absolute_error'] = result.absolute_error/result.normalization
    result['extreme'] = result.normalized_absolute_error>1
    result['extreme_05'] = result.normalized_absolute_error>0.5
    result['extreme_2'] = result.normalized_absolute_error>2
    result['underprediction'] = result.error<0
    return result


def metrics(frame):
    return dict(n=len(frame), states=frame.series.nunique(), origins=frame.forecast_origin.nunique(),
                MAE=float(frame.absolute_error.mean()), RMSE=float(np.sqrt(frame.squared_error.mean())),
                normalized_MAE=float(frame.normalized_absolute_error.mean()), bias=float(frame.error.mean()),
                normalized_bias=float(frame.normalized_error.mean()), extreme_rate=float(frame.extreme.mean()),
                extreme_rate_05=float(frame.extreme_05.mean()), extreme_rate_2=float(frame.extreme_2.mean()),
                underprediction_rate=float(frame.underprediction.mean()))


def summarize(frame, dimensions, bootstrap=True):
    rows=[]
    for key, group in frame.groupby(['model']+dimensions, observed=True, sort=True):
        if not isinstance(key,tuple): key=(key,)
        row=dict(zip(['model']+dimensions,key))
        row.update(metrics(group))
        if bootstrap:
            columns=['absolute_error','normalized_absolute_error','error','normalized_error','extreme']
            clusters=group.groupby('series')[columns].sum().to_numpy(dtype=float)
            sizes=group.groupby('series').size().to_numpy(dtype=float)
            if len(sizes)>=2:
                rng=np.random.default_rng(20261030)
                draw=rng.integers(0,len(sizes),size=(1000,len(sizes)))
                estimates=clusters[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
                for i,name in enumerate(['MAE','normalized_MAE','bias','normalized_bias','extreme_rate']):
                    row[name+'_lo'],row[name+'_hi']=np.quantile(estimates[:,i],[0.025,0.975])
        rows.append(row)
    return pd.DataFrame(rows)
