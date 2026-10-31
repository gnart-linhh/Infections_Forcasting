"""Prespecified Experiment 2 design, independent of observed forecast errors."""
from itertools import product
import pandas as pd

REFERENCE=(18387,5,4422)
LEVELS={'N':[500,2000,5000,10000,18387,22000,30000],'k':[3,5,7,9],'m':[100,500,2000,4422,8000]}
SEEDS=[0,1,2]
SEED_OFFSETS={0:0,1:1000000,2:2000000}
H=4


def configurations():
    records={}
    def add(params,section):
        record=records.setdefault(params,dict(N=params[0],k=params[1],m=params[2],sections=[]))
        record['sections'].append(section)
    for dim,levels in LEVELS.items():
        for value in levels:
            params=list(REFERENCE);params[['N','k','m'].index(dim)]=value
            add(tuple(params),'E2.A_'+dim)
    for n,k,m in product([1000,5000],[3,7],[500,2000]):add((n,k,m),'E2.B_factorial')
    add((2000,5,1000),'E2.B_center')
    for params,row in records.items():
        row['config']=f'N{params[0]}_k{params[1]}_m{params[2]}'
        row['sections']=';'.join(row['sections'])
    return pd.DataFrame(records.values()).sort_values(['N','k','m']).reset_index(drop=True)


def protocol():
    return dict(experiment='E2 parameter sensitivity',version=3,reference=dict(zip(['N','k','m'],REFERENCE)),
        configurations=configurations().to_dict('records'),seeds=SEEDS,seed_offsets={str(k):v for k,v in SEED_OFFSETS.items()},
        N_definition='attempts per generator, not accepted curves; no rejection replacement',
        nesting='original 18387-attempt family namespaces remain fixed; larger banks append independently seeded attempts without regenerating original curves',
        N_extension='N=22000 and N=30000 added at user request after the initial study; k=5,m=4422, unchanged targets and 3 seeds',
        main_cohort='all 50 states and all 50 origins from the unchanged Experiment 4 panel; 10000 common targets per config/seed',
        min_history_main=10,history='full origin-available JHU history; raw negatives/zeros floored at 1 as model input',
        horizons=[1,2,3,4],bridge_mode='omit',smoothing='gam',neighbors='exact L1, ties by original source row',
        aggregation='median future increments, cumulative sum plus last raw input floored at 1; forecast floor 1',
        selection='descriptive sensitivity only; COVID evaluation is not a hyperparameter-selection or BO feedback set',
        distances='raw L1 and L1/k; cross-k distances still have different geometry',
        early_stage=dict(seed=0,N=18387,m=4422,k=[3,5,7,9],history_lengths=[4,6,8,10],
            cases='the first 16 scheduled origins, all 50 states',
            interpretation='controlled last-L history availability at an existing genuine vintage, not a different early calendar',
            eligible='L >= max(k+1,5); unchanged forecast core requires at least 5 observations for GAM',
            comparison='report unavailable fits explicitly; compare k only within the same L and supported case set'),
        stability='three independent training banks, fixed COVID targets; seed SD is descriptive with only three seeds',
        factorial='2^3 N=[1000,5000], k=[3,7], m=[500,2000]; central point is descriptive, not an extra factorial corner',
        interactions='signed +/-1 contrasts on per-seed MAE: effect=2*mean(sign_product*MAE); not causal or a significance test',
        runtime='one index per (seed,N,k), exact maximum-m query shared across m; separate fixed-query benchmarks per m',
        memory='isolated group subprocess ru_maxrss includes packed bank load, embedding, KD tree and forecasts; excludes generation worker processes',
        generation_time='new seed max-bank generation+CSV export: metadata mtime minus run.lock mtime; imports excluded; historical seed0 generation duration unavailable',
        optional_M_control='not performed: changing k changes valid segment count M; N and M always reported')
