"""Compare Experiment 4 with full runs on exactly matched state/origin/targets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from runtime import atomic_json,file_hash

KEYS=['series','forecast_origin','target_end_date','horizon']


def compare(instances,full):
    paths=sorted((full/'04_prediction').glob('*/forecasts.csv'))
    if not paths:raise ValueError(f'No full forecasts: {full}')
    data=pd.concat([pd.read_csv(p,usecols=KEYS+['forecast','true_value']) for p in paths],ignore_index=True)
    matched=instances.merge(data,on=KEYS,validate='one_to_one')
    if matched.empty:raise ValueError('No shared targets to compare')
    return dict(targets=len(matched),states=matched.series.nunique(),origins=matched.forecast_origin.nunique(),
        label_mismatches=int(matched.actual.ne(matched.true_value).sum()),
        equal_predictions=int(matched.prediction.eq(matched.forecast).sum()),
        max_absolute_prediction_difference=float((matched.prediction-matched.forecast).abs().max()),
        experiment_mae=float((matched.prediction-matched.actual).abs().mean()),
        full_mae=float((matched.forecast-matched.true_value).abs().mean())),matched


def audit(previous,corrected,keep_full,omit_full,panel_dir,archive_dir):
    before=pd.read_csv(previous/'instances.csv');after=pd.read_csv(corrected/'instances.csv')
    old_config=json.loads((previous/'protocol.json').read_text())
    new_config=json.loads((corrected/'protocol.json').read_text())
    merged=before.merge(after,on=KEYS,validate='one_to_one',suffixes=('_keep','_omit'))
    if len(merged)!=len(before) or len(merged)!=len(after):raise ValueError('Changed evaluation cohort')
    if not merged.actual_keep.equals(merged.actual_omit):raise ValueError('Evaluation labels changed')
    if old_config['library_sha256']!=new_config['library_sha256'] or old_config['panel_sha256']!=new_config['panel_sha256']:
        raise ValueError('Library/history inputs changed')
    if old_config['truth_sha256']!=new_config['truth_sha256']:raise ValueError('Final truths changed')
    result=dict(previous_keep_vs_full_keep=compare(before,keep_full)[0],
                previous_keep_vs_full_omit=compare(before,omit_full)[0],
                corrected_omit_vs_full_omit=compare(after,omit_full)[0])
    for name in ['previous_keep_vs_full_keep','corrected_omit_vs_full_omit']:
        if result[name]['label_mismatches'] or result[name]['max_absolute_prediction_difference']:
            raise ValueError(f'Prediction parity failed: {name}')
    for name,frame in [('previous_keep',before),('corrected_omit',after)]:
        offsets=(pd.to_datetime(frame.target_end_date)-pd.to_datetime(frame.forecast_origin)).dt.days
        if not offsets.eq(frame.horizon*7).all():raise ValueError('Target dates are misaligned')
        if not np.allclose(frame.absolute_error,(frame.prediction-frame.actual).abs(),rtol=0,atol=1e-9):
            raise ValueError('Incorrect exported errors')
        result[name]=dict(targets=len(frame),MAE=float(frame.absolute_error.mean()),
            RMSE=float(np.sqrt(np.mean((frame.prediction-frame.actual)**2))),
            normalized_MAE=float(frame.normalized_absolute_error.mean()),
            median_normalized_absolute_error=float(frame.normalized_absolute_error.median()),
            extreme_rate=float(frame.extreme.mean()))
    # Check actual inputs, not just equality of forecast labels/configuration names.
    panel=pd.read_csv(panel_dir/'panel.csv')
    overlap_origins=sorted(set(panel.forecast_origin)&set(pd.read_csv(archive_dir/'origins.csv').forecast_origin))
    vintage=[]
    for row in pd.read_csv(archive_dir/'manifest.csv').itertuples():
        frame=pd.read_csv(archive_dir/row.as_of_truth)
        frame=frame.loc[frame.as_of.isin(overlap_origins)].copy();frame['series']=row.series
        vintage.append(frame.rename(columns={'as_of':'forecast_origin'}))
    expected=panel.loc[panel.forecast_origin.isin(overlap_origins)]
    history=expected.merge(pd.concat(vintage,ignore_index=True),on=KEYS[:3],how='left',
        validate='one_to_one',suffixes=('_panel','_full'),indicator=True)
    if not history._merge.eq('both').all() or not history.value_panel.eq(history.value_full).all():
        raise ValueError('Historical panel differs from full archive')
    result['matched_history']=dict(records=len(history),origins=len(overlap_origins),value_mismatches=0)
    result['cohort_preserved']=dict(targets=len(merged),library_sha256=new_config['library_sha256'],
                                  panel_sha256=new_config['panel_sha256'],label_mismatches=0)
    out=corrected/'audit';out.mkdir(exist_ok=True)
    merged.to_csv(out/'bridge_comparison.csv',index=False)
    atomic_json(out/'comparison.json',result)
    atomic_json(out/'provenance.json',dict(script_sha256=file_hash(Path(__file__)),
        previous=str(previous.resolve()),corrected=str(corrected.resolve()),
        input_sha256={str(p):file_hash(p) for p in [previous/'instances.csv',previous/'protocol.json',corrected/'instances.csv',corrected/'protocol.json',panel_dir/'panel.csv',archive_dir/'manifest.csv']},
        full_sources={str(p):file_hash(p) for root in [keep_full,omit_full] for p in sorted((root/'04_prediction').glob('*/forecasts.csv'))}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in [('previous','experiment_4'),('corrected','experiment_4_omit'),
        ('keep-full','paper_full_bridge_matured'),('omit-full','paper_full'),
        ('panel-dir','epidemic_panel'),('archive-dir','paper_weekly_archive')]:
        parser.add_argument('--'+name,type=Path,default=ROOT/'outputs'/default)
    a=parser.parse_args()
    print(json.dumps(audit(a.previous,a.corrected,a.keep_full,a.omit_full,a.panel_dir,a.archive_dir),indent=2))
