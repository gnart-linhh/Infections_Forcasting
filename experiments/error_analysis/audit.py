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


def e2_reference(e2, seed=0):
    """Validate the frozen v4 reference, including numerical source and bank identity."""
    from forecasting.model import WINDOW_FILTER
    e2=Path(e2).resolve()
    read=lambda path: json.loads(path.read_text())
    protocol=read(e2/'protocol.json');validation=read(e2/'validation.json')
    if (read(e2/'progress.json').get('status')!='complete'
        or protocol['design'].get('version')!=4
        or protocol['design'].get('window_filter')!=WINDOW_FILTER
        or validation.get('protocol_version')!=4
        or validation.get('window_filter')!=WINDOW_FILTER
        or validation.get('constant_history_rows')!=0
        or validation.get('protocol_sha256')!=file_hash(e2/'protocol.json')):
        raise ValueError('Require complete E2 v4 with nonconstant_v2 and zero constant histories')
    if protocol['design']['reference']!=dict(N=18387,k=5,m=4422):
        raise ValueError('E2 reference parameters changed')
    group=e2/f'groups/seed_{seed}_N18387_k5';runtime=read(group/'runtime.json')
    for name,value in dict(seed=seed,N=18387,k=5,h=4,bridge_mode='omit',window_filter=WINDOW_FILTER,constant_history_rows=0,cases=2500).items():
        if runtime.get(name)!=value:raise ValueError(f'E2 runtime mismatch: {name}')
    for name in ['experiments/forecasting/model.py','experiments/forecasting/neighbors.py',
                 'experiments/forecasting/smoothing.py','experiments/sensitivity/library.py']:
        if file_hash(ROOT/name)!=protocol['source_sha256'][name] or runtime['source_sha256'][name]!=protocol['source_sha256'][name]:
            raise ValueError(f'E2 numerical core changed: {name}; rerun E2 in a fresh directory')
    inputs=[e2/'protocol.json',e2/'validation.json',e2/'contexts.npz',e2/'context_metadata.json',group/'complete.json']
    context_hash=file_hash(e2/'contexts.npz')
    if context_hash!=runtime['context_sha256'] or context_hash!=read(e2/'context_metadata.json')['contexts_sha256']:
        raise ValueError('E2 contexts modified')
    completion=read(group/'complete.json')
    if completion.get('status')!='complete':raise ValueError('Incomplete E2 reference group')
    for name,sha in completion['sha256'].items():
        if file_hash(group/name)!=sha:raise ValueError(f'Modified E2 artifact: {name}')
        inputs.append(group/name)
    packed=e2/f'packed/seed_{seed}';inputs.append(packed/'metadata.json')
    for name,sha in runtime['packed_sha256'].items():
        if file_hash(packed/name)!=sha:raise ValueError(f'Modified E2 bank: {name}')
        inputs.append(packed/name)
    return dict(seed=seed,N=18387,k=5,m=4422,M=runtime['M'],window_filter=WINDOW_FILTER,
                constant_history_rows=0,context_sha256=context_hash,
                input_sha256={str(p):file_hash(p) for p in inputs})


def compare_reference(instances, reference, expected_targets=10000):
    """Require bitwise numeric equality and the entire common target set."""
    if len(instances)!=expected_targets or len(reference)!=expected_targets:
        raise ValueError('Incomplete reference cohort')
    paired=instances.merge(reference[KEYS+['prediction','actual']],on=KEYS,
                           how='outer',validate='one_to_one',suffixes=('','_reference'),indicator=True)
    if not paired._merge.eq('both').all():raise ValueError('Reference target keys differ')
    for name in ['prediction','actual']:
        if not np.isfinite(paired[[name,name+'_reference']]).all().all() or not paired[name].eq(paired[name+'_reference']).all():
            raise ValueError(f'Reference {name} mismatch')
    return dict(targets=len(paired),states=paired.series.nunique(),origins=paired.forecast_origin.nunique(),
                equal_predictions=len(paired),label_mismatches=0,max_absolute_prediction_difference=0.)


def validate_cache(cache, x, reference):
    """Check actual cached histories rather than trusting a version label."""
    from forecasting.model import WINDOW_FILTER
    metadata=json.loads((cache/'metadata.json').read_text())
    if metadata.get('version')!=2 or metadata.get('window_filter')!=WINDOW_FILTER or len(x)!=reference['M']:
        raise ValueError('Cache version/filter/row count differs from E2')
    # Histories here are increments: constant levels have all-zero increments.
    zeros=sum(int(np.all(x[start:start+200000]==0,axis=1).sum()) for start in range(0,len(x),200000))
    if zeros:raise ValueError('Constant history retained in cache')
    return dict(path=str(cache.resolve()),version=2,window_filter=WINDOW_FILTER,rows=len(x),constant_history_rows=zeros,
                metadata_sha256=file_hash(cache/'metadata.json'))


def validate_result(out, experiment):
    """Notebook gate: complete alone is insufficient to display current results."""
    from forecasting.model import WINDOW_FILTER
    out=Path(out);read=lambda path: json.loads(path.read_text())
    progress=read(out/'progress.json');protocol=read(out/'protocol.json')
    version={4:4,5:2}[experiment]
    if (progress.get('status')!='complete' or progress.get('version')!=version
        or protocol.get('version')!=version or progress.get('window_filter')!=WINDOW_FILTER
        or protocol.get('window_filter')!=WINDOW_FILTER or not progress.get('inputs_unchanged')
        or progress.get('protocol_sha256')!=file_hash(out/'protocol.json')):
        raise ValueError('Result is incomplete or uses a stale protocol/filter')
    for name,sha in protocol['input_sha256'].items():
        if file_hash(Path(name))!=sha:raise ValueError(f'Modified input: {name}')
    sources=protocol['source_sha256'] if experiment==4 else {
        **protocol['dependencies_sha256'],'experiments/analogue_quality/experiment_5.py':protocol['source_sha256']}
    for name,sha in sources.items():
        if file_hash(ROOT/name)!=sha:raise ValueError(f'Current source differs from result: {name}')
    for name,sha in progress['artifacts_sha256'].items():
        if file_hash(out/name)!=sha:raise ValueError(f'Modified result artifact: {name}')
    cache=progress['cache'];path=Path(cache['path']);metadata=read(path/'metadata.json')
    if (file_hash(path/'metadata.json')!=cache['metadata_sha256'] or cache['constant_history_rows']!=0
        or metadata['window_filter']!=WINDOW_FILTER or metadata['rows']!=cache['rows']):
        raise ValueError('Stale cache provenance')
    for name in ['x','y']:
        if file_hash(path/f'{name}.dat')!=metadata[f'{name}_sha256']:raise ValueError('Cache data modified')
    if experiment==4:
        if progress['forecast_count']!=10000 or progress['e2_parity']['equal_predictions']!=10000:
            raise ValueError('Incomplete E4/E2 exact parity')
        references=[protocol['e2_reference']]
    else:
        if progress['verified_forecasts']!=10000 or progress['verified_distance_cases']!=2500 or progress['cases']!=2500:
            raise ValueError('Incomplete E5 exact replay')
        if progress['analysis_source_sha256']!=protocol['source_sha256']:raise ValueError('E5 analysis source differs')
        references=protocol['e2_references']
    for reference in references:
        e2=Path(next(p for p in reference['input_sha256'] if p.endswith('/protocol.json'))).parent
        if e2_reference(e2,reference['seed'])!=reference:raise ValueError('E2 reference changed')
    return progress


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
