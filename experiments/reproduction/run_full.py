"""Full weekly COVID forecast/UQ run with paper parameters or a completed BO winner."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'experiments/data'))
sys.path.insert(0,str(ROOT/'experiments/forecasting'))
from runtime import acquire_run_lock,atomic_json,file_hash
from prepare_weekly_archive import prepare,common_schedule
from selection import apply_selection


def compare_runs(paper_run,bo_run,out):
    """Pair every scored target; never compare unmatched aggregate means."""
    import numpy as np
    import pandas as pd
    keys=['series','forecast_origin','target_end_date','horizon']
    frames=[]
    configs=[json.loads((d/'run_config.json').read_text()) for d in [paper_run,bo_run]]
    for key in ['method','origins','truth_sha256','source_sha256','packages']:
        if configs[0][key]!=configs[1][key]:
            raise ValueError(f'Paper/BO comparison protocol differs: {key}')
    for directory in [paper_run,bo_run]:
        frame=pd.read_csv(directory/'05_evaluation/scores.csv')
        frame=frame.loc[frame.model=='smoa']
        if frame.duplicated(keys).any(): raise ValueError('Duplicate comparison targets')
        frames.append(frame)
    joined=frames[0].merge(frames[1],on=keys,suffixes=('_paper','_bo'),validate='one_to_one')
    if len(joined)!=len(frames[0]) or len(joined)!=len(frames[1]):
        raise ValueError('BO and paper runs have different test targets')
    if not np.array_equal(joined.true_value_paper,joined.true_value_bo):
        raise ValueError('BO and paper evaluation labels differ')
    out.mkdir(parents=True,exist_ok=True)
    joined.to_csv(out/'paired_scores.csv',index=False)
    rows=[]
    for label,subset in [('all',joined)]+[(str(h),g) for h,g in joined.groupby('horizon')]:
        for column,metric in [('abs_error','MAE'),('wis','WIS')]:
            a=float(subset[column+'_paper'].mean());b=float(subset[column+'_bo'].mean())
            rows.append(dict(horizon=label,metric=metric,n=len(subset),paper=a,BO=b,difference_BO_minus_paper=b-a,
                             ratio_BO_to_paper=b/a if a else None))
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    lines=['# Full COVID evaluation: BO winner vs paper parameters','',
           f'{len(joined):,} exactly matched targets; identical raw truth and scoring code. Lower MAE/WIS is better.','',
           '| Horizon | Metric | Paper | BO | BO − paper |','| --- | --- | ---: | ---: | ---: |']
    lines += [f"| {r['horizon']} | {r['metric']} | {r['paper']:,.2f} | {r['BO']:,.2f} | {r['difference_BO_minus_paper']:,.2f} |" for r in rows]
    lines += ['', 'Positive differences mean BO performs worse on this COVID test. BO optimizes synthetic validation; a COVID improvement is not guaranteed.',
              'These are descriptive comparisons. Do not call a small difference statistically significant without an appropriate paired analysis.', '']
    (out/'report.md').write_text('\n'.join(lines))
    atomic_json(out/'metadata.json',dict(status='complete',targets=len(joined),source_sha256={str(d/'05_evaluation/scores.csv'):file_hash(d/'05_evaluation/scores.csv') for d in [paper_run,bo_run]}))


def run(a):
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=acquire_run_lock(out/'run.lock')
    started=time.time()
    state=(json.loads((out/'pipeline.json').read_text()) if a.resume and (out/'pipeline.json').exists() else
           dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),stages=[]))
    def update(status,**details):
        state.update(status=status,**details);atomic_json(out/'pipeline.json',state)
    if a.selection_dir and a.wait_for_selection and state.get('status')!='complete':
        update('waiting_for_bo',selection_dir=str(a.selection_dir),paper_run=str(a.paper_run))
        previous=None
        while True:
            path=a.selection_dir/'metadata.json'
            meta=json.loads(path.read_text()) if path.exists() else {}
            paper=json.loads((a.paper_run/'pipeline.json').read_text()) if (a.paper_run/'pipeline.json').exists() else {}
            current=(meta.get('status'),paper.get('status'))
            if current!=previous:
                print(f'Waiting: BO={current[0]}, paper={current[1]}',flush=True);previous=current
            if meta.get('status')=='complete' and paper.get('status')=='complete': break
            waiting_status='waiting_for_paper' if meta.get('status')=='complete' else 'waiting_for_bo'
            if state.get('status')!=waiting_status: update(waiting_status)
            if meta.get('status') in ['failed','interrupted'] or paper.get('status')=='failed':
                update('blocked',reason='BO or paper run failed/interrupted; inspect its log before resume')
                return
            time.sleep(30)
    execution_started=time.time()
    state['waiting_seconds']=execution_started-started
    apply_selection(a)
    library_meta=json.loads(a.synthetic.with_name('metadata.json').read_text())
    if not a.selection_dir:
        if a.k!=5 or a.closest!=4422 or library_meta.get('num_curves_per_family')!=18387 or library_meta.get('attempted_total')!=55161:
            raise ValueError('Paper mode requires N=18387/family,k=5,m=4422 and the full reference bank')
        if library_meta.get('curves_sha256')!=file_hash(a.synthetic):
            raise ValueError('Paper synthetic library checksum mismatch')
    entries,origins=common_schedule(a.manifest)
    params=json.loads((a.selection_dir/'best_params.json').read_text()) if a.selection_dir else dict(num_curves=18387,k=5,closest=4422)
    if a.embedding_cache is None:
        # Distinct winners must never share a cache just because k happens to match.
        a.embedding_cache=a.archive_dir.parent/f'embedding_{file_hash(a.synthetic)[:16]}_k{a.k}_h4_{a.bridge_mode}'
    code_paths=[Path(__file__),ROOT/'experiments/runtime.py',ROOT/'experiments/error_analysis/library.py']
    code_paths += [ROOT/'experiments/data'/n for n in ['prepare_weekly_archive.py','fetch_paper_data.py']]
    code_paths += [p for p in (ROOT/'experiments/forecasting').glob('*.py') if not p.name.startswith('test_')]
    config=dict(selection='BO winner' if a.selection_dir else 'paper fixed parameters',
                reuse_predictions_from=str(a.reuse_predictions_from.resolve()) if a.reuse_predictions_from else None,
                parameters=dict(N=params['num_curves'],k=a.k,m=a.closest,h=4),
                library=str(a.synthetic.resolve()),library_sha256=file_hash(a.synthetic),
                selection_dir=str(a.selection_dir.resolve()) if a.selection_dir else None,
                selected_params=params if a.selection_dir else None,
                manifest=str(a.manifest.resolve()),manifest_sha256=file_hash(a.manifest),
                truth_sha256={r.series:file_hash(a.manifest.parent/r.truth) for r in entries.itertuples()},
                archive_dir=str(a.archive_dir.resolve()),origins=origins,states=50,targets_per_model=len(origins)*50*4,
                method=dict(models=['smoa'],smoothing='gam',coverage_quantiles=23,coverage_levels=11,interval_draws=5000,seed=13,
                            bridge_mode=a.bridge_mode,uq_policy=a.uq_policy,
                            score_truth='raw final',calibration_truth=('final outcomes of earlier issued forecasts, including unmatured targets'
                                if a.uq_policy=='figure2_retrospective' else 'final matured outcomes'),
                            prehistory='frozen first scored vintage, as in upstream early-error initialization; retrospective warmup'),
                source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in code_paths},
                packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','matplotlib']})
    if (out/'run_config.json').exists():
        if json.loads((out/'run_config.json').read_text())!=config:
            raise ValueError('Full-run inputs/source/config changed')
        if not a.resume: raise ValueError('Full run initialized; use --resume')
    else: atomic_json(out/'run_config.json',config)
    if a.resume and (out/'05_evaluation/summary.json').exists():
        saved=json.loads((out/'pipeline.json').read_text())
        if saved.get('status')=='complete': return
    try:
        state['execution_started_utc']=datetime.now(timezone.utc).isoformat()
        state['stages']=[]
        update('preparing_data',parameters=config['selected_params'] or config['parameters'])
        archive=prepare(a.manifest,a.archive_dir,a.index,a.raw_caches)
        state['stages'].append(dict(stage=1,name='Historical weekly archive',artifact=str(a.archive_dir/'manifest.csv'),origins=archive['origins']))
        update('predicting')
        from uncertainty import INITIAL_SIZES,LONG_QUANTILES,QUANTILES
        atomic_json(out/'03_uq/policy.json',dict(models=['smoa'],initial_nb_sizes=list(INITIAL_SIZES),draws=5000,seed=13,
            standard_quantiles=QUANTILES.tolist(),coverage_quantiles=LONG_QUANTILES.tolist(),
            policy=a.uq_policy,bridge_mode=a.bridge_mode,
            calibration=('all earlier issued forecasts and final outcomes, including targets after the current origin; retrospective'
                         if a.uq_policy=='figure2_retrospective' else 'matured forecasts with target_end_date <= current origin'),
            warmup='frozen 2020-08-15 snapshot for eligible pre-2020-08-15 origins; retrospective initialization',
            negative_outcomes='Excluded from NB calibration likelihood; retained in raw scoring'))
        command=[sys.executable,'-u',str(ROOT/'experiments/forecasting/run_benchmark.py'),
                 '--manifest',str(a.archive_dir/'manifest.csv'),'--history-mode','as_of',
                 '--synthetic',str(a.synthetic),'--embedding-cache',str(a.embedding_cache),
                 '--k',str(a.k),'--closest',str(a.closest),'--h','4','--smoothing','gam',
                 '--bridge-mode',a.bridge_mode,'--uq-policy',a.uq_policy,
                 '--start-date',origins[0],'--end-date',origins[-1],'--max-origins','0',
                 '--warmup-start-date','1900-01-01','--prehistory-mode','frozen','--frozen-as-of-date',origins[0],
                 '--calibration-truth','final','--score-truth','raw','--negative-truth','clip',
                 '--probabilistic','--coverage-grid','full','--interval-draws','5000','--seed','13','--compress-matches',
                 '--output-dir',str(out/'04_prediction')]
        if a.selection_dir: command += ['--selection-dir',str(a.selection_dir)]
        if a.resume and (out/'04_prediction').exists(): command.append('--resume')
        atomic_json(out/'prediction_command.json',command)
        if a.reuse_predictions_from:
            from reuse_smoa import reuse_predictions
            provenance=reuse_predictions(a.reuse_predictions_from,out/'04_prediction',config)
            state['reused_generation_elapsed_seconds']=provenance['generation_elapsed_seconds']
            state['prediction_execution']='sMOA-only projection of prior issued outputs; no new inference or draws'
        else:
            subprocess.run(command,check=True,cwd=ROOT)
        state['stages'].append(dict(stage=2,name='UQ and prediction',artifact='04_prediction'))
        update('evaluating')
        from evaluate import evaluate_benchmark
        from evaluation_report import format_metrics
        summary=evaluate_benchmark(out/'04_prediction',a.archive_dir/'manifest.csv',out/'05_evaluation')
        if summary['series']!=50 or summary['forecasts_per_model']!=len(origins)*50*4 or set(summary['models'])!={'smoa'}:
            raise ValueError('Evaluation is missing full weekly test targets')
        state['stages'].append(dict(stage=3,name='Evaluation',artifact='05_evaluation/report.md'))
        if a.selection_dir:
            compare_runs(a.paper_run,out,out/'paper_comparison')
        notes=['All 112 weekly origins in the common 50-state, four-target window; no origin subsampling.',
               'Arizona-only later truth dates are outside the common test.',
               'Synthetic RNG, numerical GAM/GCV optimization and NB Monte Carlo differ from R.',
               'Frozen prehistory and final-label calibration follow source conventions; not a strictly real-time initialization.',
               'Upstream duplicate first-origin prehistory record is not replayed; negative NB labels explicitly excluded.',
               'Full external-model score archive and exact R numerical parity are not claimed.']
        notes += [f'Embedding bridge mode: {a.bridge_mode}. Omit uses diff(y) with h+1 future levels, as Figure 2.',
                  f'UQ policy: {a.uq_policy}. Retrospective mode uses future labels of earlier-issued forecasts; not a real-time evaluation.',
                  'Only bridge handling and UQ eligibility were changed on request; main NB initialization sizes remain [20,10,6,0.3].']
        notes += ['Only sMOA is generated, calibrated, scored and plotted.']
        if a.reuse_predictions_from:
            notes += ['Previously issued sMOA values were retained exactly; model-only projection and re-evaluation did not refit or resample.',
                      'Original generation metadata and numerical UQ compatibility checks are recorded in 04_prediction/reuse_provenance.json.']
        if not a.selection_dir: notes.append('Hyperparameters fixed from paper; no BO was run or marked complete for this run.')
        report_path=out/'05_evaluation/report.md'
        report_path.write_text(report_path.read_text()+'\n## Run protocol\n\n'+'\n'.join('- '+n for n in notes)+'\n')
        update('complete',evaluation=summary,elapsed_seconds=time.time()-execution_started,
               finished_utc=datetime.now(timezone.utc).isoformat(),publication_reproduction_complete=False,
               remaining_differences=notes)
        print('\n'+format_metrics(summary),flush=True)
        print(f'Full run complete: {out}',flush=True)
    except BaseException as exc:
        update('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/truth_states/manifest.csv')
    p.add_argument('--archive-dir',type=Path,default=ROOT/'outputs/paper_weekly_archive')
    p.add_argument('--index',type=Path,default=ROOT/'outputs/paper_data/raw/jhu_us_cases_data_links.csv')
    p.add_argument('--raw-caches',type=Path,nargs='+',default=[ROOT/'outputs/paper_data',ROOT/'outputs/epidemic_panel'])
    p.add_argument('--synthetic',type=Path,default=ROOT/'outputs/reproduction_full/01_synthetic/curves.csv')
    p.add_argument('--embedding-cache',type=Path)
    p.add_argument('--selection-dir',type=Path)
    p.add_argument('--wait-for-selection',action='store_true')
    p.add_argument('--paper-run',type=Path,default=ROOT/'outputs/paper_full')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/paper_full')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--reuse-predictions-from',type=Path,help='Completed matching run; retain exact issued sMOA values and re-evaluate without refitting')
    p.add_argument('--bridge-mode',choices=['keep','omit'],default='omit')
    p.add_argument('--uq-policy',choices=['matured','figure2_retrospective'],default='figure2_retrospective')
    p.set_defaults(k=5,closest=4422,smoothing='gam')
    run(p.parse_args())
