"""Evaluation stage: join issued predictions to raw final truth after prediction."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from uncertainty import QUANTILES,scores,coverage_rows
from evaluation_report import format_metrics,write_report

KEYS=['series','forecast_origin','target_end_date','horizon']


def evaluate_outputs(directory,truth_path,output_dir,*,create_plots=True):
    from run_backtest import load_truth
    dates,values=load_truth(truth_path,'clip',floor_values=False)
    truth={d.isoformat():float(v) for d,v in zip(dates,values)}
    with (directory/'predictions.csv').open() as f: predictions=list(csv.DictReader(f))
    if not predictions: raise ValueError('Empty predictions')
    if len({r['series'] for r in predictions})!=1: raise ValueError('Evaluation requires one series per truth CSV')
    if len({tuple(r[k] for k in KEYS) for r in predictions})!=len(predictions): raise ValueError('Duplicate forecast keys')
    quantiles={}
    if (directory/'quantiles.csv').exists():
        with (directory/'quantiles.csv').open() as f:
            for r in csv.DictReader(f):
                key=tuple(r[k] for k in KEYS)+(r['model'],)
                q=float(r['quantile'])
                if q in quantiles.setdefault(key,{}): raise ValueError('Duplicate forecast quantile')
                quantiles[key][q]=float(r['value'])
    rows=[];coverage=[]
    for r in predictions:
        if r['target_end_date'] not in truth: raise ValueError('Missing scoring truth')
        for model,field in [('smoa','forecast'),('persistence','baseline')]:
            row={k:r[k] for k in KEYS};row.update(model=model,forecast=float(r[field]),true_value=truth[r['target_end_date']])
            row['abs_error']=abs(row['forecast']-row['true_value'])
            qs=quantiles.get(tuple(r[k] for k in KEYS)+(model,))
            if quantiles and not qs: raise ValueError('Missing predictive quantiles for forecast/model')
            if qs:
                vals=[]
                for q in QUANTILES:
                    matches=[value for key,value in qs.items() if np.isclose(key,q)]
                    if len(matches)!=1: raise ValueError('Missing standard scoring quantile')
                    vals.append(matches[0])
                row.update(scores(vals,row['true_value']))
                probabilities=np.array(sorted(qs));values=np.array([qs[q] for q in probabilities])
                coverage.extend(dict(**{k:r[k] for k in KEYS},model=model,**c) for c in coverage_rows(values,row['true_value'],probabilities))
            rows.append(row)
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,table in [('scores.csv',rows),('coverage.csv',coverage)]:
        if table:
            with (output_dir/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    metrics=['abs_error']+(['wis','sharpness','overprediction','underprediction','coverage_50','coverage_80','coverage_95'] if quantiles else [])
    summary=dict(scoring_truth='raw final truth',forecasts_per_model=len(predictions),
        models={model:{metric:float(np.mean([r[metric] for r in rows if r['model']==model]))
            for metric in metrics} for model in ['smoa','persistence']})
    summary['coverage_by_level']={model:{str(level):float(np.mean([r['covered'] for r in coverage if r['model']==model and r['level']==level]))
        for level in sorted({r['level'] for r in coverage})} for model in ['smoa','persistence']}
    # Keep abs_error for existing consumers; mae makes the aggregate meaning explicit.
    for model in summary['models']:
        summary['models'][model]['mae']=summary['models'][model]['abs_error']
    summary['by_horizon']={}
    for horizon in sorted({int(r['horizon']) for r in rows}):
        summary['by_horizon'][str(horizon)]={}
        for model in ['smoa','persistence']:
            selected=[r for r in rows if int(r['horizon'])==horizon and r['model']==model]
            metrics=dict(n=len(selected),mae=float(np.mean([r['abs_error'] for r in selected])))
            if quantiles: metrics['wis']=float(np.mean([r['wis'] for r in selected]))
            summary['by_horizon'][str(horizon)][model]=metrics
    summary['artifacts']=write_report(rows,quantiles,summary,directory,truth_path,output_dir,create_plots=create_plots)
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def evaluate_benchmark(directory,manifest_path,output_dir,*,create_plots=True):
    """Evaluate every state separately and pool scores on the same issued rows."""
    from runtime import atomic_json
    from evaluation_report import LABELS
    with manifest_path.open() as handle:
        entries={row['series']:row for row in csv.DictReader(handle)}
    with (directory/'series_metrics.csv').open() as handle:
        states=list(csv.DictReader(handle))
    if len(states)!=len(entries) or {row['series'] for row in states}!=set(entries):
        raise ValueError('Cannot evaluate an incomplete or mismatched state benchmark')
    output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for row in states:
        subdir=row['output_subdir']
        result=evaluate_outputs(directory/subdir,(manifest_path.parent/entries[row['series']]['truth']).resolve(),
            output_dir/subdir,create_plots=create_plots)
        results.append(dict(series=row['series'],output_subdir=subdir,summary=result))
    # Merge detailed CSVs without retaining all coverage records in memory.
    for name in ['scores.csv','coverage.csv']:
        with (output_dir/name).open('w',newline='') as handle:
            writer=None
            for row in results:
                source=output_dir/row['output_subdir']/name
                if not source.exists(): continue
                with source.open() as input_file:
                    reader=csv.DictReader(input_file)
                    if writer is None:
                        writer=csv.DictWriter(handle,fieldnames=reader.fieldnames,lineterminator='\n');writer.writeheader()
                    writer.writerows(reader)
    total=sum(row['summary']['forecasts_per_model'] for row in results)
    models={model:{metric:float(sum(row['summary']['models'][model][metric]*row['summary']['forecasts_per_model']
        for row in results)/total) for metric in results[0]['summary']['models'][model]} for model in ['smoa','persistence']}
    summary=dict(scoring_truth='raw final truth',series=len(results),forecasts_per_model=total,models=models,by_horizon={})
    horizons=sorted({h for row in results for h in row['summary']['by_horizon']},key=int)
    for h in horizons:
        summary['by_horizon'][h]={}
        for model in models:
            groups=[row['summary']['by_horizon'][h][model] for row in results if h in row['summary']['by_horizon']]
            n=sum(group['n'] for group in groups)
            summary['by_horizon'][h][model]=dict(n=n,**{
                metric:float(sum(group[metric]*group['n'] for group in groups)/n)
                for metric in groups[0] if metric!='n'})
    table=[]
    lines=['# Evaluation — full state benchmark','',f"{len(results)} bang; {total:,} forecasts/model.",'',
        '| Model | MAE | WIS |','| --- | ---: | ---: |']
    for model,metrics in models.items():
        wis='N/A' if 'wis' not in metrics else f"{metrics['wis']:,.2f}"
        lines.append(f"| {LABELS[model]} | {metrics['mae']:,.2f} | {wis} |")
    lines.extend(['',
        'Điểm gộp có trọng số theo số forecast từng bang; không lấy trung bình đều các MAE/WIS của bang khác quy mô.',
        'Mỗi báo cáo bang có biểu đồ dự báo/thực tế, khoảng dự báo và MAE/WIS theo horizon.', '',
        '| Bang | Model | Forecasts | MAE | WIS | Báo cáo và biểu đồ |',
        '| --- | --- | ---: | ---: | ---: | --- |'])
    for result in results:
        for model,metrics in result['summary']['models'].items():
            row=dict(series=result['series'],model=model,n=result['summary']['forecasts_per_model'],
                mae=metrics['mae'],wis=metrics.get('wis'),report=f"{result['output_subdir']}/report.md")
            table.append(row)
            wis='N/A' if row['wis'] is None else f"{row['wis']:,.2f}"
            lines.append(f"| {row['series']} | {LABELS[model]} | {row['n']} | {row['mae']:,.2f} | {wis} | [Xem]({row['report']}) |")
    with (output_dir/'series_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    lines.extend(['', 'Dùng raw final truth để chấm điểm. Protocol history/calibration và ngân sách BO được ghi trong báo cáo từng bang.',
                  'Các điểm số này chỉ được tạo khi BO và prediction tương ứng đã chạy xong.', ''])
    (output_dir/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    summary['artifacts']=['report.md','scores.csv','coverage.csv','series_metrics.csv',
        *[f"{result['output_subdir']}/report.md" for result in results]]
    atomic_json(output_dir/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-dir',type=Path,required=True);p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--no-plots',action='store_true',help='Write metrics/report without Matplotlib figures')
    a=p.parse_args()
    result=evaluate_outputs(a.prediction_dir,a.truth,a.output_dir,create_plots=not a.no_plots)
    print(json.dumps(result,indent=2))
    print('\n'+format_metrics(result))
    print(f"Report: {a.output_dir/'report.md'}")


if __name__=='__main__': main()
