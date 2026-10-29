"""Replay the paper's Figure 2 with historical JHU truth, Python GAM and Hub data."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import date, timedelta, datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for directory in ['experiments', 'experiments/forecasting']:
    sys.path.insert(0, str(ROOT / directory))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'smoa-experiment0-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from model import load_library
from neighbors import ExactNeighbors
from run_backtest import parser as forecast_parser, run_experiment
from evaluate import evaluate_outputs
import uncertainty
from runtime import atomic_json, file_hash, acquire_run_lock

PAPER = 'https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013203'
STATES = ('California', 'Florida', 'New York', 'Texas')
# figure_2.R tests origin+7 against this vector. These are FIRST TARGET dates,
# not origins or Monday submission dates; mixing them shifts forecasts a week.
FIRST_TARGETS = ('2020-03-21', '2020-04-18', '2020-05-23', '2020-06-20', '2020-07-18', '2020-08-15')
PUBLISHED = dict(num_curves=18387, k=5, closest=4422)
FIGURE2_INITIAL_SIZES = (20., 10., 6., 6.)


def schedule(states=STATES, targets=FIRST_TARGETS):
    return pd.DataFrame([dict(series=state,forecast_origin=(date.fromisoformat(target)-timedelta(days=7)).isoformat(),
        forecast_date=(date.fromisoformat(target)-timedelta(days=5)).isoformat(),first_target_end_date=target,
        last_target_end_date=(date.fromisoformat(target)+timedelta(days=21)).isoformat()) for state in states for target in targets])


@contextmanager
def figure2_initialization():
    # The original figure script uses size 6 for h4; main run_smoa uses .3.
    # Scope this source-specific setting to this separate experiment process.
    previous=uncertainty.INITIAL_SIZES
    uncertainty.INITIAL_SIZES=FIGURE2_INITIAL_SIZES
    try:
        yield
    finally:
        uncertainty.INITIAL_SIZES=previous


def figure2_embeddings(library, k=5, h=4):
    x, future, ids, steps=load_library(library,k,h+1)
    return x, future[:,1:], ids, steps  # diff(y), omitting bridge, as figure_2.R


def select_examples(prediction_dir, selected_dir, origins):
    predictions=pd.read_csv(prediction_dir/'predictions.csv',dtype={'forecast_origin':str})
    chosen=predictions.loc[predictions.forecast_origin.isin(origins)]
    keys=['series','forecast_origin','horizon']
    if chosen.duplicated(keys).any() or len(chosen)!=len(origins)*4:
        raise ValueError('Missing or duplicated predeclared origin/horizon examples')
    if set(chosen.horizon)!={1,2,3,4} or set(chosen.forecast_origin)!=set(origins):
        raise ValueError('Examples do not cover the declared origins and horizons')
    expected=pd.to_datetime(chosen.forecast_origin)+pd.to_timedelta(7*chosen.horizon,unit='D')
    if not (expected.to_numpy()==pd.to_datetime(chosen.target_end_date).to_numpy()).all():
        raise ValueError('Target date is not origin + 7*horizon')
    selected_dir.mkdir(parents=True,exist_ok=True)
    chosen.to_csv(selected_dir/'predictions.csv',index=False)
    qs=pd.read_csv(prediction_dir/'quantiles.csv',dtype={'forecast_origin':str})
    qs.loc[qs.forecast_origin.isin(origins)].to_csv(selected_dir/'quantiles.csv',index=False)
    metadata=json.loads((prediction_dir/'metadata.json').read_text())
    metadata['example_selection']=dict(source=str(prediction_dir),origins=sorted(origins),selection='Prespecified from figure_2.R, not selected by error')
    atomic_json(selected_dir/'metadata.json',metadata)
    return chosen


def run_early(full_run, manifest_path, data, out):
    """Replay figure_2.R, including its explicit retrospective UQ convention."""
    from model import forecast
    from prediction import predict_origin
    from smoothing import thin_plate_gcv
    manifest=pd.read_csv(manifest_path,keep_default_na=False).set_index('series')
    plan=schedule();plan.to_csv(out/'schedule.csv',index=False)
    library=full_run/'01_synthetic/curves.csv'
    generation=json.loads((full_run/'01_synthetic/metadata.json').read_text())
    if generation['attempted_total']!=55161 or generation['num_curves_per_family']!=18387:
        raise ValueError('Published-size reference library required')
    if file_hash(library)!=generation['curves_sha256']: raise ValueError('Library checksum mismatch')
    if not (data/'provenance.json').exists():
        raise ValueError('Run experiments/data/fetch_paper_data.py first; no final-truth fallback')
    protocol=dict(name='figure2_source_v2',published_params=PUBLISHED,seed=13,
        history='Frozen JHU issue 2020-08-16, labelled 2020-08-15 by LANL',
        smoothing='Gaussian thin plate regression spline; k=max(3,round(n/2)); GCV',
        embedding='figure_2.R diff(y), omit bridge',initial_sizes=FIGURE2_INITIAL_SIZES,
        calibration='figure2_retrospective: all earlier issued forecasts and final labels, even unmatured targets',
        causal_sensitivity='same point forecasts, matured final labels only; not real-time vintages before August',
        warmup='No pre-Feb29 warmup: Jan25-Feb22 has 5 rows, failing source >=6 condition',
        truth='LANL tracked state_truths; raw final labels for MAE/WIS',
        library_sha256=generation['curves_sha256'],data_provenance_sha256=file_hash(data/'provenance.json'),
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in
            [Path(__file__),ROOT/'experiments/runtime.py',*[ROOT/'experiments/forecasting'/name for name in
             ['model.py','smoothing.py','prediction.py','neighbors.py','uncertainty.py']]]},
        input_sha256={str(p):file_hash(p) for p in [data/'hub_forecasts.csv',
            *[data/f'{state}_{vintage}.csv' for state in STATES for vintage in
              ['2020-08-15','2020-08-22','2020-08-29','2020-09-05']],
            *[(manifest_path.parent/manifest.loc[state,'truth']).resolve() for state in STATES]]})
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Protocol or inputs changed; use a fresh output directory')
    atomic_json(out/'protocol.json',protocol)
    protocol_hash=file_hash(out/'protocol.json')
    def complete(state):
        path=out/'early/prediction'/state/'metadata.json'
        if not path.exists(): return False
        saved=json.loads(path.read_text())
        return (saved.get('protocol_sha256')==protocol_hash and bool(saved.get('artifacts_sha256')) and
                all((out/name).is_file() and file_hash(out/name)==digest for name,digest in saved['artifacts_sha256'].items()))
    pending=[state for state in STATES if not complete(state)]
    x=y=index=None
    if pending:
        print('Loading full published-size Figure 2 library...',flush=True)
        x,y,ids,steps=figure2_embeddings(library)
        del ids,steps
        print(f'Building exact L1 index: {len(x):,} windows',flush=True)
        index=ExactNeighbors(x)
    with figure2_initialization():
        for state in pending:
            snapshot=pd.read_csv(data/f'{state}_2020-08-15.csv',parse_dates=['target_end_date'])
            final=pd.read_csv(manifest_path.parent/manifest.loc[state,'truth'],parse_dates=['target_end_date'])
            if snapshot.target_end_date.duplicated().any() or not snapshot.target_end_date.equals(final.target_end_date.iloc[:len(snapshot)]):
                raise ValueError('Snapshot and final truth calendars differ')
            dates=snapshot.target_end_date.dt.date.tolist()
            values=np.maximum(snapshot.value.to_numpy(float),1)
            labels=final.set_index('target_end_date').value
            issued=[];predictions=[];quantiles=[];calibration=[];ablation=[];neighbors=[]
            for i in range(5,len(snapshot)):
                origin=dates[i]
                if origin>=date(2020,8,15): break
                history=values[:i+1]
                source_plan=uncertainty.calibration_plan(issued,str(origin),policy='figure2_retrospective')
                output,indices,distances,probabilities=predict_origin(history,x,y,origin,source_plan,
                    smoothing='gam',neighbor_index=index)
                # A second UQ policy uses exactly the same points and RNG keys.
                causal_plan=uncertainty.calibration_plan(issued,str(origin))
                current=[]
                fitted,info=thin_plate_gcv(history,return_info=True)
                for j in range(1,5):
                    target=origin+timedelta(days=7*j)
                    point=next(r for r in output if r['horizon']==j and r['field']=='forecast')
                    row=dict(series=state,forecast_origin=str(origin),forecast_date=str(origin+timedelta(days=2)),
                        target_end_date=str(target),horizon=j,forecast=point['point'],baseline=float(history[-1]))
                    predictions.append(row)
                    current.append(dict(**row,true_value=float(labels.loc[pd.Timestamp(target)])))
                    for result in [r for r in output if r['horizon']==j]:
                        model='smoa' if result['field']=='forecast' else 'persistence'
                        quantiles.extend(dict(**{k:row[k] for k in ['series','forecast_origin','target_end_date','horizon']},
                            model=model,quantile=float(q),value=float(v)) for q,v in zip(probabilities,result['quantiles']))
                        calibration.append(dict(series=state,forecast_origin=str(origin),horizon=j,model=model,
                            size=result['size'],n=result['count'],method=result['method'],policy='figure2_retrospective',
                            **{k:info[k] for k in ['basis_size','edf','gcv']}))
                    fit=causal_plan[('forecast',j)]
                    rng=np.random.default_rng(np.random.SeedSequence([13,origin.toordinal(),j,0]))
                    causal=uncertainty.interval_quantiles(point['point'],fit['size'],rng,5000,fit['method']=='initial',probabilities)
                    quantiles.extend(dict(**{k:row[k] for k in ['series','forecast_origin','target_end_date','horizon']},
                        model='smoa_matured_uq',quantile=float(q),value=float(v)) for q,v in zip(probabilities,causal))
                issued.extend(current)
                if str(origin) in set(plan.forecast_origin):
                    for rank,(idx,distance) in enumerate(zip(indices[:10],distances[:10]),1):
                        neighbors.append(dict(forecast_origin=str(origin),rank=rank,embedding_row=int(idx),distance=float(distance)))
                    # Predeclared ablation, same library and dates; no parameter search.
                    for variant,history_input,smoother in [
                        ('historical_gam',history,'gam'),('historical_spline',history,'spline'),
                        ('final_gam',np.maximum(final.value.iloc[:i+1].to_numpy(float),1),'gam'),
                        ('final_spline',np.maximum(final.value.iloc[:i+1].to_numpy(float),1),'spline')]:
                        point,_,_=forecast(history_input,x,y,smoothing=smoother,neighbor_index=index)
                        for j,value in enumerate(point,1):
                            actual=float(labels.loc[pd.Timestamp(origin+timedelta(days=7*j))])
                            ablation.append(dict(series=state,forecast_origin=str(origin),horizon=j,
                                variant=variant,forecast=float(value),actual=actual,abs_error=abs(value-actual)))
            directory=out/'early/prediction'/state;directory.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(predictions).to_csv(directory/'predictions.csv',index=False)
            pd.DataFrame(quantiles).to_csv(directory/'quantiles.csv',index=False)
            pd.DataFrame(calibration).to_csv(directory/'calibration.csv',index=False)
            pd.DataFrame(ablation).to_csv(directory/'ablation.csv',index=False)
            pd.DataFrame(neighbors).to_csv(directory/'neighbors_top10.csv',index=False)
            atomic_json(directory/'metadata.json',dict(protocol='figure2_source_v2',series=state,
                history_mode='frozen_as_of',embedding_rows=len(x),protocol_sha256=file_hash(out/'protocol.json')))
            select_examples(directory,out/'early/selected'/state,set(plan.loc[plan.series==state,'forecast_origin']))
            metadata=json.loads((directory/'metadata.json').read_text())
            artifacts=[*directory.glob('*.csv'),*(out/'early/selected'/state).glob('*')]
            metadata['artifacts_sha256']={str(path.relative_to(out)):file_hash(path) for path in artifacts}
            atomic_json(directory/'metadata.json',metadata)
            print(f'{state}: historical data + GAM + source UQ complete',flush=True)
    return score_comparison(manifest_path,data,out)


def score_comparison(manifest_path,data,out):
    manifest=pd.read_csv(manifest_path,keep_default_na=False).set_index('series')
    rows=[];fans=[]
    def score(series,origin,target,horizon,model,point,q):
        actual=float(truth.loc[target])
        standard=np.array([q[round(float(p),6)] for p in uncertainty.QUANTILES])
        result=uncertainty.scores(standard,actual)
        rows.append(dict(series=series,forecast_origin=origin,target_end_date=target,horizon=horizon,
            model=model,forecast=point,actual=actual,abs_error=abs(point-actual),**result))
        fans.append(dict(series=series,forecast_origin=origin,target_end_date=target,horizon=horizon,model=model,
            forecast=point,lower95=q[.025],upper95=q[.975],lower50=q[.25],upper50=q[.75]))
    for state in STATES:
        truth=pd.read_csv(manifest_path.parent/manifest.loc[state,'truth']).set_index('target_end_date').value
        pred=pd.read_csv(out/'early/selected'/state/'predictions.csv')
        qs=pd.read_csv(out/'early/selected'/state/'quantiles.csv')
        for row in pred.itertuples():
            for model in ['smoa','persistence','smoa_matured_uq']:
                q=qs.loc[(qs.forecast_origin==row.forecast_origin)&(qs.horizon==row.horizon)&(qs.model==model)]
                lookup={round(float(p),6):float(v) for p,v in zip(q['quantile'],q.value)}
                score(state,row.forecast_origin,row.target_end_date,row.horizon,model,
                      row.baseline if model=='persistence' else row.forecast,lookup)
    hub=pd.read_csv(data/'hub_forecasts.csv')
    for (state,origin,target,horizon,model),group in hub.groupby(['series','forecast_origin','target_end_date','horizon','model']):
        truth=pd.read_csv(manifest_path.parent/manifest.loc[state,'truth']).set_index('target_end_date').value
        point=group.loc[group.type=='point','value']
        q=group.loc[group.type=='quantile']
        lookup={round(float(p),6):float(v) for p,v in zip(q['quantile'],q.value)}
        if len(point)!=1 or any(round(float(p),6) not in lookup for p in uncertainty.QUANTILES):
            raise ValueError('Official forecast lacks point or required seven quantiles')
        score(state,origin,target,int(horizon),model,float(point.iloc[0]),lookup)
    scores=pd.DataFrame(rows);scores.to_csv(out/'scores.csv',index=False)
    pd.DataFrame(fans).to_csv(out/'plot_data.csv',index=False)
    metrics=scores.groupby('model',as_index=False).agg(n=('abs_error','size'),mae=('abs_error','mean'),
        wis=('wis','mean'),coverage_95=('coverage_95','mean'),width_95=('width_95','mean'))
    metrics.to_csv(out/'metrics.csv',index=False)
    scores.groupby(['model','horizon'],as_index=False).agg(n=('abs_error','size'),mae=('abs_error','mean'),
        wis=('wis','mean'),coverage_95=('coverage_95','mean')).to_csv(out/'metrics_by_horizon.csv',index=False)
    # Intersection of series/origin/horizon keys, never compare unmatched means.
    keys=['series','forecast_origin','target_end_date','horizon']
    model_names=['smoa','COVIDhub-baseline','COVIDhub-4_week_ensemble']
    common=None
    for model in model_names:
        idx=pd.MultiIndex.from_frame(scores.loc[scores.model==model,keys])
        common=idx if common is None else common.intersection(idx)
    paired=scores.loc[scores.model.isin(model_names)&pd.MultiIndex.from_frame(scores[keys]).isin(common)]
    paired.to_csv(out/'paired_scores.csv',index=False)
    paired.groupby('model',as_index=False).agg(n=('wis','size'),mae=('abs_error','mean'),wis=('wis','mean')).to_csv(out/'paired_metrics.csv',index=False)
    ablation=pd.concat([pd.read_csv(out/'early/prediction'/state/'ablation.csv') for state in STATES])
    ablation.to_csv(out/'ablation_predictions.csv',index=False)
    ablation.groupby('variant',as_index=False).agg(n=('abs_error','size'),mae=('abs_error','mean')).to_csv(out/'ablation.csv',index=False)
    return scores,metrics


def plot_examples(manifest_path,data,out):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    manifest=pd.read_csv(manifest_path,keep_default_na=False).set_index('series')
    fans=pd.read_csv(out/'plot_data.csv',parse_dates=['target_end_date'])
    models=['COVIDhub-4_week_ensemble','COVIDhub-baseline','smoa']
    fans=fans.loc[fans.model.isin(models)]
    fig,axes=plt.subplots(3,4,figsize=(13,12.3),sharex=True,sharey=True,squeeze=False)
    fig.subplots_adjust(top=.92,bottom=.13,left=.075,right=.94,hspace=.02,wspace=.02)
    maximum=max(float(fans.upper95.max()),100000)
    minimum=min(0.,float(fans.lower95.min()))
    top=maximum+.05*(maximum-minimum)
    bottom=minimum-.05*(maximum-minimum)
    for col,state in enumerate(STATES):
        truth=pd.read_csv(manifest_path.parent/manifest.loc[state,'truth'],parse_dates=['target_end_date'])
        snapshot=pd.read_csv(data/f'{state}_2020-08-15.csv',parse_dates=['target_end_date'])
        # The source orange line is sampled at selected origins, followed by
        # Aug15/Aug22/Aug29/Sep5 (not a line through every weekly datum).
        observed_dates=set(pd.to_datetime(schedule().forecast_origin))|set(pd.to_datetime(['2020-08-15','2020-08-22','2020-08-29','2020-09-05']))
        tail=[]
        for vintage in ['2020-08-22','2020-08-29','2020-09-05']:
            later=pd.read_csv(data/f'{state}_{vintage}.csv',parse_dates=['target_end_date'])
            tail.append(later.loc[later.target_end_date==pd.Timestamp(vintage),['target_end_date','value']])
        observed=pd.concat([snapshot[['target_end_date','value']],*tail])
        observed=observed.loc[observed.target_end_date.isin(observed_dates)]
        for row,model in enumerate(models):
            ax=axes[row,col]
            ax.plot(observed.target_end_date,observed.value,color='#ff9000',lw=1.6,zorder=3)
            groups=fans.loc[(fans.series==state)&(fans.model==model)]
            for _,group in groups.groupby('forecast_origin'):
                group=group.sort_values('target_end_date')
                ax.fill_between(group.target_end_date,group.lower95,group.upper95,color='#dff4fb',linewidth=0)
                ax.fill_between(group.target_end_date,group.lower50,group.upper50,color='#a1e4ed',linewidth=0)
                ax.plot(group.target_end_date,group.forecast,color='black',lw=1.7)
            if row==0:
                ax.add_patch(plt.Rectangle((0,1.002),1,.085,transform=ax.transAxes,
                    facecolor='#dce4e4',edgecolor='#566060',clip_on=False))
                ax.text(.5,1.042,state,transform=ax.transAxes,ha='center',va='center',fontsize=12)
            if col==3:
                ax.add_patch(plt.Rectangle((1.002,0),.105,1,transform=ax.transAxes,
                    facecolor='#dce4e4',edgecolor='#566060',clip_on=False))
                ax.text(1.055,.5,'sMOA (Python)' if model=='smoa' else model,transform=ax.transAxes,
                    rotation=-90,ha='center',va='center',fontsize=11)
            ax.set_ylim(bottom,top);ax.set_xlim(pd.Timestamp('2020-03-14'),pd.Timestamp('2020-09-05'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[5,7,9]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.grid(color='#e9eeee',linewidth=.7);ax.set_axisbelow(True)
            ax.tick_params(labelsize=9)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100000))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_:f'{x:.0e}'))
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50000))
            ax.grid(which='minor',color='#f1f4f4',linewidth=.5)
            if row==2: plt.setp(ax.get_xticklabels(),rotation=20,ha='right')
    fig.supylabel('Weekly Incident COVID Cases',fontsize=13,x=.015)
    fig.supxlabel('Forecast target date',fontsize=13,y=.075)
    fig.suptitle('Figure 2 reproduction | historical JHU + published parameters + Python GAM',fontsize=13,y=.98)
    fig.legend(handles=[Line2D([],[],color='#ff9000',label='Reported data'),Line2D([],[],color='black',label='Point forecast'),
        Patch(facecolor='#dff4fb',label='95%'),Patch(facecolor='#a1e4ed',label='50%')],loc='lower center',ncol=4,frameon=False)
    for ext in ['png','svg']: fig.savefig(out/f'figure_2_python.{ext}',dpi=160,facecolor='white')
    plt.close(fig)


def write_report(full_run,data,out,metrics):
    def table(frame,columns):
        result=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        for _,r in frame.iterrows():
            result.append('| '+' | '.join(f'{r[c]:,.2f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+' |')
        return result
    paired=pd.read_csv(out/'paired_metrics.csv');ablation=pd.read_csv(out/'ablation.csv')
    summary=dict(status='complete',full_paper_reproduction=False,protocol='figure2_source_v2',
        metrics=metrics.to_dict('records'),paired_metrics=paired.to_dict('records'),ablation=ablation.to_dict('records'),
        remaining=['NumPy RNG versus R RNG; original generated library/forecast vectors not published',
            'Python thin plate/GCV optimizer not yet numerically cross-validated against mgcv fitted-value fixtures',
            'Full 50-state historical-vintage experiment and original all-model ranking not rerun in this protocol'],
        full_run_legacy=str(full_run),paper_url=PAPER)
    atomic_json(out/'summary.json',summary)
    lines=['# Experiment 0 — source-aligned Figure 2','',
        f'Paper: [Murph et al.]({PAPER}); [README nguồn](https://github.com/lanl/precog/tree/main/smoa).','',
        '## Dữ liệu thực sự đã lấy','',
        '- JHU issue 2020-08-16, SHA cố định trong danh mục `reichlab/covidData`; gắn nhãn snapshot 2020-08-15 như `get_data_as_of.R`.',
        '- Tổng county theo bang → sai phân cumulative → tổng tuần Chủ nhật–Thứ bảy; bỏ tuần chưa đủ. Không thay historical bằng final truth.',
        '- Forecast gốc `COVIDhub-baseline` và `COVIDhub-4_week_ensemble`, đúng Monday submission và incident-case targets. Nguồn thiếu forecast thì giữ thiếu.',
        '- Scoring truth vẫn là state_truths do LANL cung cấp, đọc bằng Python; provenance/URL/SHA ở `outputs/paper_data/provenance.json`.','',
        '## Sửa phương pháp','',
        '- Smoothing dùng thin plate regression spline với rank round(n/2), affine null space, curvature penalty và GCV; bỏ thay thế bằng natural smoothing spline.',
        '- Published reference: 18.387 attempts/family, 55.161 attempts, 54.081 accepted; k=5, closest=4.422. Không lấy BO winner khác tham số công bố.',
        '- Embedding bỏ bridge và NB sizes [20,10,6,6] theo nhánh Figure 2. Start Feb29, không thêm warmup khi source condition không thỏa.',
        '- Figure 2 gốc cập nhật NB bằng tất cả forecast đã phát trước đó và final outcomes, kể cả target chưa tới. Chế độ `figure2_retrospective` tái hiện quy tắc này; đây là đánh giá hồi cứu, có sử dụng future labels trong UQ.',
        '- Có đối chứng `smoa_matured_uq`: chỉ đổi UQ sang matured targets; giữ nguyên point forecast. Default forecasting ngoài experiment vẫn dùng matured.',
        '- Bố cục 3 hàng model × 4 cột bang, chung trục y, đường thực tế màu cam, forecast màu đen, bands 50/95 xanh theo hình gốc.',
        '- GCV chọn theo history, không chọn smoother/params theo lỗi COVID. Các ablation đều được công bố dù điểm tốt hay xấu.','',
        '## MAE/WIS — 7 quantiles giống Figure 2','',
        'Bảng dưới có số forecast khác nhau do COVIDHub chưa cung cấp case forecasts ở nhiều ngày early. **Không dùng bảng này để xếp hạng model.**','',
        *table(metrics,['model','n','mae','wis','coverage_95']), '',
        '## So sánh đúng giao các target có đủ 3 mô hình','',
        *table(paired,['model','n','mae','wis']),'',
        '## Ablation dữ liệu và smoothing — cùng 96 targets','',
        *table(ablation,['variant','n','mae']),'',
        '## Phạm vi kết luận','',
        'Figure 2 hiện được dựng bằng dữ liệu lịch sử thật và cùng cấu trúc thuật toán/bố cục. Chưa có vector sMOA gốc để tính sai khác số trực tiếp với R; không khẳng định đã khớp tuyệt đối hay đạt calibration/ranking Figure 3–5.',
        'Python dùng NumPy RNG/LSODA, library số khác R; thin plate/GCV cùng mô hình toán nhưng chưa có fixture fitted values của mgcv để xác nhận numerical parity.',
        'Run 50 bang/104 BO evaluations cũ dùng final truth và legacy spline; không đổi nhãn thành kết quả mới. BO/forecast mặc định mới dùng GAM; muốn full run đúng protocol phải chạy thư mục mới.',
        'Source Figure 2 ở commit trước publication `2018f2e` (2025-05-22) có cùng các lựa chọn trên. Script có đối số helper `current_end_date` không tồn tại, nhưng nhánh đó không chạy vì chỉ có 5 tuần trước Feb28.', '',
        '![Python Figure 2](figure_2_python.png)','',
        'Kiểm tra dữ liệu bảng: `scores.csv`, `paired_scores.csv`, `plot_data.csv`, `ablation_predictions.csv`; protocol và input/source hashes: `protocol.json`.','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-run',type=Path,default=ROOT/'outputs/reproduction_full')
    parser.add_argument('--manifest',type=Path,default=ROOT/'outputs/truth_states/manifest.csv')
    parser.add_argument('--paper-data',type=Path,default=ROOT/'outputs/paper_data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/experiment_0_source')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    run_lock=acquire_run_lock(args.output_dir/'run.lock')
    _,metrics=run_early(args.full_run,args.manifest,args.paper_data,args.output_dir)
    plot_examples(args.manifest,args.paper_data,args.output_dir)
    write_report(args.full_run,args.paper_data,args.output_dir,metrics)
    print(metrics.to_string(index=False),flush=True)
    print('Report:',args.output_dir/'report.md',flush=True)


if __name__=='__main__': main()

