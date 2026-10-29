"""Reports and figures made only from already issued, scored forecasts."""
from __future__ import annotations

from datetime import date
import math
import os
from pathlib import Path
import tempfile

import numpy as np

MODELS = ('smoa', 'persistence')
LABELS = {'smoa': 'sMOA', 'persistence': 'Persistence'}
COLORS = {'smoa': '#1768ac', 'persistence': '#c36b18'}
KEYS = ('series', 'forecast_origin', 'target_end_date', 'horizon')


def _number(value):
    return 'N/A' if value is None else f'{value:,.2f}'


def format_metrics(summary):
    """Human-readable MAE/WIS alongside the machine-readable JSON output."""
    lines = [f"Evaluation: {summary['forecasts_per_model']} forecasts/model; raw final truth",
             f"{'Model':<15} {'MAE':>14} {'WIS':>14}"]
    for model in MODELS:
        metrics = summary['models'][model]
        lines.append(f"{LABELS[model]:<15} {_number(metrics['mae']):>14} {_number(metrics.get('wis')):>14}")
    return '\n'.join(lines)


def _plot_evaluation(rows, quantiles, summary, output_dir):
    # Evaluation also runs on headless machines, without a writable home directory.
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'smoa-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import dates as mdates
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import StrMethodFormatter

    horizons = sorted({int(r['horizon']) for r in rows})
    series = rows[0]['series']
    origins = len({r['forecast_origin'] for r in rows})
    artifacts = []

    def save(fig, name):
        for extension in ('png', 'svg'):
            filename = f'{name}.{extension}'
            fig.savefig(output_dir / filename, dpi=170, facecolor='white')
            artifacts.append(filename)
        plt.close(fig)

    style = {'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
             'axes.titleweight': 'bold', 'axes.labelcolor': '#354052',
             'text.color': '#172537', 'axes.edgecolor': '#aab4c2'}
    with plt.rc_context(style):
        columns = min(2, len(horizons))
        nrows = math.ceil(len(horizons) / columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(13, 3.65 * nrows + 1.5), squeeze=False)
        fig.subplots_adjust(top=1 - 1.8 / fig.get_figheight(), bottom=.11, hspace=.65, wspace=.22)
        fig.suptitle(f'{series} | Issued forecasts vs observed COVID-19 cases', fontsize=17, y=.975)
        overall = summary['models']['smoa']
        fig.text(.5, .922, f"sMOA overall MAE {_number(overall['mae'])}  |  WIS {_number(overall.get('wis'))}",
                 ha='center', fontsize=12)
        interval_levels = set()
        for ax, horizon in zip(axes.flat, horizons):
            selected = sorted((r for r in rows if r['model'] == 'smoa' and int(r['horizon']) == horizon),
                              key=lambda r: r['target_end_date'])
            dates = [date.fromisoformat(r['target_end_date']) for r in selected]
            # Each panel has one fixed horizon: do not stitch different forecast
            # origins/horizons into an apparently single issued trajectory.
            for lower, upper, label, alpha in ((.025, .975, '95% predictive interval', .12),
                                               (.25, .75, '50% predictive interval', .25)):
                bounds = []
                for row in selected:
                    qs = quantiles.get(tuple(row[k] for k in KEYS) + ('smoa',), {})
                    lo = next((v for q, v in qs.items() if np.isclose(q, lower)), None)
                    hi = next((v for q, v in qs.items() if np.isclose(q, upper)), None)
                    bounds.append((lo, hi))
                if all(lo is not None and hi is not None for lo, hi in bounds):
                    ax.fill_between(dates, [lo for lo, _ in bounds], [hi for _, hi in bounds],
                                    color=COLORS['smoa'], alpha=alpha, linewidth=0)
                    interval_levels.add(label)
            baseline = sorted((r for r in rows if r['model'] == 'persistence' and int(r['horizon']) == horizon),
                              key=lambda r: r['target_end_date'])
            ax.plot(dates, [r['true_value'] for r in selected], color='#172537', marker='o',
                    markersize=4, linewidth=2.1, zorder=4)
            ax.plot(dates, [r['forecast'] for r in selected], color=COLORS['smoa'], marker='o',
                    markersize=3, linewidth=1.8, zorder=3)
            ax.plot([date.fromisoformat(r['target_end_date']) for r in baseline],
                    [r['forecast'] for r in baseline], color=COLORS['persistence'], linestyle='--', linewidth=1.5)
            metrics = summary['by_horizon'][str(horizon)]['smoa']
            ax.set_title(f"{horizon} week{'s' if horizon != 1 else ''} ahead · {len(selected)} forecasts", loc='left', pad=30)
            ax.text(0, 1.015, f"MAE {_number(metrics['mae'])}   |   WIS {_number(metrics.get('wis'))}",
                    transform=ax.transAxes, fontsize=9, va='bottom')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.set_ylabel('Weekly reported cases')
            ax.set_xlabel('Target week ending')
            ax.grid(axis='y', alpha=.18)
            ax.margins(x=.04)
        for ax in list(axes.flat)[len(horizons):]:
            ax.set_visible(False)
        legend = [Line2D([], [], color='#172537', marker='o', label='Observed (raw final truth)'),
                  Line2D([], [], color=COLORS['smoa'], label='sMOA'),
                  Line2D([], [], color=COLORS['persistence'], linestyle='--', label='Persistence')]
        for label, alpha in (('50% predictive interval', .25), ('95% predictive interval', .12)):
            if label in interval_levels:
                legend.append(Patch(facecolor=COLORS['smoa'], alpha=alpha, label=label))
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .89), ncol=len(legend),
                   fontsize=9, columnspacing=1.4, frameon=False)
        fig.text(.5, .025, f'{origins} origins | {summary["forecasts_per_model"]} forecasts/model | '
                 'Panels separate horizons; shaded bands describe sMOA uncertainty.', ha='center', fontsize=9)
        save(fig, 'forecast_vs_actual')

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        fig.subplots_adjust(top=.77, bottom=.15, wspace=.25)
        fig.suptitle(f'{series} | Forecast error by horizon', fontsize=17, y=.97)
        fig.text(.5, .885, 'Lower is better. Both models use the same issued forecast targets.', ha='center')
        x = np.arange(len(horizons))
        for ax, metric in zip(axes, ('mae', 'wis')):
            available = all(metric in summary['by_horizon'][str(h)][m] for h in horizons for m in MODELS)
            if available:
                for index, model in enumerate(MODELS):
                    values = [summary['by_horizon'][str(h)][model][metric] for h in horizons]
                    bars = ax.bar(x + (index - .5) * .36, values, width=.36,
                                  color=COLORS[model], label=LABELS[model])
                    ax.bar_label(bars, labels=[f'{v:,.0f}' for v in values], padding=4, fontsize=9)
                ax.set_ylim(0, max(ax.get_ylim()[1] * 1.18, 1))
            else:
                ax.text(.5, .5, 'WIS unavailable: no predictive quantiles', ha='center', transform=ax.transAxes)
            ax.set_title(metric.upper(), loc='left')
            ax.set_xticks(x, [str(h) for h in horizons])
            ax.set_xlabel('Forecast horizon (weeks)')
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.grid(axis='y', alpha=.18)
            ax.set_axisbelow(True)
        axes[0].legend(frameon=False)
        save(fig, 'metrics_by_horizon')
    return artifacts


def write_report(rows, quantiles, summary, prediction_dir, truth_path, output_dir, *, create_plots=True):
    """Write a reviewable report; plotting never fits a model or selects a trial."""
    artifacts = _plot_evaluation(rows, quantiles, summary, output_dir) if create_plots else []
    lines = [f"# Evaluation — {rows[0]['series']}", '',
             f"{summary['forecasts_per_model']} forecasts/model, "
             f"{len({r['forecast_origin'] for r in rows})} origins; "
             f"targets {min(r['target_end_date'] for r in rows)} → {max(r['target_end_date'] for r in rows)}.", '',
             'MAE đo sai số tuyệt đối trung bình của dự báo điểm. WIS chấm phân phối dự báo, '
             'cân bằng độ rộng khoảng dự báo và phần phạt khi thực tế nằm ngoài khoảng; cả hai càng thấp càng tốt.', '',
             '| Model | Số forecasts | MAE | WIS |', '| --- | ---: | ---: | ---: |']
    for model in MODELS:
        metrics = summary['models'][model]
        lines.append(f"| {LABELS[model]} | {summary['forecasts_per_model']} | {_number(metrics['mae'])} | {_number(metrics.get('wis'))} |")
    lines.extend(['', 'MAE/WIS tổng hợp là trung bình trên các cặp origin–horizon đã phát hành; '
                  'một target tuần có thể được dự báo ở nhiều horizon. '
                  'Persistence giữ nguyên giá trị quan sát cuối cùng tại origin.', ''])
    if artifacts:
        lines.extend(['## Dự báo và thực tế', '', '![Forecast vs actual](forecast_vs_actual.png)', '',
                      'Đen: COVID thực tế. Xanh: sMOA. Cam nét đứt: persistence. '
                      'Vùng xanh là khoảng dự báo trung tâm 50% và 95% của sMOA nếu có quantiles. '
                      'Mỗi ô giữ một horizon cố định; các điểm liên tiếp đến từ những origin khác nhau.', '',
                      '## MAE và WIS theo horizon', '', '![MAE and WIS](metrics_by_horizon.png)', ''])
    lines.extend(['| Horizon (tuần) | Model | Số forecasts | MAE | WIS |', '| ---: | --- | ---: | ---: | ---: |'])
    for horizon, models in summary['by_horizon'].items():
        for model, metrics in models.items():
            lines.append(f"| {horizon} | {LABELS[model]} | {metrics['n']} | {_number(metrics['mae'])} | {_number(metrics.get('wis'))} |")
    config = {}
    if (prediction_dir / 'metadata.json').exists():
        import json
        config = json.loads((prediction_dir / 'metadata.json').read_text()).get('config', {})
    if config:
        history = 'historical as-of' if config.get('as_of_truth') else 'final truth (không phải vintage lịch sử)'
        lines.extend(['', f"History dùng cho dự báo: **{history}**. "
                      f"UQ calibration truth: `{config.get('calibration_truth', 'unspecified')}`."])
        selection = config.get('selection_protocol', {})
        if selection:
            lines.extend(['', f"BO profile: `{selection.get('profile', 'custom')}`, "
                          f"{selection.get('initial_points', '?')} LHS + {selection.get('iterations', '?')} EI. "
                          'Đồ thị này mô tả lần chạy đã lưu, không chứng minh đã chạy đủ ngân sách bài báo.'])
    def relative(path):
        return os.path.relpath(Path(path).resolve(), output_dir.resolve())
    lines.extend(['', '## Nguồn và quy ước', '',
                  f"- Predictions đã lưu: [predictions.csv](<{relative(prediction_dir / 'predictions.csv')}>).",
                  f"- Dữ liệu COVID thực tế: [truth CSV](<{relative(truth_path)}>).",
                  '- Chấm raw final truth: giữ nguyên cả giá trị 0/âm do chỉnh sửa dữ liệu.',
                  '- WIS = 2 × mean pinball loss tại 7 quantiles [.025, .1, .25, .5, .75, .9, .975], theo helper nguồn.',
                  '- `N/A` nghĩa là chưa có predictive quantiles để tính WIS.',
                  '- Điểm chi tiết: [scores.csv](scores.csv); tổng hợp: [summary.json](summary.json).', ''])
    (output_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return ['report.md', *artifacts]
