"""Post-selection paired test of the observed BO winner and published parameters.

The fresh test namespace and cap are declared before BO starts. No test score
is returned to BO. Paper training uses the winner's training seed namespace:
smaller banks are prefixes by attempted family/index, not accepted-curve count.
"""
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
from objective import FAMILIES, generate_bank, embeddings, evaluate
from bank import save_packed, write_audit
from runtime import atomic_json, file_hash

PAPER_PARAMETERS = dict(num_curves=18387, k=5, closest=4422)
COMPARISON_PROTOCOL = dict(
    paper_params=dict(PAPER_PARAMETERS),
    test_role=2, test_trial=1, test_attempts_per_family=30, test_curve_cap=50,
    shuffle_namespace='[root_seed,4,1]',
    common_origins='all origins from max(winner_k,paper_k)+1 known observations',
    training='same role=0 and generation trial; common family/index seed prefixes',
    metric='pooled MAE on identical curve/origin/horizon targets',
    uncertainty='95% percentile bootstrap of paired delta MAE, resampling whole test curves',
    bootstrap_draws=10000,
)
KEYS = ['curve_id', 'family', 'known_steps', 'horizon']


def paper_bank(winner_bank, winner_audit, winner_n, paper_n, seed, generation_trial, workers):
    """Retain shared random training curves; replay/extend only if more are needed."""
    expected_keys = {(f, j) for f in FAMILIES for j in range(winner_n)}
    actual_keys = {(r['family'], r['index']) for r in winner_audit}
    if actual_keys != expected_keys or len(winner_audit) != len(expected_keys):
        raise ValueError('Selected training audit does not match attempted family/index grid')
    for row in winner_audit:
        key = [seed, 0, generation_trial, FAMILIES.index(row['family']), row['index']]
        if row['role'] != 0 or row['seed_key'] != key:
            raise ValueError('Unexpected winner training seed namespace')
    accepted = [r for r in winner_audit if r['accepted']]
    if len(accepted) != len(winner_bank):
        raise ValueError('Selected training bank/audit length differs')
    if winner_n >= paper_n:
        curves = [curve for curve, row in zip(winner_bank, accepted) if row['index'] < paper_n]
        audit = [row for row in winner_audit if row['index'] < paper_n]
    else:
        curves, audit = generate_bank(paper_n, seed, 0, generation_trial, workers=workers)
        shared = {tuple(row['seed_key']): curve for row, curve in zip(accepted, winner_bank)}
        for row, curve in zip((r for r in audit if r['accepted']), curves):
            key = tuple(row['seed_key'])
            if key in shared and not np.array_equal(curve.ts, shared[key].ts):
                raise ValueError('Replayed shared training curve differs from selected bank')
    return curves, audit


def paired_scores(winner_records, paper_records):
    winner, paper = pd.DataFrame(winner_records), pd.DataFrame(paper_records)
    for frame in [winner, paper]:
        if frame.empty or frame.duplicated(KEYS).any():
            raise ValueError('Empty or duplicate comparison targets')
    paired = winner.merge(paper, on=KEYS, suffixes=('_winner', '_paper'),
                          how='outer', validate='one_to_one', indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('Winner and paper must have exactly the same test targets')
    for column in ['actual', 'baseline', 'baseline_abs_error']:
        if not np.array_equal(paired[column+'_winner'], paired[column+'_paper']):
            raise ValueError(f'Comparison labels differ: {column}')
    paired = paired.drop(columns='_merge')
    paired['delta_abs_error'] = paired.abs_error_winner-paired.abs_error_paper
    return paired


def summarize_pairs(paired, seed, draws=10000):
    winner_mae = float(paired.abs_error_winner.mean())
    paper_mae = float(paired.abs_error_paper.mean())
    blocks = paired.groupby('curve_id').agg(delta_sum=('delta_abs_error', 'sum'),
                                           n=('delta_abs_error', 'size'))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7, 1]))
    samples = rng.integers(0, len(blocks), (draws, len(blocks)))
    bootstrap = blocks.delta_sum.to_numpy()[samples].sum(axis=1)/blocks.n.to_numpy()[samples].sum(axis=1)
    return dict(scored_forecasts=len(paired), test_curves=len(blocks), winner_mae=winner_mae,
        paper_mae=paper_mae, persistence_mae=float(paired.baseline_abs_error_winner.mean()),
        delta_mae=winner_mae-paper_mae,
        relative_mae_change=(winner_mae/paper_mae-1) if paper_mae else None,
        winner_lower_error_fraction=float((paired.delta_abs_error < 0).mean()),
        equal_error_fraction=float((paired.delta_abs_error == 0).mean()),
        delta_mae_bootstrap_95=np.quantile(bootstrap, [.025, .975]).tolist(),
        interpretation='Negative delta favors BO winner; one shared training seed, not a global optimality claim')


def run_comparison(out, winner_params, winner_bank, winner_audit, seed, generation_trial,
                   workers=1, mode='upstream', smoothing='gam', progress=None):
    if smoothing != 'gam':
        raise ValueError('Winner-versus-paper experiment requires GAM')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    def update(phase, **details):
        atomic_json(out/'progress.json', dict(status='running', phase=phase, **details))
        if progress: progress(phase, **details)
    protocol = dict(COMPARISON_PROTOCOL, root_seed=seed, paper_params=PAPER_PARAMETERS,
        winner_params=winner_params, generation_trial=generation_trial, embedding_mode=mode,
        smoothing=smoothing, min_history=max(winner_params['k'], PAPER_PARAMETERS['k'])+1)
    atomic_json(out/'protocol.json', protocol)
    update('generate_fresh_test')
    test, audit = generate_bank(COMPARISON_PROTOCOL['test_attempts_per_family'], seed, 2, 1, workers=workers)
    train_keys = {tuple(r['seed_key']) for r in winner_audit}
    if any(tuple(r['seed_key']) in train_keys or r['role'] != 2 or r['trial'] != 1 for r in audit):
        raise ValueError('Test must be disjoint from training and legacy heldout namespace')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4, 1]))
    selected = rng.permutation(len(test))[:COMPARISON_PROTOCOL['test_curve_cap']]
    accepted_audit = [row for row in audit if row['accepted']]
    if len(accepted_audit) != len(test):
        raise ValueError('Test bank and accepted audit differ')
    pd.DataFrame([dict(curve_id=curve_id, family=accepted_audit[index]['family'],
        attempt_index=accepted_audit[index]['index'],
        seed_key=':'.join(map(str,accepted_audit[index]['seed_key'])), observations=len(test[index].ts))
        for curve_id,index in enumerate(selected)]).to_csv(out/'test_manifest.csv',index=False)
    test = [test[i] for i in selected]
    save_packed(test, out/'test_curves.npz'); write_audit(out/'test_audit.jsonl.gz', audit)
    update('score_winner')
    x, y = embeddings(winner_bank, winner_params['k'], mode=mode)
    _, winner_records = evaluate(test, x, y, winner_params['k'], winner_params['closest'],
        smoothing=smoothing, min_history=protocol['min_history'], workers=workers,
        on_progress=lambda done,total: update('score_winner', scored_origins=done,total_origins=total))
    del x, y
    pd.DataFrame(winner_records).to_csv(out/'winner_forecasts.csv', index=False)
    update('prepare_paper_training')
    paper, paper_audit = paper_bank(winner_bank, winner_audit, winner_params['num_curves'],
        PAPER_PARAMETERS['num_curves'], seed, generation_trial, workers)
    write_audit(out/'paper_training_audit.jsonl.gz', paper_audit)
    save_packed(paper, out/'paper_training.npz')
    update('score_paper', accepted_training_curves=len(paper))
    x, y = embeddings(paper, PAPER_PARAMETERS['k'], mode=mode)
    _, paper_records = evaluate(test, x, y, PAPER_PARAMETERS['k'], PAPER_PARAMETERS['closest'],
        smoothing=smoothing, min_history=protocol['min_history'], workers=workers,
        on_progress=lambda done,total: update('score_paper', scored_origins=done,total_origins=total))
    del x, y, paper
    pd.DataFrame(paper_records).to_csv(out/'paper_forecasts.csv', index=False)
    paired = paired_scores(winner_records, paper_records)
    paired.to_csv(out/'paired_forecasts.csv', index=False)
    result = summarize_pairs(paired, seed, COMPARISON_PROTOCOL['bootstrap_draws'])
    result.update(winner_params=winner_params, paper_params=PAPER_PARAMETERS,
                  paper_accepted_training_curves=sum(r['accepted'] for r in paper_audit))
    tables = []
    for name, records in [('bo_winner',winner_records), ('paper_params',paper_records)]:
        frame = pd.DataFrame(records); frame['model'] = name; tables.append(frame)
    scores = pd.concat(tables, ignore_index=True)
    metrics = scores.groupby('model', as_index=False).agg(n=('abs_error','size'), mae=('abs_error','mean'))
    metrics.to_csv(out/'metrics.csv', index=False)
    scores.groupby(['model','horizon'],as_index=False).agg(n=('abs_error','size'),mae=('abs_error','mean')).to_csv(out/'metrics_by_horizon.csv',index=False)
    scores.groupby(['model','family'],as_index=False).agg(n=('abs_error','size'),mae=('abs_error','mean')).to_csv(out/'metrics_by_family.csv',index=False)
    atomic_json(out/'summary.json', result)
    make_report(out, result)
    names = ['protocol.json','test_curves.npz','test_manifest.csv','test_audit.jsonl.gz','paper_training.npz',
             'paper_training_audit.jsonl.gz','paired_forecasts.csv','metrics.csv','summary.json']
    atomic_json(out/'metadata.json', dict(status='complete', artifacts_sha256={name:file_hash(out/name) for name in names},
        test_used_for_selection=False, source_sha256=file_hash(Path(__file__))))
    update('complete')
    atomic_json(out/'progress.json', dict(status='complete'))
    return result


def make_report(out, result):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'smoa-comparison-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    horizon = pd.read_csv(out/'metrics_by_horizon.csv')
    fig, ax = plt.subplots(figsize=(8,5))
    for name, group in horizon.groupby('model'):
        ax.plot(group.horizon,group.mae,'o-',label=name)
    ax.set(xlabel='Forecast horizon',ylabel='Synthetic test MAE',xticks=[1,2,3,4],
           title='GAM: BO winner versus published parameters | same independent test')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(out/f'mae_comparison.{ext}',dpi=160)
    plt.close(fig)
    ci=result['delta_mae_bootstrap_95']
    lines=['# BO GAM winner versus published parameters','',
        'Test mới được sinh sau selection: role=2, trial=1; không dùng để chọn winner hay phản hồi cho BO.',
        'Hai cấu hình dùng cùng test curves, mọi origins từ max(k_winner,k_paper)+1, cùng 4 horizons và GAM.',
        'Training dùng cùng seed namespace và chung prefix family/index; winner giữ đúng bank được chọn trong BO.', '',
        '| Cấu hình | num_curves/family | k | closest | Test MAE |', '| --- | ---: | ---: | ---: | ---: |']
    for label,params,key in [('BO winner',result['winner_params'],'winner_mae'),('Paper',result['paper_params'],'paper_mae')]:
        lines.append(f"| {label} | {params['num_curves']} | {params['k']} | {params['closest']} | {result[key]:,.4f} |")
    lines += ['',f"Test: {result['test_curves']} curves, {result['scored_forecasts']:,} paired forecasts.",
        f"Delta MAE (winner − paper): **{result['delta_mae']:,.4f}**; 95% curve-block bootstrap CI [{ci[0]:,.4f}, {ci[1]:,.4f}].", '',
        'Delta âm có lợi cho winner. CI mô tả biến động giữa các test curves, không bao gồm biến động do đổi training seed.',
        'Kết quả một ngân sách BO/seed không chứng minh tối ưu toàn cục. Không dùng COVID truth trong selection hay test này.', '',
        '![MAE by horizon](mae_comparison.png)','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
