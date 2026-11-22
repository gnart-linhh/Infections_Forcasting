"""Bounded model-agnostic SHAP of the actual sMOA function and its contrast."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.runtime import atomic_json


class DirectForecast:
    """Every new masked history executes original preprocessing and rematching.

    Two outputs share the exact same computation: the robust count forecast and
    robust-minus-original at one fixed horizon. No surrogate or stale ranking.
    """
    def __init__(self, engine, frozen, horizon, reference_engine=None, policy=None, rho=1.):
        self.engine, self.frozen, self.horizon = engine, frozen, horizon
        self.reference_engine = reference_engine
        self.policy,self.rho=policy,rho
        self.evaluations = 0

    def __call__(self, histories):
        histories = np.atleast_2d(histories)
        if not np.isfinite(histories).all() or (histories < 0).any():
            raise ValueError('SHAP inputs must be finite nonnegative count histories')
        rows = []
        for history in histories:
            if self.policy is None:
                values, _ = self.engine.variants(history,self.frozen['temperature'],self.frozen['beta'])
                robust,original=values['M3'][self.horizon-1],values['M0'][self.horizon-1]
            else:
                original=self.engine.match(history).baseline[self.horizon-1]
                robust=self.policy.forecast(self.engine,history,self.rho)[0][self.horizon-1]
            if self.reference_engine is not None:
                original = self.reference_engine.match(history).baseline[self.horizon-1]
            rows.append([robust, robust-original])
        self.evaluations += len(histories)
        return np.asarray(rows)


def background_histories(synthetic, length, observed_history, count, seed):
    """Independent synthetic count windows, scaled using observed history only.

    Scale each whole background window to the observed input mean. This prevents
    unit/level mismatch without changing the target forecasting preprocessing.
    All T observations are explained; none outside the window remain fixed.
    """
    available = []
    for curve, record in zip(synthetic['background'], synthetic['background_records']):
        start = record['onset']
        values = np.asarray(curve.ts[start:start+length], float)
        if len(values) == length:
            available.append(values)
    if not available:
        raise ValueError('Independent synthetic SHAP backgrounds missing')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(available), min(count, len(available)), replace=False)
    level = max(float(np.mean(np.maximum(observed_history, 1))), 1.)
    return np.array([z*level/max(float(z.mean()), np.finfo(float).tiny) for z in (available[i] for i in chosen)])


def choose_cases(cases, predictions, horizon, maximum,baseline_model='M0',forecast_model='M3'):
    allowed = {c.case_id: c for c in cases}
    table = predictions.loc[predictions.case_id.isin(allowed) & predictions.horizon.eq(horizon)]
    point = table.pivot(index='case_id', columns='model', values='prediction')
    context = table.drop_duplicates('case_id').set_index('case_id')
    gain = (point[baseline_model]-context.actual).abs()-(point[forecast_model]-context.actual).abs()
    labels = {
              'rising': context.loc[context.phase.eq('rising'), 'growth_rate'].idxmax()
                        if context.phase.eq('rising').any() else context.dispersion.idxmax(),
              'high_disagreement': context.dispersion.idxmax()}
    if gain.max()>0:labels['improved']=gain.idxmax()
    if gain.min()<0:labels['worse']=gain.idxmin()
    if context.phase.eq('declining').any():labels['declining']=context.loc[context.phase.eq('declining'),'growth_rate'].idxmin()
    # Four explanation roles may refer to the same case. Record this explicitly.
    selected = list(dict.fromkeys(labels.values()))
    # Prefer additional history lengths when roles overlap.
    for length in sorted(context.history_length.unique()):
        if len(selected)>=maximum:break
        if not any(allowed[i].history.size==length for i in selected):
            selected.append(context.loc[context.history_length.eq(length)].index[0])
    for ident in context.index:
        if len(selected) >= min(maximum, len(context)):
            break
        if ident not in selected:
            selected.append(ident)
    return [allowed[i] for i in selected[:maximum]], labels, gain.to_dict()


def analogue_plot(diagnostic, predictions, path,baseline_model='M0',forecast_model='M3'):
    """Render saved matched paths and issued predictions; no matching required."""
    import matplotlib.pyplot as plt
    z = np.maximum(np.asarray(diagnostic['history']), 1)
    paths = np.asarray(diagnostic['analogue_paths'])
    block = predictions.loc[predictions.case_id.eq(diagnostic['case_id'])]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, series in enumerate(paths):
        ax.plot(np.arange(-5, 5), series, color='gray', alpha=.35,
                label='Selected anchored analogues' if i == 0 else None)
    for i, series in enumerate(diagnostic.get('original_analogue_paths', [])):
        ax.plot(np.arange(-5,5), series, color='steelblue', alpha=.2,
                label='Original-library analogues' if i == 0 else None)
    ax.plot(np.arange(1-len(z), 1), z, 'ko-', label='Permitted vintage history')
    for model in [baseline_model,forecast_model]:
        group = block.loc[block.model.eq(model)].sort_values('horizon')
        ax.plot(group.horizon, group.prediction, 'o-', label=model)
    actual = block.loc[block.model.eq(baseline_model)].sort_values('horizon')
    ax.plot(actual.horizon, actual.actual, 'k--', label='Final actual (evaluation only)')
    ax.axvline(0, color='black', linestyle=':')
    ax.set(title=f'{block.region.iloc[0]}, T={len(z)}, reference {block.reference_date.iloc[0]}',
           xlabel='Weeks from last observed week (omit-bridge convention)', ylabel='Weekly counts')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_explanation(explanation, title):
    """Avoid SHAP's misleading waterfall arrows for an all-zero attribution."""
    import matplotlib.pyplot as plt
    import shap
    if np.all(np.asarray(explanation.values) == 0):
        fig,ax=plt.subplots(figsize=(8,4))
        ax.barh(explanation.feature_names,np.zeros(len(explanation.values)),color='gray')
        ax.axvline(0,color='black',linewidth=1)
        ax.set(xlim=(-1,1),xlabel='SHAP contribution (all values are zero)',title=title)
        ax.text(.5,.5,'No attributed adjustment',ha='center',va='center',transform=ax.transAxes)
        fig.tight_layout()
    else:
        shap.plots.waterfall(explanation,max_display=len(explanation.values),show=False)
        plt.gcf().suptitle(title,fontsize=10)
    return plt.gcf()


def render_explanations(out):
    """Regenerate every explanation figure from saved values and diagnostics."""
    import matplotlib.pyplot as plt
    import shap
    metadata = json.loads((out/'explanation_metadata.json').read_text())
    if metadata.get('status')=='skipped_zero_adjustment':return
    artifact_root=out/metadata.get('artifact_directory','.')
    predictions = pd.read_csv(artifact_root/'predictions.csv.gz', dtype={'case_id': str})
    diagnostics = {row['case_id']: row for row in json.loads((artifact_root/'matching_diagnostics.json').read_text())}
    with np.load(out/'explanations.npz', allow_pickle=False) as values:
        for i, case in enumerate(metadata['cases']):
            history = values[f'case_{i}_history']
            features = [f'week_{j-len(history)+1}' for j in range(len(history))]
            for column, target in enumerate(['robust_forecast', 'robust_minus_original']):
                explanation = shap.Explanation(values=values[f'case_{i}_values'][:, column],
                    base_values=values[f'case_{i}_base'][column], data=history, feature_names=features)
                plot_explanation(explanation,f'{target}; T={len(history)}; h={metadata["selected_horizon"]}')
                plt.gcf().savefig(out/f'shap_case_{i}_{target}.png', dpi=150, bbox_inches='tight'); plt.close()
            analogue_plot(diagnostics[case['case_id']], predictions, out/f'analogues_case_{i}.png',metadata.get('baseline_model','M0'),metadata.get('forecast_model','M3'))
    frame = pd.read_csv(out/'shap_values.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, target in zip(axes, ['robust_forecast', 'robust_minus_original']):
        group = frame.loc[frame.target.eq(target)].groupby('feature').absolute_shap.mean().sort_values()
        group.plot.barh(ax=ax)
        ax.set(title=target, xlabel='Mean |SHAP| (count units)', ylabel='Observed lag')
    fig.tight_layout(); fig.savefig(out/'shap_summary.png', dpi=150); plt.close(fig)


def explain_experiment(engine, cases, predictions, frozen, synthetic, config, out, reference_engine=None, policy=None, rho=1.,baseline_model='M0',forecast_model='M3',artifact_directory='.'):
    import shap
    selected, roles, gains = choose_cases(cases, predictions, config['xai']['horizon'], config['xai']['max_cases'],baseline_model,forecast_model)
    contrast=predictions.loc[predictions.model.isin([baseline_model,forecast_model])].pivot(index=['case_id','horizon'],columns='model',values='prediction')
    if np.array_equal(contrast[baseline_model],contrast[forecast_model]):
        atomic_json(out/'explanation_metadata.json',dict(status='skipped_zero_adjustment',baseline_model=baseline_model,forecast_model=forecast_model,reason='No nonzero forecast change; no contrastive improvement plots generated'))
        return
    importance, stability, metadata, value_artifacts = [], [], [], {}
    rng_state = np.random.get_state()
    np.random.seed(config['seed'])
    for i, case in enumerate(selected):
        length = len(case.history)
        features = [f'week_{j-length+1}' for j in range(length)]
        wrapper = DirectForecast(engine, frozen, config['xai']['horizon'], reference_engine,policy,rho)
        values_by_background, base_by_background = [], []
        for repeat in range(2):
            background = background_histories(synthetic, length, case.history,
                                              config['xai']['background_size'], config['seed']+repeat)
            explainer = shap.KernelExplainer(wrapper, background, feature_names=features)
            values = np.asarray(explainer.shap_values(case.history[None, :],
                                nsamples=config['xai']['nsamples'], l1_reg=0, silent=True))
            # Current SHAP shape is (samples, features, outputs).
            values = values.reshape(1, length, 2)[0]
            base = np.asarray(explainer.expected_value)
            output = wrapper(case.history)[0]
            reported=predictions.loc[predictions.case_id.eq(case.case_id)&predictions.horizon.eq(config['xai']['horizon'])].set_index('model').prediction
            np.testing.assert_allclose(output,[reported[forecast_model],reported[forecast_model]-reported[baseline_model]],rtol=1e-12,atol=1e-8)
            residual = float(np.max(np.abs(base+values.sum(axis=0)-output)))
            if residual > 1e-6*max(float(np.max(np.abs(output))), 1.):
                raise ValueError('Direct SHAP additivity check failed')
            values_by_background.append(values)
            base_by_background.append(base)
        phi, other = values_by_background
        for target_index, target in enumerate(['robust_forecast', 'robust_minus_original']):
            stability.append(dict(case_id=case.case_id, target=target,
                                   mean_absolute_background_change=float(np.abs(phi[:, target_index]-other[:, target_index]).mean()),
                                   rank_correlation=float(spearmanr(np.abs(phi[:, target_index]), np.abs(other[:, target_index])).statistic)
                                   if np.ptp(np.abs(phi[:, target_index])) and np.ptp(np.abs(other[:, target_index])) else None))
            for j, name in enumerate(features):
                importance.append(dict(case_id=case.case_id, history_length=length, horizon=config['xai']['horizon'],
                                       target=target, feature=name, shap_value=float(phi[j, target_index]),
                                       absolute_shap=float(abs(phi[j, target_index]))))
        metadata.append(dict(case_id=case.case_id, history_length=length, roles=[r for r, ident in roles.items() if ident == case.case_id],
                             retrospective_absolute_error_gain=float(gains[case.case_id]),
                             evaluations=wrapper.evaluations, additivity_residual=residual))
        value_artifacts[f'case_{i}_values'] = phi
        value_artifacts[f'case_{i}_background_alternative'] = other
        value_artifacts[f'case_{i}_base'] = base_by_background[0]
        value_artifacts[f'case_{i}_history'] = case.history
    np.random.set_state(rng_state)
    frame = pd.DataFrame(importance)
    frame.to_csv(out/'shap_values.csv', index=False)
    global_importance = frame.groupby(['history_length', 'horizon', 'target', 'feature'], as_index=False).absolute_shap.mean()
    global_importance.to_csv(out/'shap_importance.csv', index=False)
    pd.DataFrame(stability).to_csv(out/'explanation_stability.csv', index=False)
    np.savez_compressed(out/'explanations.npz', **value_artifacts)
    atomic_json(out/'explanation_metadata.json', dict(
        method='Kernel SHAP of exact RobustSMOA, no surrogate', selected_horizon=config['xai']['horizon'],
        targets=[f'{forecast_model} forecast count', f'{forecast_model} minus {baseline_model} count'], cases=metadata, retrospective_roles=roles,
        baseline_model=baseline_model,forecast_model=forecast_model,artifact_directory=artifact_directory,
        contrast_reference='original library' if reference_engine is not None else 'same library, original aggregation',
        forecast_target=forecast_model,
        background='Independent synthetic whole curves; whole histories rescaled to the observed input mean, never to future outcomes',
        background_curve_hashes=[r['sha256'] for r in synthetic['background_records']],
        nsamples=config['xai']['nsamples'], background_size=config['xai']['background_size'],
        bounds='Two background runs per case; nsamples is a coalition-sampling budget, each coalition averages its background histories',
        actual_matching_calls=engine.match_calls, seed=config['seed'],
        limitations=['Correlated weeks and hybrid out-of-distribution histories',
                     'Dependence on synthetic background distribution and scale conditioning',
                     'Small retrospective case sample is not population-level global importance',
                     'Kernel coalition sampling is approximate when fewer than all coalitions are used',
                     'Non-causal explanations; never reuse neighbour rankings for changed inputs']))

    render_explanations(out)
