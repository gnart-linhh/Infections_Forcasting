# Robust Synthetic Method of Analogues for emerging-disease forecasting

This project reproduces [Murph et al., *Synthetic method of analogues for emerging infectious disease forecasting*, PLOS Computational Biology (2025)](https://doi.org/10.1371/journal.pcbi.1013203) and investigates two direct modifications: similarity-weighted analogue medians and optional regularization using analogue-future dispersion. The [original R implementation](https://github.com/lanl/precog/tree/main/smoa) is the methodological reference; actual workspace Python code remains the implementation baseline. Numerical parity with R is not claimed; see the existing [reproduction audit](REPRODUCTION_AUDIT.md).

The forecasting rule requires **no target-disease training, tuning, residual calibration, or learned ensemble weights**. It receives exactly 6, 8, or 11 permitted weekly observations and forecasts weeks 1–4. Hyperparameters are selected exclusively on independently generated synthetic curves and frozen before opening external evaluation data. Target futures enter only in a separate scoring step after issued forecasts have been saved.

## Refactor and preservation

The previous final-project implementation was a historically adaptive COVID ensemble. It is no longer the proposed method or an active dependency. Its complete pre-refactor source, configuration, tests, notebook, README, and dependency specification are preserved in `outputs/final_project/pre_robust_source_262eb2f1e3b8913e.zip`. Previous research outputs remain untouched under their original directories. Restoring that source in a separate checkout allows inspection of its old serialized model classes.

The existing final-project directory was renamed in place from `experiments/adaptive/` to `experiments/robust/`; its tests were similarly renamed. The obsolete `features.py`, `ensemble.py`, and target-residual `calibration.py` were removed. Reusable point/interval metrics were preserved. Original midterm forecasting, generation, smoothing, matching, uncertainty, and experiment code remain unchanged. Uncommitted user figures were preserved. No parallel project was created.

The audit identified exact short-history inputs in the existing E1 SQLite archive, compatible synthetic embeddings, and reusable runtime/hash utilities. The earlier full-history neighbour rankings are unsuitable for this study: changing the supplied history changes GAM smoothing and therefore requires rematching. The new pipeline never loads those rankings or stored E1 queries. It reuses only the frozen reference embedding and raw permitted histories.

## Installation and layout

Use the existing environment or Python 3.13:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

SHAP is retained for direct model-agnostic explanations. Scikit-learn is a dependency of SHAP; the proposed forecasting system does not import it to train a model. No boosting framework, neural network, or new calibration model is introduced.

The active final-project files are:

| Location | Responsibility |
| --- | --- |
| `experiments/robust/method.py` | Weighted medians, dispersion regularization, and outcome-free sMOA interface |
| `experiments/robust/tuning.py` | Independent synthetic roles, provenance checks, and frozen selection |
| `experiments/robust/data.py` | Read-only reference and permitted-history adapters; separate outcome join |
| `experiments/robust/evaluation.py` | Paired point/interval metrics and cluster bootstrap |
| `experiments/robust/pipeline.py` | Shared execution, persistence, and E1–E4 analyses |
| `experiments/robust/explain.py` | Direct and contrastive Kernel SHAP; analogue plots |
| `experiments/run.py` | Single CLI |
| `configs/experiments.json` | Single central final-project configuration |
| `tests/robust/` | Method, leakage, provenance, metric, and explanation tests |
| `notebooks/demo.ipynb` | Concise executed demonstration |

Original reproduction modules and numbered midterm experiments remain in their existing directories.

## Original behavior and the four variants

The baseline calls the existing `experiments/forecasting/model.py::forecast` directly. It floors supplied levels at 1; fits the existing thin-plate/GCV smoother to **only the supplied T observations**; differences its last k+1 fitted levels, with k=5; finds the closest 4,422 reference windows by exact L1 distance; takes the coordinatewise median of their four future increments; cumulatively reconstructs from the final floored raw observation; and floors point forecasts at 1.

The reference embedding uses `nonconstant_v2` and the existing paper-aligned **omit-bridge convention**. Its five stored future increments are loaded with the first omitted, yielding four forecast positions. This established indexing convention is preserved, not corrected silently. A regression check on the real early-stage sample reproduced all **600 original E1 seed-0 forecasts exactly**, including T=6,8,11. `model.py`, `smoothing.py`, `neighbors.py`, and `uncertainty.py` were not modified.

| ID | Aggregation | Regularization |
| --- | --- | --- |
| M0 | Original median increments | None |
| M1: sMOA-W | Similarity-weighted median increments | None |
| M2: sMOA-R | Original median increments | Frozen β |
| M3: sMOA-WR | Similarity-weighted median increments | Frozen β |

All variants share the exact history, smoothing, reference library, k, neighbour count, neighbour identities, and horizon conventions. No separate library is tuned for a variant.

For selected distances d, weights are proportional to `exp(-(d-min(d))/tau)`, where `tau = temperature * IQR(d)`. If the IQR vanishes, the scale is the mean distance above the minimum; entirely equal distances use uniform weights. Computation in relative distance units avoids dependence on count magnitude and stabilizes the exponential. Weighted medians use stable sorting and average adjacent positive-mass values at a half-mass boundary, with four machine epsilons of tolerance. Uniform weights delegate exactly to NumPy median, preserving its even-sample convention.

For each horizon, dispersion U is the IQR of **cumulative future count increments** divided by `max(mean(floored permitted history),1)`. These quantities have compatible count units. No log/normalized library values are confused with counts. With `lambda = 1/(1 + beta*U)`, the regularized forecast is `max(0, anchor + lambda*(aggregate-anchor))`, evaluated in equivalent convex form to avoid subtraction overflow. The anchor is the original method's final floored observation. β=0 and U=0 recover the aggregation exactly. Lambda is evaluated in log space; only numerical underflow is guarded by the smallest positive float64 value.

The proposed API is disease-agnostic. For another disease, load the synthetic reference once and supply a permitted vector:

```python
from experiments.robust.pipeline import load_config
from experiments.robust.data import load_reference

config = load_config("configs/experiments.json")
engine, _, _ = load_reference(config)
# Load temperature and beta from a completed SYNTHETIC validation run.
# No target-disease outcomes are passed to engine.variants().
forecasts, diagnostics = engine.variants(
    [10, 14, 17, 22, 31, 40], temperature=4.0, beta=0.0
)
```

The values in this API example illustrate the interface, not experimental observations. Reference paths and actual selected parameters must match the intended frozen run.

## Data and provenance

The large input artifacts are local and Git-ignored; a fresh clone must prepare or restore them. The configured paths are:

- Original library: `outputs/reproduction_full/01_synthetic/curves.csv`.
- Matching embedding: `outputs/paper_embedding_k5_h4_omit_nonconstant_v2/`.
- Packed reference curves: `outputs/experiment_2/packed/seed_0/`.
- Real early-stage inputs and scoring observations: `outputs/experiment_1_nonconstant_v2/results.sqlite`.

The library contains 54,081 accepted curves and 12,921,078 matching windows. The loader verifies the source CSV, x/y matrices, packed-curve checksum, curve counts, and compatible provenance. A set of complete curve-content hashes is constructed from the reference bank for disjointness checks. Old real rankings are never substituted for a changed query. In-memory caching is bounded and keyed by the complete supplied history within one frozen engine.

Synthetic validation uses 108 newly generated eligible weekly count curves: 12 per family and seed, across three original generator families and seeds 0/1/2. A separate background role contains 9 curves, and a held-out check role contains 36. All roles have distinct SeedSequence keys `[42, 3187361291, role, replicate, family, attempt]`. Whole-curve hashes are checked against every reference curve and other accepted roles. No sliding windows from a single curve are split between reference, validation, background, or check sets. Attempts, rejections, keys, hashes, cadence, units, onset, artifact checksums, and generator source are recorded. No input dataset is duplicated or committed.

Synthetic curves come from the original generator families and parameter distributions. The extra held-out seed/curve check is independent but **not an out-of-distribution generator test**. Synthetic validation can be easier than a genuinely new disease; no cross-disease guarantee follows from it.

## Selection and early-stage protocol

The original neighbour count is fixed at 4,422 to isolate the proposed modifications. The small, prespecified search crosses temperature coefficients [.25,1,4] with β [0,.1,.3,1]. Primary selection uses pooled synthetic MAE. Candidates must keep RMSE within 5% and severe-underprediction frequency within 2 percentage points of M0. If none pass, the fallback restricts β to zero. Among candidates within 1% of the best eligible MAE, prefer smaller β, then larger temperature (closer to uniform weighting). This policy was specified before external evaluation.

For each independent validation curve, use its first 6/8/11 weekly count observations after the first observed value >=10 and predict the next four weeks. Multiple histories from a curve are dependent and are not treated as independent curves. The selected configuration is persisted in the synthetic cache before any target-disease database read; the `validation` command succeeds even if the configured target database does not exist.

The primary real experiment uses the existing E1 **early** arm: 50 states × three history lengths =150 cases, with 600 forecast targets per model. T means the **first T observations since the recorded observed outbreak onset**. Origins range from April to June 2020; later T values generally have different origins. Onset metadata came from an earlier known vintage. Only the T recorded raw observations enter the forecaster. Stored GAM queries and fits are ignored and all smoothing/matching is recomputed.

Monday `forecast_origin` is issuance; Saturday `reference_date` is the last observed weekly endpoint; `target_date = reference_date + 7*horizon days`. Archive data cutoffs are checked against issuance. Administrative negative history revisions are handled by the original level floor; raw final outcomes remain available for scoring. The real data adapter is COVID-specific; the forecasting rule and synthetic selection do not depend on a disease label.

A separate `evaluation.design = "fixed_origin"` configuration can use the most recent T observations at later common origins. It produces a distinct dataset identifier and run. These results must not be pooled with the early-origin study. The default study reports only the early design.

`issued_predictions.csv.gz` contains forecasts, input hashes, phase, dispersion, lambda, uncertainty quantiles, and matching summaries **without actuals**. It is written before `load_real_outcomes` opens the outcome table. The evaluator then checks unique cases, weekly targets, and complete joins and writes scored predictions. Forecast and synthetic-selection functions cannot accept real future outcomes. The tests explicitly construct two datasets with identical histories but very different futures and require identical issued forecasts and quantiles.

COVID data were examined during the midterm. This refactor removes target-outcome fitting, but its COVID results remain development evidence rather than untouched cross-disease confirmation. No independent other-disease dataset was available locally.

## Commands and outputs

Run from the repository root:

```bash
.venv/bin/python -m experiments.run --experiment validation
.venv/bin/python -m experiments.run --experiment all --smoke
.venv/bin/python -m experiments.run --experiment all
```

The experiment choices are `validation`, `early` (E1), `ablation` (E2), `robustness` (E3), `probabilistic` (E4), `xai` (E5), and `all`. Numerical preparation, frozen selection, and paired point/interval tables are shared rather than duplicated across scripts; `xai` and `all` additionally execute direct SHAP. Smoke uses the first two configured states and one curve per synthetic family/seed/role, retains the full reference library, and stores separately labelled `smoke_` outputs. Its smaller validation can select different parameters and must not be pooled with full-study results.

Programmatically call `load_config(...)` and `run(config, experiment=..., smoke=...)`. `forecast_cases(engine, cases, frozen, ...)` is the outcome-free batch interface. The central configuration controls all scientific candidates, role sizes/seeds, observed history lengths, horizons, evaluation design, NB draws, explanation budgets, and output root. Unsupported changes to the baseline protocol are rejected rather than silently ignored.

One output root, `outputs/robust_smoa/`, contains compact synthetic role caches and run directories identified by configuration, code, provenance, environment, and external input checksums. A run saves frozen parameters, validation scores, issued/scored predictions, matching diagnostics, metrics by history/horizon, four-variant ablation, robustness, availability/failure counts, cluster comparisons, initial-NB uncertainty scores, runtime, and status. Completed XAI runs add direct/contrastive SHAP values, summary/waterfalls, analogue plots, and stability metadata. Original outputs remain in place. Conflicting explicit output-directory identities are rejected.

Regenerate tables, the comparison figure, and saved explanation/analogue figures without rematching:

```bash
.venv/bin/python -m experiments.run --regenerate outputs/robust_smoa/<run_id>
```

The original midterm entry points remain available:

```bash
.venv/bin/python experiments/data/prepare_state_truth.py --help
.venv/bin/python experiments/data/prepare_weekly_archive.py --help
.venv/bin/python experiments/comparison/experiment_1.py --help
.venv/bin/python experiments/reproduction/run_full.py --help
.venv/bin/python -u experiments/run_pipeline.py \
  --manifest outputs/truth_states/manifest.csv --protocol source \
  --workers 4 --output-dir outputs/reproduction_gam_full
```

These preparation commands may be expensive. Existing compatible inputs make regeneration unnecessary. Missing data cause an explicit failure; smoke mode never silently fabricates a replacement dataset.

## Executed findings

Implementation, smoke, synthetic validation, real early-stage evaluation, held-out synthetic check, and direct XAI have run. These completion claims are distinct from demonstrated forecast improvement.

Verified full-run artifacts are available locally: [metrics](outputs/robust_smoa/full_ea4f7cf7284d9893/metrics.csv), [frozen parameters](outputs/robust_smoa/full_ea4f7cf7284d9893/selected_config.json), [direct explanation metadata](outputs/robust_smoa/full_ea4f7cf7284d9893/explanation_metadata.json), and [exact baseline regression](outputs/robust_smoa/full_ea4f7cf7284d9893/baseline_regression.json). Regenerating from saved artifacts reproduced the metric table exactly, preserved the numerical SHAP artifact, and required no matching. Final numerical forecasts matched the first completed robust run exactly. These links require the local, Git-ignored results directory.

Full synthetic validation selected **temperature=4 and β=0**. Selected validation MAE was 66,395.42 versus 66,502.53 for M0 across 1,296 synthetic targets. Regularization is implemented and tested but was not selected for deployment; M2=M0 and M3=M1 in the frozen full study. Positive-β results remain in `validation_metrics.csv`; they were not retuned using real errors.

| Dataset | Model | Forecasts | MAE | RMSE | Bias |
| --- | --- | ---: | ---: | ---: | ---: |
| COVID early, 50 states | M0 / M2 | 600 | 1,436.10 | 4,032.80 | 386.58 |
| COVID early, 50 states | M1 / M3 | 600 | 1,427.57 | 4,004.40 | 366.28 |
| Independent synthetic check, 36 curves | M0 / M2 | 432 | 193,885.78 | 741,618.21 | -151,517.28 |
| Independent synthetic check, 36 curves | M1 / M3 | 432 | 193,899.56 | 740,191.51 | -151,995.69 |

The real pooled MAE difference is **-8.53 counts, or 0.59%**. The state-cluster bootstrap interval for the difference was [-19.40,-1.07] in this explored sample. Shared epidemic shocks, selection history, multiple comparisons, and the small effect limit interpretation. Held-out synthetic MAE was slightly worse, and real severe-underprediction frequency rose from 17.00% to 17.83%. This is not evidence that every modification improves generalization or protects against missed surges. At T=11, weighted MAE was slightly worse than M0.

Metrics include MAE/RMSE/bias, relative MAE improvement, 95th/99th percentile absolute errors, severe underprediction (`prediction < .5*actual` for positive actual), and severe overprediction (`prediction > 2*max(actual,1)`). Phase uses growth between the last two and preceding two permitted observations, with ±10% thresholds. High volatility means standard deviation of permitted first differences / local mean >.5. Definitions are fixed, not fitted to external errors. Comparisons use common finite four-model samples and report counts/exclusions. Region/whole-curve bootstrap retains all horizons and T cases within a cluster; it does not remove cross-region epidemic dependence.

## Cold-start uncertainty and direct explanations

The original NB implementation is retained. The primary cold-start study freezes published initial sizes [20,10,6,.3] across horizons, uses 5,000 draws, and retains the original initial mean `point+1` and simulated median recentering. There is **no dispersion fitting on target outcomes**. M0 is labelled `original_nb_initial`; modified points with the same construction are labelled `modified_point_nb_initial`. These are explicit frozen comparators, not a newly calibrated uncertainty model. Median recentering can give negative lower bounds, which are preserved and disclosed rather than silently changing the original distribution.

The seven quantiles [.025,.1,.25,.5,.75,.9,.975] yield WIS (twice mean pinball loss), coverage, widths and upper/lower misses at 50/80/95%. Evaluation uses inclusive interval endpoints; the untouched legacy scorer excluded the lower endpoint. The real M0 initial-NB 95% coverage was 72.83%, and M3 coverage 72.50%; WIS changed from 959.63 to 954.09. Undercoverage remains severe. The old online target-error MLE is not part of this primary method, and no uncertainty improvement is claimed.

Kernel SHAP explains **the actual M3 count forecast** at a fixed horizon and **M3−M0**. Each changed sequence reruns original GAM preprocessing, exact matching, weighted aggregation, regularization and reconstruction. Only mathematically identical complete histories reuse a cached result. No surrogate is used. All T observations are features; no outside observations remain fixed.

Backgrounds are separate synthetic whole-curve windows of matching length, rescaled as whole histories to the observed mean using no future information. Four cases cover an improved forecast, a worsened forecast, the fastest observed rising phase, and high analogue disagreement. Kernel sampling is bounded at 64 coalitions with four background histories, repeated on a second background. Output metadata records actual evaluations, additivity residuals, seeds, case roles, and background hashes. SHAP values are in count units; the contrastive target explains the forecasting modification directly.

The global summary is **importance over this small selected case sample**, not a population-level claim. Additivity held to numerical precision, but background rank correlations varied substantially (including a contrastive correlation near .03). Correlated weeks, synthetic-background dependence, scale conditioning, implausible hybrid sequences, and Monte Carlo coalition sampling limit attribution stability. SHAP is not causal. Analogue plots complement it using completed outcomes only for retrospective visualization.

## Tests and demonstration

```bash
.venv/bin/python -m unittest discover -s tests/robust -v
.venv/bin/python -m unittest discover -s tests/forecasting -v
.venv/bin/python -m unittest discover -s tests/reproduction -v
.venv/bin/python -m unittest discover -s experiments/synthetic -v
.venv/bin/python -m unittest discover -s experiments/optimization -v
.venv/bin/python -m unittest discover -s experiments/comparison -v
.venv/bin/python experiments/notebooks.py notebooks/demo.ipynb --timeout 300
```

The full repository sweep passed **182 tests**, including 18 robust-method tests and all 53 original forecasting tests. New checks cover NumPy median fidelity at uniform weights, even-sized medians, distance ties, extreme temperatures and finite values, β=0/U=0 invariants, monotonic lambda, count-scale cumulative dispersion, truncated GAM input lengths, changed-input rematching, future-outcome isolation, whole-curve separation, direct/contrastive SHAP additivity, metric examples, shared samples, and deterministic forecasts/quantiles.

The updated notebook executed four code cells in a fresh kernel. It calls reusable modules, displays the original and modified forecasts, frozen NB intervals, selected analogues, direct and contrastive waterfall plots, and actual smoke metrics. Its data prerequisites are the same as the CLI. Existing research artifacts and the prior implementation archive are preserved; no trained target-disease component remains in the active final-project pipeline.
