# Context-aware Synthetic Method of Analogues for emerging-disease forecasting

This project reproduces [Murph et al., *Synthetic method of analogues for emerging infectious disease forecasting*, PLOS Computational Biology (2025)](https://doi.org/10.1371/journal.pcbi.1013203) and investigates two direct modifications: similarity-weighted analogue medians and optional regularization using analogue-future dispersion. The [original R implementation](https://github.com/lanl/precog/tree/main/smoa) is the methodological reference; actual workspace Python code remains the implementation baseline. Numerical parity with R is not claimed; see the existing [reproduction audit](REPRODUCTION_AUDIT.md).

The forecasting rule requires **no target-disease training, tuning, residual calibration, or learned ensemble weights**. It receives exactly 6, 8, or 11 permitted weekly observations and forecasts weeks 1–4. Hyperparameters are selected exclusively on independently generated synthetic curves and frozen before opening external evaluation data. Target futures enter only in a separate scoring step after issued forecasts have been saved.

The current main research experiment is **synthetic-trained context-aware aggregation**, available with `.venv/bin/python -m experiments.run --experiment policy`. It preserves the full original matching library and learns only from independent synthetic outbreaks. The weighted-median/regularization and library studies below remain historical baselines; the final section documents the new policy, strict no-change selection rule, and observed negative transfer result.

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

## Synthetic reference-library study

Run the extension with the same central configuration:

```bash
.venv/bin/python -m experiments.run --experiment library --smoke
.venv/bin/python -m experiments.run --experiment library
```

`experiments/robust/library_study.py` reuses the original embedding, curve-code/window-start lookup, forecasting, independent synthetic roles, NB quantiles and direct SHAP. Only `change_point_sir()` was added to the original generator module; the three existing generators and their reference artifacts remain unchanged. The new study stores results under `outputs/robust_smoa/library_study/` with separate immutable configuration/source identifiers. The prior `all` workflow remains available.

The full-bank audit counted 18,387 rollercoaster curves (4,457,364 windows), 18,387 wiggle curves (4,409,779 windows), and 17,307 seasonal curves (4,053,935 windows). Neither curve counts nor window counts are exactly balanced. The original accepted bank has 54,081 curves and 12,921,078 windows, including both count/proportion scales and weekly/monthly cadences. These are preserved rather than silently filtered. Audit tables report cadence/units, lengths, peaks, eligible windows and flat-history filtering. A seeded 100,000-window sample had approximately 1.02% exact duplicate historical-difference vectors. Rounded L1-normalized difference shapes provide a clearly labelled near-duplicate proxy, not an exhaustive duplicate search.

The predefined policies are L0 (full original), L0S (uniform original subset), L1 (equal-window family allocation), L2 (balanced historical-dynamics strata), and L3 with either 25% or 50% change-point windows. Every sampled candidate has **100,000 windows**; m=4,422, k=5, GAM, bridge omission, temperature=4, and beta=0 remain fixed. L2 uses twelve predefined combinations of signed sum/absolute-sum trend, difference volatility, and turning indicators; it does not learn bins from COVID. Sparse strata redistribute unused quotas deterministically. Curves may contribute unequal numbers of windows; curve counts are reported separately.

Sampled libraries save original row IDs, not dense copies of the 12.9-million-window embedding. Manifests include row-file hashes, family/window counts, contributing curve counts, seed, policy and generator hash. Full original curve identities and hashes are in `curve_diagnostics.csv.gz`; new curve IDs, seeds and parameter draws are in `change_point_curves.json`. Dense sampled arrays exist only in memory while their nearest-neighbour index is used. Every library owns a separate matching cache.

The single new family integrates one conserved susceptible/infected/recovered state with 0–2 transmission changes at weekly boundaries. Recovery is sampled over .5–2/week, initial reproduction over .8–3.5, and transmission multipliers over .4–2.5; population is log-uniform from 200,000 to 100 million. Weekly counts are rounded negative susceptible increments. This differs from stitching independent waves: later transmission changes inherit susceptible depletion and the current infected pool. No extra observation-noise mechanism was added. Eligible reference draws use independent SeedSequence role `[42,3691107317,0,attempt]`; whole-curve hashes are checked against original references and validation/check/background roles. Two thousand accepted curves provide enough nonconstant windows for both mixtures. The first 800-curve capacity attempt stopped before evaluation because it produced only 33,395 eligible windows; increasing capacity was a budget requirement, not an error-based tuning decision.

A separate change-point stress role uses seed role 1 and reproduction=4, recovery=.4, outside the new reference family's initial ranges. Its hashes are checked against reference and other roles. This is a parameter-shift/mechanism diagnostic, not an independent real disease. Independent validation/check/background roles remain whole-curve separated. Validation reports pooled MAE **and NMAE**, using `max(mean(permitted floored history),1)` as the count scale, plus history/horizon/phase/family/seed/count-scale strata. Selection ranks the size-controlled challengers by **unweighted synthetic-validation MAE**; L0 remains an additional full-size reference. No temperature/beta or joint tuning is performed.

In the completed full experiment, the size-controlled validation winner was **L3 with 50% change-point windows**: MAE 79,967.79 versus 80,670.33 for L0S. Full L0 remained substantially better on validation (66,502.53). The challenger selection does not imply that L3 should replace full L0.

On the same 600 COVID targets, L0 median/weighted MAE remained 1,436.10/1,427.57. L0S reached 1,587.08/1,593.72; L1 1,589.04/1,606.09; L2 1,624.85/1,616.34; L3-25 1,675.41/1,693.47; L3-50 1,834.14/1,901.03. Thus rebalancing did not improve this retrospective real sample, augmentation was worse despite its size-controlled synthetic-validation ranking, and weighting helped full L0 but hurt several sampled libraries. The original remains the preferred empirical reference; no library or generator settings were changed after inspecting these errors.

The study saves validation predictions, stratified validation scores, issued forecasts for every library before attaching any real labels, paired metrics by history/horizon/phase/region, region-cluster bootstrap differences, extreme-error and severe under/overprediction measures, and frozen initial-NB WIS/coverage/width/miss scores. Region bootstrap retains all within-state histories/horizons but does not remove shared epidemic shocks. COVID has been inspected repeatedly during project development, so these are exploratory comparisons, not untouched confirmation. Frozen NB assumptions are not recalibrated for modified libraries.

The `xai/` artifacts explain the chosen challenger's actual weighted forecast and its difference from full original-library median forecasting. Changed histories rematch under both libraries. The independent synthetic backgrounds are unchanged; no future labels generate masks. Saved analogue plots show original and changed-library paths together, while matching diagnostics identify family fractions, entropy, distances and future-change IQR. Case selection by completed gain/loss is retrospective only; attribution remains background-dependent and non-causal. The smallest sampled windows are not evidence of better predictive quality simply because a distance decreases.

Focused tests cover fixed-budget sparse balancing, row uniqueness, reproducibility, scale-invariant strata, embedding/bridge alignment, isolated library caches, fixed-aggregation matching, SIR conservation, weekly count representation, unchanged-transmission limits, and invalid change points. The repository sweep passed **186 tests**, including 22 robust tests. Full-run measurements were about 110 seconds, 1.9 GiB peak RSS and 20 MB stored artifacts on this machine; matching/inference timers include index construction and frozen NB simulation and are not pure nearest-neighbour microbenchmarks. Resource figures vary with concurrency.

Verified library-study artifacts: [full metrics](outputs/robust_smoa/library_study/full_2636262839d7f94f/metrics.csv), [synthetic selection](outputs/robust_smoa/library_study/full_2636262839d7f94f/selected_library.json), [validation strata](outputs/robust_smoa/library_study/full_2636262839d7f94f/validation_by_stratum.csv), [matching diagnostics](outputs/robust_smoa/library_study/full_2636262839d7f94f/matching_diagnostics.csv), and [baseline regression](outputs/robust_smoa/library_study/full_2636262839d7f94f/baseline_regression.json). All 1,200 real L0 median/weighted predictions matched the preceding Robust sMOA run exactly. Selected-row uniqueness, hashes, equal budgets, current source hashes and outcome-free issued files were verified. The final run took 95.8 seconds with 1,722 MiB process peak RSS; its stored artifacts occupy about 20.6 MB.

For RQ4, L3-50 selected 70.6% change-point analogues on average in the real cases, despite allocating 50% of windows to them. Its mean nearest distance (402.84) was smaller than L0S's (575.59), while its median MAE was worse (1,834.14 versus 1,587.08). Its family entropy was also lower (.87 versus 1.08). These are descriptive mechanism diagnostics, not causal findings. The parameter-shift stress test likewise favored full L0 (median MAE about 1.298 million) over the 100,000-window challengers (about 1.596–1.613 million). Consequently RQ1 finds no real-data benefit from the tested recompositions; RQ2 finds no real-data benefit from the new dynamics; RQ3 finds a library-dependent weighting interaction; RQ4 shows a changed analogue population and background-dependent forecast attribution, without establishing that closer matches yield better futures.

## Current main extension: synthetic-trained context-aware aggregation

```bash
.venv/bin/python -m experiments.run --experiment policy --smoke
.venv/bin/python -m experiments.run --experiment policy
```

This extension keeps the **full original library** and the original GAM/k=5/m=4,422/matching/omit-bridge semantics. It is inspired by the synthetic-only offline training and forecast-value ranking of [Murph et al. (2026), epiFFORMA](https://doi.org/10.1038/s41467-026-70937-8). It is **not epiFFORMA**: it combines summaries from one sMOA match, uses six features, and trains a classifier on the best ranked candidate rather than reproducing epiFFORMA's multi-model GBM forecasting-loss procedure. [Osthus et al. (2026)](https://doi.org/10.1371/journal.pcbi.1014630) motivates examining synthetic-to-real transfer; no Transformer, genetic observations, or new target-disease inputs are introduced.

The prior library experiment's full-versus-subset comparison changes neighbourhood density by a factor of 129.2: m selects 0.0342% of full L0 but 4.422% of a 100,000-window library. It therefore cannot isolate composition from size. Its historical selection excluded L0 and forced a challenger despite better L0 validation. **Future library selection now includes L0 as the no-change candidate**, with a full-library manifest supported. The old results and frozen L3-50 artifacts are preserved as exploratory evidence, not promoted as the default. The library study is not rerun as this stage's main improvement experiment.

Two focused modules were added: `policy.py` defines candidates/features and rank inference; `policy_experiment.py` handles synthetic role preparation, training, selection, evaluation, and policy TreeSHAP. Existing `tuning.py` was generalized with optional independent role namespaces, and the exact direct-SHAP wrapper now accepts the trained policy. Original forecasting, smoothing, matching, reference arrays and old results are preserved. No obsolete source files were removed in this stage. `configs/experiments.json`, the existing CLI, requirements, tests, and the single demo notebook were updated.

### Candidates and features

For each matched history, construct lower Q25 and upper Q75 of **anchored cumulative future trajectories**, plus the exact original median-increment reconstruction. Original sMOA need not lie between those two quantiles. Sort the three forecast values stably at each horizon and retain the mapping to identities 0=lower, 1=original, 2=upper. Training and inference use the same ordering. Targets are the rank with minimum synthetic absolute error, preferring the original candidate on exact ties. Equal candidate values recover their shared value exactly.

The six features are: (1) difference between the last-two and preceding-two means / max(preceding mean,1); (2) standard deviation / mean of the floored permitted history; (3) last floored observation / its permitted-window maximum; (4) IQR of cumulative analogue future changes / permitted-window mean; (5) horizon; and (6) history length. Original flooring and positive denominators handle zeros. No target forecast errors, long lags, annual seasonality, or fitted real-data normalization are used.

LightGBM was unavailable. The implementation uses one scikit-learn `RandomForestClassifier`, with 100 trees, max depth in [3,5], minimum leaf size 8, one worker and fixed primary seed 42. Training instance weights give equal total influence to family/count-scale strata (fixed scale boundaries 1,000 and 100,000). All curves have the same T/horizon schedule. This balances classification training rather than allowing huge raw forecast errors to dominate it; it does not guarantee scale-independent validation performance. Probabilities over the three ranks become nonnegative convex weights. The selected forecast is `(1-rho)*original + rho*policy`; rho=0 returns the original array exactly.

### Synthetic protocol and selection

Fresh independent namespaces separate training, validation, check, background and the original reference. The full study uses 288 training curves, 108 validation curves, 72 check curves and 9 backgrounds across three generator families and seeds 0/1/2. Namespace 3931189327 is distinct from the preceding robust/library studies; smoke uses a separate namespace. Every accepted whole-curve hash is checked against reference and previously explored role hashes, and roles are checked against each other. Old synthetic check datasets are development evidence and are not reused as the new policy check. A separate 36-curve change-point SIR stress set is generated with another fresh role and never used for selection.

Matching/candidate/feature records are cached compactly for training/validation. Both tree-depth choices train on those same records, without repeating matching. Check outcomes are evaluated only after policy selection. Model and selected parameters are persisted before reading real cases; all real issued predictions are saved before opening the outcome table. Role generation keys, content hashes, reference checksums, source hashes and package versions are recorded. Alternate fitting seeds 42/43/44 diagnose sensitivity at the frozen depth and do not select a new primary seed or rho.

The predefined rho grid is [0,.25,.5,1]. Promotion requires **more than 1% improvement in both validation MAE and history-normalized MAE**, RMSE no more than 5% worse than M0, and severe underprediction no more than 2 percentage points worse. Among eligible candidates minimize the mean of MAE/M0-MAE and NMAE/M0-NMAE, preferring simpler rho/depth on exact ties. If none qualify, rho=0 is retained; a validation-selected classifier is kept only as an unblended research candidate and for explanation. No real outcomes select features, hyperparameters, balancing rules or thresholds.

The output models are M0=exact original, M1=previous weighted median (temperature 4, beta 0), M2=unblended learned rank policy, M3=synthetically selected blend. New `context_policy/` runs preserve all earlier `robust_smoa/` outputs. Artifacts include candidate identities/values, rank ordering, probabilities, rho and forecast adjustments; overall/paired metrics and improved/worsened fractions; history/horizon/phase/region/family/scale/seed breakdowns; classifier serialization; both XAI layers; timings and peak memory. Existing frozen-initial NB intervals are explicit reference comparators, not target-trained calibration. The same initial mean=point+1 and simulated median recentering are extended to valid nonnegative policy points below 1; calibration is not guaranteed.

### Executed policy results

Full validation selected depth 5 but **rho=0**. Unblended validation raw MAE improved from 96,659.26 to 94,440.80 (2.3%), while NMAE worsened from 16.0420 to 16.2049. No blend passed the declared dual-metric promotion rule. The selected deployed forecast therefore remains M0, independently of what COVID scores later showed.

| Dataset | Original MAE | Weighted-median MAE | Unblended policy MAE | Selected blend MAE |
| --- | ---: | ---: | ---: | ---: |
| Synthetic validation | 96,659.26 | 96,962.25 | 94,440.80 | 96,659.26 |
| Fresh synthetic check | 112,767.70 | 112,665.37 | 106,146.04 | 112,767.70 |
| Fresh change-point stress | 829,493.56 | 825,814.71 | 818,022.55 | 829,493.56 |
| COVID early, 600 targets | 1,436.10 | 1,427.57 | 1,510.45 | 1,436.10 |

The fresh check's raw MAE improves but NMAE worsens from 3.7803 to 4.2764. Stress NMAE improves from 7.4478 to 7.0964. On COVID, unblended policy error is worse in both raw and normalized terms. These mixed results do not establish robust synthetic-to-real improvement. COVID cases were examined in prior project development and remain retrospective exploratory evidence. The no-change decision is a scientifically valid negative result; it was not overridden after external evaluation.

### Policy and final-forecast explanations

TreeSHAP explains class probabilities for **lowest/middle/highest forecast rank**. It uses synthetic-training tree-path statistics and explains independent synthetic validation features. Outputs include rank-specific beeswarms, waterfalls and importance by horizon/T. In the full run, last-observation-relative-to-maximum, history length and analogue dispersion had the largest mean absolute policy attributions. These are classifier preference explanations, not causal drivers or an automatic decomposition of sMOA counts.

Direct Kernel SHAP runs the actual selected forecasting function: original preprocessing/matching, cumulative candidates, sorting, six features, classifier probabilities and blending. It explains the final count and final-minus-original using independent synthetic backgrounds. Every changed history rematches; identical complete histories may reuse the engine cache. Since rho=0 was selected, the final-minus-original function is exactly zero. The independent policy TreeSHAP can still explain nontrivial unblended preferences, but those preferences do not affect the deployed forecast. Four retrospective cases and saved analogue paths complement the candidate/weight decomposition. Correlated observations, synthetic backgrounds, hybrid perturbations and limited coalition sampling remain important non-causal interpretation limits.

The updated notebook executed four cells in a fresh kernel and illustrates six hypothetical weekly observations, all three candidates, rank identities, learned probabilities, unblended and selected forecasts, TreeSHAP for that input, direct Kernel SHAP for that input, and separately labelled retrospective smoke results. Full/smoke metrics are never pooled.

The repository sweep passed **191 tests**, including 27 robust tests. New tests cover cumulative quantiles, original-candidate rank exceptions, stable ties, exact shared-value/rho-zero invariants, deterministic probabilities, real-label fitting rejection, changing real futures without changing issued policy predictions, synthetic-only selection, no-change fallback, and actual-policy rematching through the direct SHAP wrapper. Original tests remain intact. The first full run took about 63 seconds with 1.56 GiB process peak RSS: candidate training preparation 15.3s, validation 4.8s, classifier fitting/selection .49s, check preparation 3.0s, real matching preparation 2.1s, and XAI 15.6s. Timings include reference matching and NB preparation where stated and vary with cache reuse and concurrency.

Verified final policy artifacts: [metrics](outputs/robust_smoa/context_policy/full_4ddb0602f2ba4bf8/metrics.csv), [synthetic selection](outputs/robust_smoa/context_policy/full_4ddb0602f2ba4bf8/selected_policy.json), [validation grid](outputs/robust_smoa/context_policy/full_4ddb0602f2ba4bf8/validation_selection.csv), [policy TreeSHAP](outputs/robust_smoa/context_policy/full_4ddb0602f2ba4bf8/tree_shap_metadata.json), [direct SHAP](outputs/robust_smoa/context_policy/full_4ddb0602f2ba4bf8/explanation_metadata.json), and [baseline regression](outputs/robust_smoa/context_policy/full_4ddb0602f2ba4bf8/baseline_regression.json). All 1,200 original/weighted COVID forecasts reproduced the preceding Robust sMOA outputs exactly. Selected M3=M0 identity holds on every dataset, weights are valid, issued files contain no actuals, and current source hashes match the saved run. TreeSHAP additivity errors were at floating-point precision; direct additivity checks passed. Zero-attribution contrasts are rendered explicitly rather than using the plotting library's misleading zero-waterfall arrows. The final cached replay took 46.3s, including .65s training/selection, 3.09s end-to-end real issuance and 17.1s XAI, with 1,447 MiB peak RSS. The fresh synthetic check hashes were disjoint from 153 previously explored synthetic-role curves. The final repository sweep again passed 191 tests and the final notebook executed all four cells.
