# Case 2 & 3: Next-Timepoint ROI Forecasting — Design Plan

> Design specification, not implemented. Written per discussion after Case 1 was brought to a good stopping point (see [case1-summary-report.md](case1-summary-report.md)). **Working definition, stated explicitly for confirmation before building anything**: Case 2 and Case 3 are the *same underlying task* — predicting the next ROI timepoint(s), a forecasting/generative problem, distinct from Case 1's decoding — tackled by two different modeling families for direct comparison. **Case 2** = deep sequence models (GRU/Transformer, extending Case 1's architecture family). **Case 3** = Bayesian PGM + dynamical-systems models (SLDS/rSLDS-style), the research direction the user wants to pursue as a differentiated angle. If this isn't the intended split, everything below needs re-scoping.

## 1. Why this is a genuinely different problem from Case 1

Case 1 is discriminative: `X[t-31:t] → y[t]` (a label for a timepoint that's *inside* the input window). Case 2/3 is generative/forecasting: `X[t-L:t] → X[t+1]` (or a short horizon `X[t+1:t+h]`) — the target is never in the input, and there's no task label involved at all. This matters for three concrete reasons:

- **No leakage subtlety to worry about** the way Case 1 had (§2 of [ml-design-report.md](ml-design-report.md) — the window included its own target timepoint). Here causality is unambiguous: input strictly precedes target.
- **No classes, no class imbalance, no HRF targets.** The task is continuous-valued regression over all 300 ROIs simultaneously — evaluation shifts entirely to regression metrics (MSE, R², possibly held-out log-likelihood for Case 3's probabilistic models).
- **Baseline vs. task period no longer matters the same way.** Case 1 treated "baseline" as a real class; here, forecasting should probably run over the *entire* continuous run (task blocks and inter-block rest alike), since the goal is modeling brain dynamics generally, not decoding a specific condition.

## 2. Data reuse — no new pipeline needed

Both cases can reuse the exact same processed bundles from Case 1 (`data/processed/hcp_ya_s1200/runs/`, 20 subjects, MOTOR task) — `X.npy` already contains the full continuous ROI time series; nothing about `02_data.ipynb` needs to change. Only the **windowing logic** changes:

```python
# Case 1 (existing, data_setup.py):
X[t-31:t] -> y[t], y_hrf[t]        # target inside the window

# Case 2/3 (new):
X[t-L:t] -> X[t+1]                  # one-step-ahead forecast, target strictly after
# or, for multi-step:
X[t-L:t] -> X[t+1 : t+1+h]          # h-step horizon
```

Same subject-level 14/3/3 split (`MOTOR_TRAIN/VAL/TEST_SUBJECTS` in `data_setup.py`) applies unchanged — reuse, don't re-derive. A new `build_forecasting_window_index()` alongside the existing `build_window_index()` in `data_setup.py` is the only new data-layer code needed (same file, new function, not a new module).

## 3. Case 2 — deep sequence forecasting baselines

Directly reuses `model_builder.py`'s `GRUDecoder`/`TransformerDecoder` shapes with a different head: instead of a classifier (→6) and HRF regressor (→5), a single **forecast head**: `Linear(hidden_size or d_model → 300)`, trained with MSE (or Gaussian NLL, see below) against `X[t+1]`.

- **v1 (recommended starting point)**: one-step-ahead, sequence-to-one — architecturally almost identical to Case 1's existing models, swap the loss and output dimension. Cheapest possible way to get a real baseline number before investing in Case 3.
- **v2 (stretch)**: multi-step autoregressive rollout (feed the model's own t+1 prediction back in to predict t+2, etc.) — the real test of whether the model has learned actual dynamics vs. just short-range smoothing. Rollout error growth rate is itself a diagnostic (models that learn true dynamics degrade slower over the horizon than ones that memorize local correlations).
- **Loss choice worth deciding up front**: plain MSE assumes homoscedastic, isotropic noise across all 300 ROIs, which is almost certainly wrong (different ROIs have different noise characteristics). A Gaussian NLL with a learned per-ROI (or full covariance) variance is a small change with a real payoff: it gives you a proper likelihood, which is what makes Case 2 and Case 3 **comparable on the same footing** (§5) rather than comparing MSE against a Bayesian model's marginal likelihood, which isn't apples-to-apples.

## 4. Case 3 — Bayesian PGM + dynamical-systems models

**Core model: Switching Linear Dynamical System (SLDS)**, specifically the **recurrent SLDS (rSLDS)** formulation (Linderman et al. 2017), implemented via **`lindermanlab/ssm`** (github.com/lindermanlab/ssm) rather than from scratch — a purpose-built, actively-referenced library for exactly this model class, with existing tutorial notebooks for rSLDS specifically. Building this from scratch would be a large, error-prone undertaking (variational EM / Gibbs sampling for a switching model is nontrivial); using an established library is the right call here, the same way Captum was the right call for interpretability rather than hand-rolling gradient attribution.

Model structure (for grounding, not final spec — tune during implementation):

```
discrete regime:     z_t  ~  Categorical(recurrent transition depending on continuous state)
continuous state:    x_t  =  A_{z_t} x_{t-1} + b_{z_t} + noise      (regime-dependent linear dynamics)
observation:         y_t  =  C x_t + d + noise                       (y_t = the 300-dim ROI vector)
```

- The **discrete regimes `z_t` are themselves interpretable outputs** — a structural advantage over Case 2's black-box hidden state. This creates a direct, falsifiable research question: *do the model's unsupervised discrete regimes correspond to the known MOTOR task conditions, without ever being told the labels?* If yes, that's a genuinely interesting, portfolio-worthy result (unsupervised discovery of task structure from dynamics alone) — testable by cross-tabulating inferred `z_t` against the existing `y` labels from Case 1's bundles (already on disk, free to reuse for this validation even though Case 2/3 training itself doesn't use them).
- **POSSM as a stretch/v2 direction, not v1**: POSSM (Azabou & Dyer, discussed in [literature-notes-tokenization.md](literature-notes-tokenization.md)) hybridizes a spike/event-style tokenizer with a recurrent SSM backbone for speed and cross-session generalization — a substantially bigger engineering lift (custom tokenization + hybrid architecture) than a library-backed rSLDS. Reasonable to reference conceptually now, but not a v1 build target given the 20-subject data scale.
- **Continuous-state dimensionality** is a real hyperparameter to sweep (state dim likely << 300, since the ROI observations are assumed to be a linear/noisy projection of a lower-dimensional latent dynamical state) — this is itself an interesting output: what's the effective dimensionality of MOTOR-task brain dynamics, per this model class?

## 5. Evaluation — designed to connect back to Case 1, not float free

| Metric | Case 2 | Case 3 |
|---|---|---|
| One-step-ahead MSE / R² per ROI | ✓ | ✓ (via posterior predictive mean) |
| Held-out log-likelihood | only if using Gaussian-NLL loss (§3) | ✓ natively (it's a generative model) |
| Multi-step rollout error growth | ✓ (v2) | ✓ (v2, via simulating from the fitted dynamics) |
| **Regime/label correspondence** | n/a (no discrete state) | ✓ — cross-tabulate `z_t` against Case 1's `y` labels |
| **Downstream decoding probe** | linear probe on hidden state → predict `y` | linear probe on `x_t` (continuous latent) → predict `y` |

The **downstream decoding probe** is the most valuable cross-case comparison: train a simple linear classifier on top of each model's learned internal representation (Case 2's hidden state, Case 3's inferred `x_t`) to predict Case 1's movement labels, *without* fine-tuning the forecasting model itself. This directly tests whether a representation learned purely from unsupervised forecasting captures task-relevant structure — a meaningfully different and arguably more interesting claim than Case 1's supervised decoding accuracy, and a natural bridge between all three cases.

## 6. Connection to the RAG/concept-vector loop

Directly extends [interpretability-methods-notes.md §4.1](interpretability-methods-notes.md#41-a-neurolens-rag-specific-variant-literature-derived-concept-hypotheses): that loop proposed using RAG-retrieved literature to generate candidate *concept* phrases, then testing them as CAVs against Case 1's hidden representation. Case 3 gives a second, independent test bed for the same candidate concepts — **if a literature-derived concept (e.g., "bilateral motor coordination") is real, it should show up two independent ways**: as a CAV direction Case 1's decoder is sensitive to, *and* as something correlated with Case 3's unsupervised discrete regimes or continuous latent state. Convergence between the two would be meaningfully stronger evidence than either alone — genuinely worth designing for once both exist, not just a nice-to-have.

## 7. Resume/portfolio framing

Case 3 in particular is the concrete, buildable evidence for the research direction described in conversation (PGM + dynamical systems in a Bayesian setting) — rather than a claim on a resume, it becomes a real project with real comparative results against a deep-learning baseline (Case 2) on identical data. That comparison — "does structured Bayesian dynamics modeling capture something a black-box sequence model doesn't, on the same forecasting task" — is a substantive, well-posed research question suitable for a portfolio project *and* answerable in a bounded amount of work, unlike an open-ended "build something transformer-level" framing.

## 8. Compute feasibility (M3, 16GB)

Genuinely favorable — more so than Case 1 in some ways. `ssm`'s inference (variational EM / Gibbs sampling for rSLDS) is CPU-bound and doesn't need MPS/GPU acceleration at all; state dimensions in the tens (not hundreds) keep it lightweight. Case 2's GRU/Transformer forecasting models are architecturally near-identical in size to Case 1's (~165K-305K params), so training cost is comparable to what's already been run repeatedly this project. No new hardware or memory concerns expected.

## 9. Proposed build sequencing

1. **Case 2 first** — cheapest path to a real baseline number, since it's a small, well-understood modification of existing `model_builder.py`/`engine.py` code (new windowing function, new head, new loss). Gets a forecasting result on the board fast.
2. **Case 3 second** — install `ssm`, fit a plain SLDS (no recurrence) as a sanity-check baseline before the fuller rSLDS, sweep state dimensionality, then compare against Case 2 on the metrics in §5.
3. **Downstream decoding probe (§5)** and **regime/label correspondence (§4)** as the payoff analyses once both models exist — these produce the actually-interesting comparative findings, not the raw forecasting MSE numbers themselves.
4. RAG/concept-vector convergence test (§6) — later, once Case 1's concept-vector loop (§4.1 of interpretability-methods-notes.md) is itself built, which it currently isn't.

## Open questions

- Confirm the Case 2/Case 3 definition stated at the top — if wrong, this entire plan needs re-scoping before implementation starts.
- Forecast horizon for v1: single-step only, or commit to multi-step from the start?
- Loss choice for Case 2 (§3): plain MSE now vs. Gaussian NLL from the start, given it's needed for a fair Case 2 vs. Case 3 comparison later anyway.
- Whether to also forecast `y_hrf`-style structure, or keep Case 2/3 purely about raw ROI dynamics (the latter seems cleaner and more aligned with "understand brain dynamics" rather than re-importing Case 1's task-specific targets).

## References

- [case1-summary-report.md](case1-summary-report.md)
- [ml-design-report.md](ml-design-report.md)
- [interpretability-methods-notes.md](interpretability-methods-notes.md) §4.1
- [literature-notes-tokenization.md](literature-notes-tokenization.md) — POSSM background
- [lindermanlab/ssm](https://github.com/lindermanlab/ssm) — rSLDS implementation
- [Recurrent SLDS tutorial notebook](https://github.com/lindermanlab/ssm/blob/master/notebooks/4-Recurrent-SLDS.py)
- Linderman et al. 2017, "Bayesian Learning and Inference in Recurrent Switching Linear Dynamical Systems" — original rSLDS: [github.com/slinderman/recurrent-slds](https://github.com/slinderman/recurrent-slds)
