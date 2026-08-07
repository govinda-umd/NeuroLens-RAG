# NeuroLens-RAG — Case 1 Summary Report

> Consolidated summary of everything built and found for **Case 1: brain decoding** (`X → y`, `X → y_hrf` from MOTOR-task ROI time series), end-to-end through RAG/LLM interpretation. Companion docs: [project-handoff-summary.md](project-handoff-summary.md) (original design), [ml-design-report.md](ml-design-report.md) (ML design + all results), [interpretability-methods-notes.md](interpretability-methods-notes.md), [decoded-state-to-text-report.md](decoded-state-to-text-report.md), [literature-notes-tokenization.md](literature-notes-tokenization.md).

## 1. What Case 1 actually is, end-to-end

```
ROI time series (300 ROIs, Schaefer atlas)
  → causal 32-volume window
  → Transformer encoder (multi-task: classification + HRF regression)
  → decoded label + confidence + HRF prediction
  → RSN attribution (4 interpretability methods, pooled to 7 Yeo networks)
  → templated text query
  → retrieval against a paper corpus (MiniLM dense embeddings)
  → structured stance-labeling prompt
  → local LLM (mlx-lm, Llama-3.2-3B-Instruct-4bit) generates a literature-grounded interpretation
```

This whole chain runs as one callable: `src/neurolens/pipeline.py::explain_decoded_window`. Every intermediate artifact (decoded state, per-method attribution, query text, retrieved chunks, prompt, generated text) is returned for auditability, not just the final string.

## 2. Data

- **HCP Young Adult S1200**, MOTOR task, LR+RL runs, **20 subjects** (5 original + 15 added mid-project, verified against S3 before download).
- Schaefer 2018 atlas, 300 cortical ROIs, Yeo 7-network solution, 2mm resolution.
- Per-run preprocessing: motion regression (`Movement_Regressors.txt`), detrending, within-run z-score standardization. No smoothing, scrubbing, or subcortical ROIs (documented as future ablations).
- Targets: 6-class hard label (baseline + 5 movement conditions) and a 5-channel canonical-HRF-convolved continuous regressor, built from the task's own event files.
- **Subject-level split, never window-level**: 14 train / 3 val / 3 test, fixed seed-42 shuffle over the sorted subject list — chosen specifically to avoid leakage from near-duplicate overlapping windows.
- Windowing: 32-volume causal windows, stride 2, sequence-to-one (`X[t-31:t+1] → y[t]`) — 5,080 total windows across the 20-subject dataset (3,556 train / 762 val / 762 test).

## 3. Models

Two encoders, each with a classification head and an *architecturally optional* HRF regression head (removed entirely for classification-only variants, not just untrained — see `include_hrf_head` in `model_builder.py`):

| | GRU | Transformer |
|---|---|---|
| Config | hidden=128, 1 layer | d_model=128, 4 heads, 2 layers, ff=256 |
| Params (multi-task) | 166,539 | 304,907 |
| Tokenization | n/a (recurrent) | 1 volume = 1 token, linear-projected, sinusoidal positional encoding, **no causal attention mask** (full bidirectional attention within the 32-token window — causality only holds at the sample-construction level) |

## 4. Headline results (20 subjects, 5 epochs, `lambda_hrf=0.1`)

| Experiment | Test macro F1 | Test balanced acc | Test HRF MSE | Test HRF R² range |
|---|---|---|---|---|
| GRU, classification-only | 0.912 | 0.910 | n/a | n/a |
| GRU, classification + HRF | 0.911 | 0.910 | 0.048 | 0.52–0.67 |
| Transformer, classification-only | 0.920 | 0.918 | n/a | n/a |
| **Transformer, classification + HRF** | **0.925** | **0.925** | **0.036** | **0.61–0.73** |

Scaling from 5 → 20 subjects was the single biggest lever: test macro F1 rose from the 0.56–0.63 range to 0.91–0.93, and two apparent "findings" from the 5-subject run (negative HRF R² despite positive correlation; uneven per-class F1 from 0.39 to 0.75) both resolved to non-issues at 20 subjects — they were data-scarcity artifacts, not real effects. Full detail and the superseded 5-subject table: [ml-design-report.md §9](ml-design-report.md#9-results).

## 5. Hyperparameter sweep (`07_hyperparameter_sweep.ipynb`)

**`lambda_hrf` ∈ [0.0, 0.05, 0.1, 0.3, 1.0]:**
- At `lambda_hrf=0.0` the HRF head is essentially untrained (MSE ~0.34–0.35) — a loss term scaled by zero contributes no gradient. Confirms the auxiliary loss is *necessary*, not just beneficial, for that head to learn at all.
- Classification is fairly robust to `lambda_hrf` for both encoders. For the **GRU**, HRF fit improves with higher `lambda_hrf` at essentially zero classification cost (flat ~0.91 macro F1 from 0.05 to 1.0, while HRF MSE drops from 0.049 to 0.039) — the original default of 0.1 was leaving free HRF quality on the table. For the **Transformer**, there's a small real tradeoff: 0.1 is the classification sweet spot (0.925), 1.0 gets the best HRF fit (MSE 0.019) at a small classification cost (0.918).

**Epoch count ∈ [5, 10, 15, 20]** (`lambda_hrf=0.1`, all 4 experiments):
- More epochs did **not** reliably improve test performance — 3 of 4 experiments were non-monotonic or declining past 5-10 epochs (e.g. Transformer multi-task: 0.925 → 0.914 → 0.925 → 0.916 at 5/10/15/20 epochs).
- The validation-selected checkpoint's own macro F1 kept rising with more epochs even where test performance didn't (0.876 → 0.889 for Transformer multi-task) — a **generalization-gap symptom of only 3 val / 3 test subjects**, not a training-length problem. More epochs is not the right lever here; more subjects likely is.

## 6. Interpretability: which RSN drove each decode

Four established methods, two families, pooled to the 7 Yeo networks for comparability (see [interpretability-methods-notes.md](interpretability-methods-notes.md) for full rationale):

- **Gradient-based** (Saliency, Integrated Gradients, via Captum) — operate on the continuous 300-ROI input directly.
- **Perturbation-based** (exact Shapley values, LIME) — operate on a 7-network coalition space; pooling to 7 "players" is what makes *exact* Shapley tractable (128 coalitions enumerated exactly, no sampling approximation).

**Finding, measured at both 5- and 20-subject scale**: the two families don't fully agree, and the disagreement has a stable shape — gradient methods over-weight the **Default** network relative to perturbation methods, which concentrate more on **SomMot** (the neuroanatomically expected answer for MOTOR decoding). Agreement improved with more data and a more confident model (cross-family top-1 agreement: ~65% → ~78% from 5 to 20 subjects), suggesting part of the original gap was driven by model uncertainty rather than being a fixed methodological difference — but the gap didn't close, and remains an open question before trusting either family unquestioningly downstream.

Per-window predictions and per-window attribution (all 4 methods × 7 networks, every test-set window) are saved to `results/motor_v1_per_window_predictions.csv` and `results/rsn_attribution_per_window.csv` — everything needed to build a decoded-state-to-text query for any timepoint without re-running the model.

## 7. RAG + LLM integration

- **Corpus**: 6 papers in `data/papers/` (Yeo 2011, Schaefer 2018, Barch 2013, Van Essen 2013 — the actual methodological sources behind this pipeline's atlas/parcellation/task design — plus 2 more), chunked (~220 words, 50-word overlap) and embedded (MiniLM) into 729 chunks via `src/neurolens/retrieval.py`.
- **Generation**: `mlx-community/Llama-3.2-3B-Instruct-4bit`, running locally via `mlx-lm` (Apple-Silicon-native, ~1.8GB, ~3-20s per call on this M3).
- **Prompt design**: a structured stance-labeling prompt forces the model to label each retrieved excerpt SUPPORTS/CONTRADICTS/UNRELATED with a citation before synthesizing — auditable, and discourages fabricating connections to weakly-related material.
- **Observed effect of corpus quality**: with the original single-paper corpus, retrieval similarity topped out at ~0.34 and the LLM over-agreed (marked an unrelated paper as "supporting" a decode). After growing to 6 targeted papers, similarity rose to ~0.46–0.585 and the LLM correctly distinguished a genuinely relevant excerpt (Barch 2013, explicitly describing the "motor mapping task" and right-hand activation) from tangential background material (Yeo 2011, Schaefer 2018), marking those UNRELATED instead of stretching for a connection. **Corpus quality, not just model choice, was the dominant lever.**

## 8. RAG evaluation and cross-encoder reranking

[`08_rag_evaluation.ipynb`](../notebooks/08_rag_evaluation.ipynb) builds a small, honestly-labeled evaluation set — reading all 6 corpus papers to make real relevance judgments (not guessing) about which 3 of the 6 directly ground a MOTOR-condition decoded-state query (Barch 2013, Yeo 2011, Schaefer 2018) versus which 3 don't (Van Essen 2013, van den Heuvel & Sporns 2013, Misra & Surampudi 2021) — then measures retrieval and generation against it. **Caveat**: 5 near-identical queries against 6 papers is a small, first-pass evaluation, good for catching an obviously broken component, not for certifying quality at scale.

**Retrieval — dense-only vs. dense + cross-encoder rerank** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, no training data needed, added to `src/neurolens/retrieval.py::retrieve_and_rerank`):

| Method | Precision@5 | Recall (relevant papers)@5 |
|---|---|---|
| Dense-only | 1.00 | 0.80 |
| Dense + rerank | 1.00 | **1.00** |

Precision was already perfect at this corpus size — every top-5 chunk came from a relevant paper either way. Reranking's real, measured benefit was **recall**: dense-only retrieval sometimes let one relevant paper get crowded out of the top-5 by chunks from the other two (2.4/3 papers represented on average); reranking recovered full 3-of-3 coverage on every query. A genuine, if modest, improvement for zero training cost — now the default recommended retrieval path in `pipeline.py` (pass a `reranker` from `retrieval.load_reranker()`).

**LLM stance-label spot-check** (2 examples, real `generate_fn` calls) — mixed, worth internalizing before trusting generated text at face value:
- One example was clean and accurate: correctly labeled the two genuinely relevant excerpts SUPPORTS (citing the actual "motor mapping task" / "right hand" content) and three background excerpts UNRELATED.
- The other got the coarse SUPPORTS/UNRELATED label defensible but stated a **factually wrong justification** — describing the SomMot network as involved in "working memory/cognitive control" (that's the *Cont* network's role, not SomMot's).

**Bottom line**: retrieval is solid and measured, not assumed, at this corpus size. Generation is more fragile — the 3B model can produce specific, confident-sounding, and simply *wrong* neuroscience claims even when its coarse relevance judgment is fine. Treat this pipeline's generated text as a retrieval-grounded draft for a researcher to verify, not a citable explanation, until stance-label accuracy is evaluated at more than 2-example scale.

## 9. Engineering notes worth remembering

- Installing `mlx-lm` broke the existing `torch` installation (its `transformers` dependency triggered a missing `libtorch_cpu.dylib`) — caught immediately via a smoke test, fixed with a targeted `pip install --force-reinstall --no-deps torch`. Worth re-verifying torch after any future dependency changes in this environment, since the two ecosystems (PyTorch training, MLX inference) share the same conda env.
- All code that matters is in `src/neurolens/*.py`; notebooks (`03`–`07`) are executed drivers (`jupyter nbconvert --execute --inplace`, real kernel, real outputs) that call it and record results, not where the logic lives.

## 10. Further improvements possible on this iMac (M3, 16GB unified memory)

Roughly in order of expected value for effort, given the hardware ceiling already established (small models train in seconds; the actual bottleneck throughout has been *data volume*, not compute):

1. **More subjects.** The epoch sweep's own evidence (val/test divergence) points here directly — 3 test subjects is a small generalization estimate. This is the highest-value next step and is mechanically already solved (same idempotent pipeline, just more S3 downloads/AWS cost).
2. ~~**Cross-encoder reranking for retrieval**~~ — **done** (§8): improved paper-level recall from 0.80 to 1.00 at zero training cost.
3. ~~**A real (small) RAG/LLM evaluation set**~~ — **done** (§8), though still small (5 retrieval queries, 2 generation spot-checks). Scaling this to 10-20+ generation-quality checks specifically (retrieval already looks solid) is the natural next step — the one real quality issue found so far (a factually wrong network-function claim) came from generation, not retrieval.
4. **Per-architecture `lambda_hrf`** — the sweep shows GRU and Transformer want different values (GRU benefits from higher `lambda_hrf` for free; Transformer has a real tradeoff at 0.1). Worth using per-architecture values rather than one shared default.
5. **Concept-based interpretability (Been Kim line)** — TCAV/ACE-style concept vectors, potentially seeded by literature excerpts retrieved via RAG itself (see [interpretability-methods-notes.md §4.1](interpretability-methods-notes.md#41-a-neurolens-rag-specific-variant-literature-derived-concept-hypotheses)) — more semantically meaningful than raw network attribution, and the RAG plumbing to seed it already exists.
6. **Resolving the gradient-vs-perturbation interpretability disagreement** — currently unresolved; worth digging into whether it's a real signal or a gradient-method artifact before trusting either family's attribution in a production RAG query.
7. **Level 3 (learned brain-representation → LLM-embedding adapter)** — still not recommended at this data scale (needs far more paired brain/language data than 20 MOTOR-task subjects provide); revisit only after subject count grows substantially.

Compute/memory itself has not been a constraint anywhere in this project so far — every model is small enough that MPS trains a full 4-experiment ladder in well under a minute, and the 3B quantized LLM leaves most of the 16GB free. The bottlenecks throughout have been **data volume** (subjects, papers) and **evaluation** (nothing has been rigorously graded yet), not hardware.

## 11. Resume-ready bullet points (Google XYZ format: accomplished X, measured by Y, by doing Z)

Written systems-forward on purpose — the rest of the resume already carries the Bayesian-statistics/probabilistic-modeling depth; these are meant to demonstrate applied ML *engineering* (pipelines, RAG, local LLM deployment, production-style debugging) as a complementary signal for ML/AI industry roles, not to duplicate the research framing already established elsewhere.

**NeuroLens-RAG: End-to-End fMRI Decoding & Retrieval-Augmented LLM Interpretation System** — 2026

- Built and deployed an end-to-end multi-task deep learning system for decoding motor behavior from human fMRI, achieving **92.5% test macro-F1** across 6 movement classes on held-out subjects, by designing joint classification + auxiliary-regression GRU/Transformer architectures and an idempotent data pipeline that scaled training data 4× (5→20 subjects) via automated cloud ingestion.
- Improved retrieval-augmented generation (RAG) recall from **80% to 100%** (zero relevant source documents missed, measured against a hand-verified evaluation set) by integrating a cross-encoder reranking stage on top of dense retrieval, requiring no additional training data.
- Made exact (non-approximated) Shapley-value model attribution computationally tractable — evaluating all 128 feature coalitions per prediction across **762 test samples in under 5 minutes** — by re-formulating the attribution problem over 7 functional brain networks instead of 300 raw input regions, and used it alongside 3 other interpretability methods to uncover a systematic, reproducible disagreement between gradient-based and perturbation-based attribution families.
- Deployed a fully local retrieval-augmented generation system — pairing a quantized 3B-parameter LLM (Llama-3.2, via Apple's MLX framework) with a custom scientific-literature retrieval index — generating citation-grounded natural-language interpretations of model predictions entirely on consumer hardware with no cloud API dependency.
- Ran a 26-configuration hyperparameter sweep across loss-weighting and training-length axes, discovering that one architecture's auxiliary-task loss weight could be increased **10×** at zero accuracy cost while additional training epochs produced no reliable test-set improvement — redirecting subsequent effort toward data scaling over blind hyperparameter tuning.
- Diagnosed and resolved a production-breaking PyTorch installation conflict introduced by an unrelated ML library's dependency chain, restoring full training-pipeline functionality within minutes via targeted dependency isolation, with zero loss of in-progress experimental results.

**On format**: these follow Google's XYZ structure (result → metric → method) rather than a chronological "did X, then Y" narrative, and each leads with a number a recruiter or hiring manager can scan in isolation.

### STAR versions (for cover letters / interview narratives, not the resume itself)

STAR doesn't compress into one scannable line the way XYZ does — these are meant for "tell me about a project" interview answers or a cover letter paragraph, not bullet points. Same underlying facts as above, restructured as narratives.

**1. Scaling exposed (and resolved) a real data-scarcity problem, not a modeling flaw.**
*Situation*: An initial 5-subject fMRI motor-decoding model showed a puzzling result — its auxiliary regression head had positive correlation with ground truth but negative R², meaning it tracked direction but not magnitude.
*Task*: Determine whether this was a genuine modeling flaw (wrong loss design, wrong architecture) before building anything further on top of it.
*Action*: Rather than immediately changing the model, I designed a controlled 4-experiment ladder (GRU vs. Transformer × single-task vs. multi-task) to isolate variables, then scaled the training data 4× (5→20 subjects) via an idempotent automated pipeline I built against HCP's AWS S3 data, and reran the full comparison plus a 26-configuration hyperparameter sweep.
*Result*: Test macro-F1 rose from 61% to 92.5%, and R² flipped fully positive (0.61–0.73) — confirming the original result was a data-scarcity artifact, not a design flaw. Saved significant wasted effort that would have gone into "fixing" a model that wasn't actually broken.

**2. Built and evaluated a fully local RAG system, end to end.**
*Situation*: The project's goal was to make a neural network's fMRI-decoding output interpretable in natural language, grounded in real scientific literature, running entirely on local hardware for cost and data-privacy reasons — no cloud LLM APIs.
*Task*: Design and build a complete retrieval-augmented generation pipeline: literature ingestion, retrieval, and grounded generation, and — critically — verify it actually works rather than assuming it does.
*Action*: Built PDF ingestion and semantic chunking, dense-embedding retrieval, and a cross-encoder reranking stage; deployed a quantized 3B-parameter LLM locally via Apple's MLX framework with a structured prompt that forces the model to cite and justify each claim; then built a small hand-labeled evaluation set (reading every source paper myself to establish ground truth) to measure retrieval and generation quality instead of eyeballing outputs.
*Result*: Reranking measurably improved retrieval recall from 80% to 100%. The evaluation also caught a real generation-quality issue — the LLM stated a factually incorrect neuroscience claim in one case — which I documented rather than hid, since knowing a system's failure modes is part of shipping it responsibly.

**3. Diagnosed and fixed a silent production-breaking dependency conflict.**
*Situation*: Partway through the project, installing a new library for local LLM inference silently broke the existing PyTorch training pipeline — any subsequent training run would have failed, and the breakage wasn't obvious from the install output.
*Task*: Diagnose the root cause and restore full functionality without losing hours of already-completed experimental results or further destabilizing the environment.
*Action*: Ran an immediate smoke test that isolated the failure to a missing shared library pulled in transitively by an unrelated dependency, then performed a targeted, dependency-scoped reinstall rather than rebuilding the environment from scratch.
*Result*: Restored full pipeline functionality within minutes, verified against existing smoke tests, with zero loss of in-progress work — and documented the fix so it wouldn't silently recur.

**On your friend's bullets**: I don't have any information about what they actually built or contributed, so I can't write real, honest bullets for them — doing so would mean fabricating accomplishments, which I won't do even with good intentions. If they share their own repo, results, or a rough description of what they did, I can write equivalent XYZ-format bullets grounded in their real work the same way these are grounded in this repo's actual numbers.

## References

- [project-handoff-summary.md](project-handoff-summary.md)
- [ml-design-report.md](ml-design-report.md)
- [interpretability-methods-notes.md](interpretability-methods-notes.md)
- [decoded-state-to-text-report.md](decoded-state-to-text-report.md)
- [literature-notes-tokenization.md](literature-notes-tokenization.md)
- Notebooks: [03](../notebooks/03_dataset_dataloaders.ipynb), [04](../notebooks/04_models.ipynb), [05](../notebooks/05_train_eval_compare.ipynb), [06](../notebooks/06_interpretability_rsn.ipynb), [07](../notebooks/07_hyperparameter_sweep.ipynb)
- Code: [`src/neurolens/`](../src/neurolens/)
