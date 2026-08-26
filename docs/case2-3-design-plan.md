# Cases 1-3: A Representation Learning Throughline — Design Plan

> Design specification for Case 2 and Case 3, revised after correcting the framing of the whole project. **The organizing theme across all three cases is representation learning** — decoding (Case 1), contrastive multimodal alignment (Case 2), and probabilistic/generative dynamics (Case 3) — the same throughline as Eva Dyer's group's work (POYO, POSSM; see [literature-notes-tokenization.md](literature-notes-tokenization.md)). This supersedes the previous version of this document, which mis-scoped Case 2 as plain next-timestep forecasting.

## 0. Corrected framing

**Case 1** was originally conceived as a stepping stone toward representation learning — build a working encoder, verify it end-to-end, learn the pipeline. It ended up also being a strong standalone result (92.5% test macro-F1 multi-task decoding) — a genuine bonus, not the point. Reframed correctly: Case 1's Transformer produces a pooled 128-dim representation trained jointly for classification and HRF regression; that representation *is* the actual artifact of interest, and the decoding accuracy is one way (not the only way) of evidencing that it's good. This reframing doesn't require new code — it changes how the existing work gets described (see §5, resume framing) and motivates the concept-activation-vector work in §4.

**Case 2** is not forecasting. It's **contrastive multimodal representation learning**: two encoders — one for ROI time series, one for the condition/label — trained to align in a shared embedding space via a contrastive objective, CLIP-style (Radford et al. 2021). "Predicting the next timepoint" reappears here, but properly motivated, as an optional generative capability *conditioned on* the learned joint representation (§2.4) — not as the primary objective the way the previous version of this doc had it.

**Case 3 — renamed, not just revised.** The Bayesian PGM/dynamical-systems idea (SLDS/rSLDS, §3 below) described in the original version of this doc is now treated as a *separate*, still-unscoped future direction, not "Case 3." As of this session, **Case 3 refers to a discriminative brain–HRF co-embedding** — the same window's hemodynamic-response signal aligned with the brain encoder in a shared space, exactly the way Case 2 aligns brain and text (§3.5). Confirmed discriminative rather than generative specifically because TCAV needs a differentiable scalar decision to test, which a discriminative alignment provides and a sequence-generation decoder does not.

## 1. Case 1, reframed (no new code, just correct the story)

Nothing to build here — this section exists so the resume writeup and any future paper/writeup describes Case 1 accurately:

- The shared Transformer encoder (`model_builder.py::TransformerDecoder`, minus its heads) is a **multi-task-trained representation** of a 32-timepoint window of brain activity into a 128-dim vector, optimized jointly for a discriminative signal (movement class) and a continuous generative signal (HRF regression).
- The strong decoding accuracy (92.5% macro-F1) is evidence the representation captures task-relevant structure — but it's one probe among several. §4 (CAV testing) adds a second, independent probe: does the representation's *directions* correspond to human-interpretable concepts (movement effector, laterality), not just its ability to feed a linear classifier well.

## 2. Case 2 — Contrastive multimodal representation learning (ROI timeseries ↔ condition)

### 2.1 The setup

Two encoders, trained jointly with a contrastive objective — structurally identical to CLIP's image-text setup, with brain windows in place of images and condition descriptions in place of captions:

```
brain encoder:  X[t-31:t]  (32 x 300 ROI window)  -> z_brain   (d-dim)
label encoder:  condition description               -> z_text    (d-dim, 6 fixed prototypes)

loss: temperature-scaled cosine-similarity cross-entropy against all 6 prototypes
  logits = (z_brain @ z_text.T) / temperature      # [batch, 6]
  loss = cross_entropy(logits, true_class)
```

**Precision worth stating rather than glossing over**: this is *not* literal CLIP-style InfoNCE with in-batch random negatives — CLIP has a unique caption per image and an effectively unbounded implicit vocabulary, so it must sample negatives from the batch. Here the text side is a small, fixed, *closed* vocabulary (6 conditions), so the strictly stronger and more standard thing to do is compare against **all 6** known prototypes every step, not just whichever ones happen to land in the current batch. The result is a hybrid, honestly named: **supervised contrastive learning against semantically-initialized class prototypes** — the same mechanics as CLIP's similarity-based classification head, but without CLIP's large-open-vocabulary motivation for batch-sampled negatives.

- **Brain encoder**: reuse `model_builder.py`'s `GRUDecoder`/`TransformerDecoder` architecture, with the classification/HRF heads removed and replaced by a projection head to the shared embedding dimension `d` (small MLP, standard CLIP-style projection).
- **Label encoder**: two options, worth building the cheap one first —
  1. **v1 (cheap, not built)**: a learned embedding table, one vector per class (6 learnable prototypes) — the label side degenerates to a small lookup. Considered and skipped: **decided to go straight to v2**.
  2. **v2 (chosen, built)**: encode the condition's **natural-language description** ("right hand movement") with the same MiniLM sentence-transformer already used in `retrieval.py` for paper retrieval — reusing an existing model, not adding a new dependency. This makes the setup genuinely CLIP-like (text side, not just class-index side) and opens the door to richer descriptions than a bare class name — e.g. incorporating Case 1's RSN attribution output ("right hand movement, primarily somatomotor") as the paired text, connecting Case 2 directly to the interpretability work in Case 1 rather than being a disconnected new project. Since there are only 6 fixed condition descriptions, MiniLM only ever needs to encode those 6 strings once — the "text encoder" reduces to a small trainable projection on top of a frozen, precomputed [6, 384] matrix, which is also the sample-efficient right call given there's no room to learn anything meaningful in the text encoder itself from just 6 examples.

### 2.2 Why this is a meaningfully different (and more modern) objective than Case 1

Case 1 asks "does a linear/softmax layer on top of this representation predict the label" (discriminative probing). Case 2 asks "can the representation be trained so that brain activity and condition semantics live in the same geometric space" (representation alignment) — the difference between training a classifier and training an embedding space, which is the same distinction between a standard CNN classifier and CLIP in computer vision. It's self-supervised in the sense that the label is used as a *pairing signal* for contrastive alignment, not as a direct supervised target the way Case 1's cross-entropy loss uses it.

### 2.3 Evaluation — two capabilities the contrastive setup gives for free

1. **Prototype-based classification via the joint embedding** (*not* zero-shot — all 6 classes are seen during contrastive training, since the true label picks the positive text prototype each step; true zero-shot would mean classifying a condition whose text-brain pairing was never trained on, which this setup doesn't attempt): classify a brain window by finding its nearest condition-embedding by cosine similarity, with no separate classifier head. Directly comparable, as a representation-quality metric, against Case 1's supervised decoding accuracy (92.5%) — a meaningfully lower number wouldn't be a failure, it would quantify the gap between "aligned representation" and "task-optimized representation," itself an interesting reportable finding.
2. **Cross-modal retrieval**: given a text query ("left foot movement"), retrieve the most characteristic brain windows, and vice versa. **This reuses `retrieval.py`'s cosine-similarity retrieval mechanism verbatim** — the same code that retrieves paper chunks for RAG can retrieve brain windows once both live in embedding space, just pointed at a different embedding matrix. Worth building as literal code reuse, not just a conceptual parallel — a concrete "shared retrieval infrastructure across modalities" systems-design point.

### 2.4 Generation (v2/stretch) — where "predict the next timepoint" reappears

Once `z_brain`/`z_text` exist, they can condition a small generative decoder to predict future ROI activity (`X[t+1:t+h]`) — the original forecasting idea, now motivated as *conditional* generation from a learned multimodal representation rather than an unconditional sequence model. Not v1 — build the contrastive alignment and its two evaluations first; conditional generation is a natural extension once the representation is validated.

### 2.5 Case 3 (confirmed): discriminative brain–HRF co-embedding

Structurally identical to Case 2 (§2.1), with the HRF signal in place of text:

```
brain encoder:  X[t-31:t]  (32 x 300 ROI window)         -> z_brain  (d-dim)
HRF encoder:    y_hrf[t]   (the SAME window's HRF trace)  -> z_hrf    (d-dim)

loss: contrastive alignment between z_brain and z_hrf
```

Two design points worth stating precisely rather than assuming:

- **This is safe, and it's worth being precise about *why*, since HRF has a real leakage trap elsewhere in this project.** `y_hrf` is derived from full-run event convolution, and using it as a *forecast input* (predicting *future* activity from it) would leak event-timing information a causal system wouldn't have. Co-embedding the *same window's* HRF as an alignment target is a different use entirely — exactly what Case 1's existing auxiliary HRF head already does safely (§2.2 of `project-summary.md`). No new leakage risk.
- **Unlike Case 2's 6 fixed text prototypes, HRF is continuous and per-window** — every window has its own HRF trace, not a value drawn from a small closed set. That means, for the first time in this project, both sides of the contrastive loss have genuine in-batch negatives, making a literal *symmetric* CLIP loss (brain→HRF and HRF→brain, averaged) possible — something Case 2's closed 6-item text vocabulary structurally couldn't support (§2.1's "not literal in-batch-negative InfoNCE" note).

**Open design question, not yet resolved**: with three modalities (brain, text, HRF) potentially sharing structure, does a "concept" need to show up consistently across all three to count as validated, or does brain-vs-text validation (as CAV/TCAV already does for Case 2) remain sufficient on its own? Text enters Case 3 through the same frozen-MiniLM-plus-trained-projection path already built for Case 2 — that part isn't new work. Deliberately parked for a dedicated pass once Case 3 itself is built, not decided in passing here.

**Explicitly out of scope for the current concept-based model comparison**: the generative direction (HRF→brain, e.g. activating a "rightness" concept and inspecting the synthetic brain signal that comes out) and the general question of how to evaluate a generative model's concept-sensitivity. Both noted as a separate, later idea.

### 2.5.1 Case 3 build + 100-subject validation — **done**

Built in `src/neurolens/case3.py` (`HRFEncoder`, `BrainHRFModel`, `symmetric_contrastive_loss`, `train_case3`) and validated end-to-end in [`17_case3_brain_hrf_coembedding.ipynb`](../notebooks/17_case3_brain_hrf_coembedding.ipynb). Full numbers: [`results/case3_validation_results.json`](../results/case3_validation_results.json).

Because Case 3 never sees a class label during training, it has no classifier head by construction — `BrainWithPostHocClassifier` fits a linear probe on frozen Case-3 features post-hoc, giving both (a) a macro-F1 comparable to Case 1/2 and (b) a target-class logit for TCAV's directional derivative, letting Case 1's existing `concepts.py` machinery (`extract_pooled_features`, `train_cav`, `tcav_score`, `run_concept_analysis`) run on Case 3's self-supervised representation completely unchanged.

**Primary result** (5 epochs, 100 subjects): post-hoc test macro-F1 = **GRU 0.916, Transformer 0.917** — both within a point of Case 1 (0.920, fully supervised) and Case 2 (0.918, supervised contrastive), despite Case 3 never using a single class label during representation training. This is the standard "linear probe" evaluation convention for self-supervised representations, and it's a genuinely strong result for the project's throughline: a representation learned purely from same-window brain-HRF co-occurrence recovers almost all of the label-supervised representations' linearly-decodable task structure.

**Fixed 5-concept CAV/TCAV** (Transformer, the better architecture): probe accuracy 0.989–0.993 across all 5 concepts — even more separable than Case 1/2's originally-reported numbers. Every concept's own TCAV score against its target class saturates at exactly 1.0.

**A real methodological finding, caught before it was written up wrong**: applying the §6-of-`population-level-evaluation-plan.md` cross-class rank-bootstrap test here initially returned near-zero P(rank1) for several concepts against their *own* intended class — which looked like a null result but was a tie-breaking artifact. When several classes' TCAV scores all saturate at the 1.0 ceiling simultaneously (as they do here, given how separable Case 3's representation turned out to be), `max()`'s implicit lowest-index-wins tie-breaking silently biased which class "won" the rank test, independent of which class the direction actually implicated. Fixed by breaking ties uniformly at random (`cross_class_rank_bootstrap_test` in `src/neurolens/concepts.py`, now also reports `frac_ties_at_max`); after the fix, tied concepts correctly return P(rank1) ≈ 1/(number of classes tied at the ceiling) — e.g. ~0.50 for `hand` (a 2-way tie between `left_hand`/`right_hand`, both scoring 1.0), ~0.25 for `left_side` (a 4-way tie). **This is an honest limitation of the rank-bootstrap significance test, not of Case 3's representation**: the test loses discriminating power exactly in the high-separability regime it's being asked to validate, because its underlying TCAV score is binarized (fraction-positive-gradient) rather than continuous. A magnitude-based variant (mean signed directional derivative, not just its sign) would very likely recover discriminating power here — noted as follow-up work, not yet built.

## 3. A separate, deprioritized future direction: Bayesian dynamical systems (SLDS/rSLDS)

Not "Case 3" — see the disambiguation in §0. Unchanged in spirit from the original version of this plan, but no longer sequenced as "the next case": fit a recurrent switching linear dynamical system via **`lindermanlab/ssm`** on the raw ROI time series, and test whether its unsupervised discrete regimes correspond to the known movement conditions — a PGM-native, structurally interpretable alternative to Case 2's learned embedding space. If ever revisited, comparable to Case 2/Case 3(HRF) directly: does contrastive alignment and unsupervised dynamical regime-switching converge on similar task structure, discovered two independent ways?

## 4. Concept Activation Vector (CAV/TCAV) testing for Case 1 — **done**

Implemented in `src/neurolens/concepts.py` and [`09_concept_vectors.ipynb`](../notebooks/09_concept_vectors.ipynb). Scope: **label-derived concepts** (not literature-derived yet — that remains the harder open problem from [interpretability-methods-notes.md §4.1](interpretability-methods-notes.md#41-a-neurolens-rag-specific-variant-literature-derived-concept-hypotheses)). Five concepts tested on Case 1's Transformer, probed at the pooled 128-dim representation shared by both heads: `hand`, `foot`, `tongue` (each vs. everything else), and `right_side`/`left_side` (lateralized movements against each other).

**Result**: effector concepts scored exactly as expected (TCAV 1.0 for their own classes, 0.0 elsewhere, >0.99 probe accuracy). Laterality concepts surfaced a real, documented limitation — a CAV fit only on lateralized classes extrapolated a specific (not random) direction onto `baseline`/`tongue`, which were never part of its training set; see [interpretability-methods-notes.md §4.1](interpretability-methods-notes.md#41-a-neurolens-rag-specific-variant-literature-derived-concept-hypotheses) for the full write-up. This validates the TCAV mechanism against ground truth before the literature-derived version is attempted.

For each: gather positive/negative example windows from the training set by true label, extract pooled representations from the trained model, fit a linear probe (the CAV direction), then compute TCAV sensitivity scores against each class's logit for held-out test windows. This is real, verifiable science on this dataset (we know the true labels), and validates the CAV *mechanism* before ever attempting the much harder literature-derived version.

## 5. Data / paper corpus correction

The current 6-paper corpus over-indexes on infrastructure papers (Yeo 2011 parcellation, Van Essen 2013 HCP overview) that ground the *pipeline's methods* but aren't directly comparable *results*. What's actually useful for the RAG "compare our result to the literature" framing is **papers that decoded/classified the HCP MOTOR task themselves** — their reported accuracies and findings are the ones worth setting our 92.5% macro-F1 against. See a candidate list gathered separately.

## 6. Resume/portfolio framing (corrected)

Case 1, described accurately, is a **multi-task representation learning** result, not merely "a decoding model" — that's the vocabulary a PhD-level ML/AI resume should use, and it wasn't used the first time. Case 2, once built, becomes a genuinely strong complementary bullet: contrastive multimodal representation learning is squarely "modern AI lingo" (the same family as CLIP) and is a different, more advanced claim than a discriminative classifier. Case 3 (self-supervised brain-HRF co-embedding, §2.5) supports a third distinct claim: a representation learned with zero class-label supervision recovers nearly all of Case 1/2's linearly-decodable task structure — the standard self-supervised "linear probe" evaluation story, and evidence the throughline isn't an artifact of supervision.

## 7. Compute feasibility (M3, 16GB)

Unchanged from before — Case 2's two small encoders plus a contrastive loss are comparable in size/cost to Case 1's existing models; `ssm` (Case 3) is CPU-bound and lightweight. No new hardware concerns.

## 8. Case 2 v2 results — **done**

Trained and evaluated in [`10_contrastive_representation.ipynb`](../notebooks/10_contrastive_representation.ipynb). Full numbers: [`results/case2_contrastive_results.json`](../results/case2_contrastive_results.json).

**Primary result** (window=32, Transformer, matching Case 1's config for direct comparison): test macro-F1 = **0.907**, vs. Case 1's direct-classifier **0.925** — the contrastively-aligned representation trails the freely-learned classifier by a real but modest ~1.8 points. This is exactly the "aligned vs. task-optimized representation" gap predicted in §2.3: constraining the decision boundary to be a semantically-initialized text-embedding direction, rather than a fully free linear layer, costs a small amount of raw accuracy. Text-to-brain retrieval precision@20 was a perfect 1.00 for all 6 classes.

**Window size (16/24/32/48/64) × architecture (GRU/Transformer) sweep** — the standout finding: **the architecture gap is much larger and more consistent here than in Case 1.** Transformer beat GRU at every single window size (0.888–0.907 vs. 0.815–0.851, a ~5–9 point gap) — versus only a ~1–2 point Transformer-over-GRU gap on Case 1's direct classification task. So re-running the GRU-vs-Transformer comparison for a different objective *was* worth doing — the contrastive objective rewards the Transformer's architecture more than plain supervised classification did, plausibly because aligning to a fixed, semantically structured target space benefits more from self-attention's flexibility than from a single recurrent summary vector. Window size itself showed no strong monotonic trend for either architecture (peaks around 24–32 for Transformer, 48 for GRU); GRU degraded most at the longest window (64), consistent with recurrent architectures generally handling longer sequences less gracefully than self-attention.

**Text-embedding-model comparison** (at the best window/architecture config) — a genuine, worth-remembering negative result: the default `sentence-transformers/all-MiniLM-L6-v2` (384-dim) was the *best* of three (0.907), beating both a smaller model (`paraphrase-MiniLM-L3-v2`, 0.891) and a larger, generally higher-quality-on-benchmarks model (`all-mpnet-base-v2`, 768-dim, 0.888). Don't over-read this as "bigger text models are worse" — with only 6 fixed, simple condition descriptions repeated across the whole dataset, there's very little for a "better" general-purpose sentence embedding model to actually improve; general text-similarity benchmark quality isn't obviously the thing that matters when the text side is this small and this simple.

**Retrieval precision, revisited at multiple k**: precision@k was initially reported only at k=20 (1.00 for all 6 classes) — expanded to k ∈ {5, 10, 20, 50} and it's a perfect 1.00 at every k tested, against class sizes as small as 117. The choice of k didn't matter here, but this also means the true precision-degradation point is still unknown (beyond k=50).

## 8.5. Case 2 v3 results — forecasting horizon sweep — **done**

Built in [`11_case2_forecasting.ipynb`](../notebooks/11_case2_forecasting.ipynb): a ridge-regression linear probe on the *frozen* Case 2 backbone's pooled features, predicting future ROI activity at increasing horizons, using the leak-safe windowing from `data_setup.build_forecast_window_index()` (§2.4/§4 of this doc — a candidate window is dropped unless the forecast target's label matches the window's own current label, so a horizon can never silently cross into an adjacent condition block).

**Result: the representation carries real forward-predictive signal for roughly 3-4 seconds (4-6 TRs), then decays to noise.** Test R² = 0.296 at horizon=1 TR (0.7s), falling smoothly through 0.19 (h=2), 0.11 (h=3), 0.03 (h=4), crossing zero between horizon 4 and 6, then staying mildly negative through h=16 (11.5s). One point to discount rather than trust: horizon=20 (14.4s) shows a sharper drop (R²=-1.13), but only 196 training windows survive the leak-safety filter there (vs. 3,332 at horizon=1) — an order-of-magnitude sample-size confound, not a real acceleration of decay. Horizon=24 (17.3s) correctly returns **zero** leak-free windows, confirming the leak-safety filter works exactly as intended at the boundary — MOTOR condition blocks are only ~12s long.

**Answering the original question directly**: with this frozen representation and a linear probe, usable forecast horizon is about 3-4 seconds — short relative to a 12-second condition block. Not yet properly deconfounded from sample size (worth subsampling horizon=1's larger training set down to match smaller horizons' counts before treating the decay curve as a clean scientific claim), and not yet compared against a non-frozen or Case-3 (dynamical-systems) alternative, which is architecturally built for exactly this kind of forward prediction.

## 8.6. GRU vs. Transformer, paired significance test at 100-subject population scale — **done**

Built in [`13_architecture_comparison_bootstrap.ipynb`](../notebooks/13_architecture_comparison_bootstrap.ipynb): both architectures trained on the *same* random 65/13/12 split each repeat (30 repeats, same repeated-splits methodology as [`12_population_level_evaluation.ipynb`](../notebooks/12_population_level_evaluation.ipynb)), enabling a paired Wilcoxon signed-rank test on the per-repeat differences rather than comparing two independent confidence intervals by eye.

| | GRU | Transformer | Paired Wilcoxon p |
|---|---|---|---|
| **Case 1** (direct decoding) | 0.901 [0.872, 0.924] | **0.922 [0.902, 0.945]** | **3.7×10⁻⁹** |
| **Case 2** (contrastive) | 0.877 [0.835, 0.905] | **0.918 [0.881, 0.940]** | **1.9×10⁻⁹** |

**Transformer is the statistically significantly better architecture for both cases**, not just numerically higher on average — both p-values are far beyond any conventional significance threshold. The gap is real and specifically **larger for Case 2 than Case 1** (+4.1 points vs. +2.0 points, roughly double), confirming at full population scale what the 20-subject window/architecture sweep in §8 suggested: the contrastive objective rewards the Transformer's self-attention architecture more than plain supervised classification does. Transformer is the architecture used throughout the completed RAG-CAV loop (`14_rag_cav_loop.ipynb`) and the natural default for any future Case 2 v3/Case 3 work.

One methodological note worth being explicit about: an early version of this notebook crashed mid-run on a `matplotlib` API change (`boxplot(labels=...)` renamed to `tick_labels=` in matplotlib 3.9+) *after* the expensive Case 1 training had already completed but *before* results were saved to disk — losing that work and requiring a full rerun. Fixed by moving result-saving to happen immediately after each part's statistics are computed, before any plotting — a general lesson applied here: never let optional visualization code sit between expensive computation and persisting its results.

## 8.7 Case 2 CAV/TCAV and the RAG-CAV loop — **done**

The Case 1-only limitation noted in §4 no longer holds, though not via the route this section originally described. An early text-derived CAV direction (the difference between two condition prototypes, pulled back through the brain projection's transpose, `src/neurolens/concepts_case2.py`) reproduced Case 1's finding independently — `hand`/`foot`/`tongue` separate almost perfectly, laterality reads markedly weaker — but that weakness was later diagnosed as an artifact of the text-arithmetic derivation method itself, not a real representation gap: fitting a linear probe on Case 2's frozen features the same way as Case 1/3 gets laterality TCAV back up to 0.92–1.00, indistinguishable from the other two cases (see the interview-prep doc's `results/case2_fitted_probe_cav_sweep.json` finding). **The fitted-probe route is now the standard CAV/TCAV mechanism for Case 2** — see `docs/interview-prep-neurolens-rag.md`'s standardization note after §11.3 for the full reasoning.

The RAG-CAV verification loop (§4.1 of `interpretability-methods-notes.md`) is now built for Case 2 too, with a real fix for a measured failure: asked to freely judge agreement between literature and CAV evidence, the LLM defaulted to AGREE regardless of the actual TCAV score (10/12 real cases) — fixed by computing the verdict deterministically from (stance, TCAV) in code instead of asking the LLM to decide it (`notebooks/15_case2_cav_rag_loop.ipynb`, `docs/project-summary.md` §3.6).

## 9. Proposed build sequencing (updated)

1. ~~**CAV/TCAV for Case 1**~~ (§4) — **done**.
2. **Paper corpus correction** (§5) — real HCP MOTOR-decoding comparison papers *and* real motor-cognition/neuroscience hypothesis papers identified (Ehrsson et al. 2003, Meier et al. 2008, Wang et al. 2020, a GNN decoding paper); Ehrsson/Meier downloaded and indexed, Wang et al./GNN paper still pending (low priority, reprioritized away from).
3. ~~**Case 2 v1**~~: skipped directly to v2.
4. ~~**Case 2 v2**~~ — **done** (§8).
5. ~~**Case 2 v3 (stretch)**~~ — **done** (§8.5): linear-probe forecasting on the frozen representation, usable horizon ~3-4 seconds.
6. ~~**RAG↔CAV loop**~~ — **done** for both Case 1 and Case 2 (§8.7).
7. **Dataset scale-up, 100 → 200 subjects** — in progress as of this writing (`data/subject_discovery_v4.json`, `02_data_complete.ipynb`); motivated directly by §4's laterality-concept weakness, to test whether it's a data-scarcity artifact rather than a fundamental representational limit.
8. ~~**Case 3 (HRF co-embedding, §2.5)**~~ — **done** at 100-subject scale (§2.5.1): built, trained (GRU + Transformer), post-hoc-probed, and CAV/TCAV-wired. Queued for retraining on the 200-subject data once the scale-up completes, alongside Case 1/2, as part of the full 3-case sweep (item 9).
9. **3-case × 2-architecture × capacity-variant sweep, on 200-subject data** — sequenced after the data scale-up, per explicit decision. Retrain Case 1, Case 2, and Case 3 across GRU/Transformer with varied depth (layers) and attention heads — no new architecture code needed, both `GRUDecoder`/`TransformerDecoder` already take these as constructor args — then re-run the fixed 5-concept CAV/TCAV analysis (with bootstrap significance testing, §2.5.1) across every resulting model.
10. **Structural-connectivity (SC) graph input** — scoped as a separate future project (§4.3 of `project-summary.md`); subject-level DTI availability confirmed (97.7% of MOTOR-eligible candidates), but building an actual per-subject SC matrix and a graph-aware architecture is nontrivial additional work, not bundled into the above.
11. **Bayesian dynamical systems (SLDS/rSLDS)** (§3) — a separate, deprioritized direction, no longer "next after Case 2."

## References

- [case1-summary-report.md](case1-summary-report.md)
- [ml-design-report.md](ml-design-report.md)
- [interpretability-methods-notes.md](interpretability-methods-notes.md) §4.1
- [literature-notes-tokenization.md](literature-notes-tokenization.md) — POYO/POSSM background, the Eva Dyer throughline this design follows
- Radford et al. 2021, "Learning Transferable Visual Models From Natural Language Supervision" (CLIP) — the contrastive objective Case 2 (and, symmetrically, Case 3) follows
- [lindermanlab/ssm](https://github.com/lindermanlab/ssm) — rSLDS implementation, for §3's separate deprioritized direction (not Case 3)
