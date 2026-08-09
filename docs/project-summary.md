# NeuroLens-RAG: Multi-Model Representation Learning with Literature-Grounded Mechanistic Validation

> The single, top-level entry point for the project. Written in paper form (Introduction, Methods, Results, Discussion) rather than as a code walkthrough, so it reads as a scientific summary, not a change-log. Component docs go deeper on each piece: [case1-summary-report.md](case1-summary-report.md), [case2-3-design-plan.md](case2-3-design-plan.md), [interpretability-methods-notes.md](interpretability-methods-notes.md), [population-level-evaluation-plan.md](population-level-evaluation-plan.md), [decoded-state-to-text-report.md](decoded-state-to-text-report.md).

## Abstract

Decoding accuracy alone cannot distinguish a model that has learned real neurobiological structure from one that has learned a shortcut correlated with the label. This project decodes movement conditions from human fMRI (HCP MOTOR task, 100 subjects) using two complementary representation-learning objectives — supervised multi-task decoding and contrastive brain-language alignment — and then asks the harder question: *what did the model actually learn, and is that consistent with independent domain knowledge?* Four attribution methods and Concept Activation Vectors (CAVs) probe the model's internal representation directly; a retrieval-augmented generation (RAG) system grounds that probing in real neuroscience literature rather than a hand-picked concept list; and every headline claim is backed by population-level bootstrap statistics rather than a single training run. The result is a closed loop — literature generates falsifiable hypotheses about the model, and CAV testing checks them against the model's actual internals — that neither retrieval nor interpretability could produce alone.

## 1. Introduction

### 1.1 Motivation

A model that decodes hand-vs-foot movement from fMRI at 92% accuracy has *not* thereby demonstrated it represents "hand" and "foot" the way a neuroscientist means those terms — it has demonstrated that a linear or attention-weighted function of the input separates the classes well. These are different claims. The gap between them matters practically (does a decoder generalize to slightly different conditions?) and scientifically (can we trust the model's internal representation as a proxy for anything about brain organization?). Closing that gap requires evidence beyond accuracy: does the representation organize itself along axes a domain expert would recognize, and does that organization agree with what's already published about the domain?

### 1.2 Scope

The project has two decoding "Cases," a shared interpretability layer, and a literature-grounding layer:

- **Case 1** — supervised multi-task decoding: `X → y` (movement class) and `X → y_hrf` (auxiliary continuous hemodynamic-response regression), via GRU or Transformer encoders.
- **Case 2** — contrastive representation learning: a brain encoder and a text encoder (of the condition description) trained to align in a shared embedding space, CLIP-style, evaluated via prototype classification, cross-modal retrieval, and linear-probe forecasting.
- **Interpretability**: four independent attribution methods (gradient- and perturbation-based) plus Concept Activation Vectors (CAVs), which test whether the model's representation is organized along human-interpretable concept directions.
- **RAG-CAV loop**: real neuroscience literature is retrieved, an LLM extracts candidate concept claims from it, and those claims are tested directly against the model via CAVs — turning retrieval from a citation lookup into a mechanistic audit.
- **Statistical validation**: every comparative claim (architecture A vs. B, metric X's confidence interval) is backed by population-level resampling, modeled on a published methodological standard (Misra & Pessoa, 2025, *eLife*), not a single split.

Case 3 (a Bayesian dynamical-systems model, SLDS) is designed (§4) but not yet built.

## 2. Methods

### 2.1 Data

HCP Young Adult S1200, MOTOR task, LR+RL runs, 100 subjects (scaled from an initial 5 → 20 → 100 over the project, each subject verified against S3 before download). Schaefer-300 cortical atlas, 7-network (Yeo) parcellation. Preprocessing: motion regression, detrending, within-run z-scoring. Targets: a 6-class hard label (baseline + 5 movement conditions) and a 5-channel canonical-HRF-convolved continuous regressor built from the task's own event files. Subject-level splitting throughout (never window-level, to avoid leakage from overlapping windows): 10 subjects reserved for hyperparameter selection only, 65/13/12 train/val/test over the remaining 90.

### 2.2 Case 1 — multi-task decoding

32-volume causal windows (stride 2), `X[t-31:t] → y[t], y_hrf[t]`. GRU (hidden=128) or Transformer (d_model=128, 4 heads, 2 layers) encoder, pooled representation feeding a classification head and an *architecturally optional* HRF regression head (removed entirely, not just untrained, for classification-only ablations). Loss: `cross_entropy(y) + λ_hrf · mse(y_hrf)`, `λ_hrf=0.1`.

### 2.3 Case 2 — contrastive brain-language representation learning

A brain encoder (same GRU/Transformer backbones, heads removed) projects to a `d=64` embedding `z_brain`. A text encoder embeds each of the 6 condition descriptions once via a frozen pretrained sentence embedding model (MiniLM), with only a small trainable projection on top — appropriate given there are only 6 fixed text targets, not enough signal to learn a text encoder from scratch. Training: temperature-scaled cosine-similarity cross-entropy against all 6 known prototypes (a supervised-contrastive variant, not literal in-batch-negative InfoNCE, since the text side is a small closed vocabulary rather than CLIP's open per-example caption set — see §2.1 of `case2-3-design-plan.md` for the precise distinction). Evaluated via prototype-based classification (nearest text-prototype by cosine similarity), text-to-brain retrieval, and — via a frozen-backbone linear probe — forecasting of future ROI activity at increasing horizons, with a leak-safety constraint requiring the forecast target to share the source window's condition label (preventing a forecast from silently crossing into an adjacent condition block).

### 2.4 Interpretability

**Attribution** (which resting-state network drove a decode): Saliency and Integrated Gradients (gradient-based, via Captum) and exact Shapley values and LIME (perturbation-based), all pooled to the 7 Yeo networks — pooling to 7 "players" is what makes *exact* (non-sampled) Shapley-value computation tractable (128 coalitions enumerated exhaustively).

**Concept Activation Vectors (CAV/TCAV, Kim et al. 2018)**: given labeled positive/negative example inputs for a concept, fit a linear probe in the model's pooled representation to find the direction separating them (the CAV), then measure the directional derivative of a target class's logit along that direction for held-out examples — the fraction with a positive derivative is the TCAV score, i.e. "how often would nudging this representation toward the concept increase the model's confidence in this class." See §3 below for the worked example and diagram.

### 2.5 Retrieval-augmented generation and the RAG-CAV loop

A corpus of 8 papers — deliberately including genuine neuroscience hypothesis papers (Ehrsson et al. 2003 on body-part-specific motor imagery; Meier et al. 2008 on non-classical motor cortex organization), not only infrastructure papers — is chunked, embedded (MiniLM), and optionally reranked with a cross-encoder before being passed to a local, quantized LLM (Llama-3.2-3B via Apple's MLX framework, entirely on-device). The LLM performs three roles in sequence: (1) label each retrieved excerpt SUPPORTS/CONTRADICTS/UNRELATED to a decoded state, (2) extract a candidate concept phrase from a relevant excerpt, and (3) synthesize a final explanation integrating the decode, the literature, and an independent CAV test of whether the model's decision is actually sensitive to the extracted concept. Step 2→3 is the loop: a phrase is mapped (currently via keyword matching, §3.3 discusses the tradeoff) onto a concept the CAV machinery can test, closing the gap between "the literature says X" and "does the model's representation actually reflect X."

### 2.6 Statistical validation

Modeled on Misra & Pessoa (2025, *eLife*)'s methodology: subject-level (not data-point-level) resampling, an empirical stopping rule (add resamples in batches until the confidence interval stabilizes, rather than fixing a count in advance), and — for architecture comparisons — a *paired* design (both architectures trained on the identical resampled split each repeat) enabling a Wilcoxon signed-rank test on the per-repeat differences. One disclosed deviation: repeated random re-partitioning was used instead of literal with-replacement bootstrap, because the data-loading infrastructure de-duplicates by subject, making weighted resampling a no-op without further code changes (§population-level-evaluation-plan.md).

## 3. Results

### 3.1 Case 1

| Metric | Value |
|---|---|
| Single 100-subject split, Transformer | 91.3% test macro-F1 |
| Population-level (20 repeated splits) | **92.0%**, 95% CI [90.9%, 93.4%] |
| Post-hoc bootstrap (test-subject resampling, fixed model) | 92.6%, 95% CI [90.9%, 94.2%] — wider than the repeated-splits CI, explained below |

### 3.2 Case 2

| Metric | Value |
|---|---|
| Official-split, Transformer | 90.8% test macro-F1 |
| Post-hoc bootstrap (test-subject resampling, fixed model) | 90.8%, 95% CI [88.0%, 93.3%] |
| Text-to-brain retrieval, R-precision (= recall = F1 at k = class size) | **0.83–0.89** across the 6 classes |
| Forecasting horizon (frozen representation, linear probe) | Real signal ~3–4 seconds (4–6 TRs), decaying to noise beyond |
| Modality-gap check (cross-modal / within-brain distance ratio) | **1.01** — brain and text embeddings are exactly as close to each other as brain embeddings are to each other |
| Brain-only silhouette score (class labels, text ignored) | 0.43 — real class structure independent of the text anchors |

**Retrieval quality, corrected**: an earlier version of this report cited "precision = 1.00 at every k from 5–50" as the text-to-brain retrieval result. That number is real but misleading on its own — with ~470–580 true-class windows per class in the test set, retrieving only the top 5–50 necessarily caps *recall* at 1–10%, however good the ranking is; precision alone cannot reveal that. `text_to_brain_retrieval_metrics()` (`src/neurolens/contrastive.py`) now reports precision, recall, and F1 together at each k, and R-precision (precision evaluated at k = the true class size, the point where precision, recall, and F1 necessarily coincide) gives the single honest summary number: **0.83–0.89 across the 6 classes** — strong, consistent with the ~90% classification macro-F1, but not the artificially perfect 1.00 the precision-only view implied.

Retrieval precision alone also can't distinguish genuine cross-modal mixing from a "modality gap" (Liang et al., 2022) — two well-aligned-for-ranking but geometrically separate clouds. Three diagnostics were run to check directly: a cross-modal/within-modality distance ratio (1.01, i.e. no systematic offset), a brain-only silhouette score (0.43, i.e. real intrinsic class structure, not borrowed from the text prototypes), and a PCA visualization (`results/case2_embedding_space_pca.png`) showing each text prototype sitting inside its matching-class brain cluster rather than in a separate region. All three agree: the space is genuinely mixed, not just well-ranked. (One secondary check — a linear probe classifying brain-vs-text from the embedding alone — hit 100% accuracy, but this is a statistical artifact of comparing 6 text points against thousands of brain points with no possible held-out split, not evidence of a real gap; reported for completeness, not trusted on its own.)

### 3.3 Architecture comparison — GRU vs. Transformer, paired significance test

| | GRU | Transformer | Paired Wilcoxon p |
|---|---|---|---|
| Case 1 | 0.901 [0.872, 0.924] | **0.922 [0.902, 0.945]** | 3.7×10⁻⁹ |
| Case 2 | 0.877 [0.835, 0.905] | **0.918 [0.881, 0.940]** | 1.9×10⁻⁹ |

Transformer is statistically significantly better for both objectives, with roughly double the margin under the contrastive objective (+4.1 points) versus direct decoding (+2.0 points) — the contrastive setup rewards the Transformer's architecture more than plain supervised classification does.

**A methodologically informative discrepancy**: Case 1's post-hoc bootstrap CI (§3.1) is *wider* than its repeated-splits CI, despite using one fixed model rather than 20 independently retrained ones. This is not evidence the model is less reliable — with only 12 test subjects, with-replacement resampling from such a small discrete pool has inherently high sampling variance (a single resample can lose several subjects entirely or triple-count one), independent of model quality. The reference paper bootstrapped 92 participants; a much larger pool damps exactly this effect. The repeated-splits estimate, which draws from the full 90-subject pool each time, is the more trustworthy of the two at this subject count.

### 3.4 Interpretability

The four attribution methods split into two internally-consistent clusters (gradient-based: Saliency/IG, ~97% mutual top-1 agreement; perturbation-based: Shapley/LIME, ~97%) that agree with each other only ~78% of the time across families — gradient methods systematically over-weight the Default network relative to perturbation methods, which concentrate on SomMot (the neuroanatomically expected answer for MOTOR decoding). Unresolved.

CAV testing on label-derived concepts (`hand`, `foot`, `tongue`, `right_side`, `left_side`) shows near-perfect effector-concept separation for both architectures (probe accuracy 98.5–99.9%) — real evidence the representation organizes movement information along human-interpretable directions, not merely along whatever separates the raw classes. A genuinely informative artifact: a laterality-concept extrapolation result (baseline/tongue windows scoring high on a laterality direction they were never trained to represent) **flipped polarity** between the 20-subject and 100-subject runs — strong evidence this specific result is a linear-probe extrapolation artifact, not a reproducible neuroscience finding.

### 3.5 The RAG-CAV loop, demonstrated (Case 1)

A decoded `left_hand` window showed CAV sensitivity of 1.0 to both the `hand` and `left_side` concepts, converging independently with Ehrsson et al.'s contralateral-representation claim — literature and mechanistic testing agreeing without either informing the other is stronger evidence than either alone. A documented limitation: for a `left_foot` decode, the loop tested a `hand`-related concept extracted from a retrieved excerpt (because that's what the excerpt discussed, not because it was anatomically relevant) and got a correct null result (TCAV=0.0), which the LLM's synthesis then mischaracterized as "a discrepancy with the literature" — conflating "irrelevant to this trial" with "the model disagrees." A concrete, scoped bug, not a fundamental flaw in the approach.

### 3.6 The CAV-RAG loop, extended to Case 2

Case 2's shared brain-text embedding space permits a concept direction to be derived from text alone — the difference between two condition prototypes' embeddings, pulled back into the brain encoder's hidden space via the linear projection's transpose (`src/neurolens/concepts_case2.py`) — with no labeled brain examples needed at all, unlike Case 1's logistic-regression probe. Run against the same 5 concepts as Case 1, this independently-derived method reproduces Case 1's core finding: `hand`, `foot`, and `tongue` are separated almost perfectly (TCAV 0.98–1.0 for the matching decoded class), while `left_side`/`right_side` laterality is markedly weaker and noisier (0.26–0.74, no clean separation). Finding the same asymmetry via two unrelated derivation methods (a supervised probe on brain data, and pure text arithmetic) is stronger evidence it's a real property of the learned representation than either method alone could give.

**Faithfulness, measured, not assumed**: the original motivation for extending the loop to Case 2 was to check whether the LLM's generated synthesis text is actually faithful to the CAV evidence it's given, not just fluent. The final-synthesis prompt was extended to require a structured `VERDICT: AGREE/DISAGREE/UNCLEAR` tag, checked against a verdict computed independently from the real TCAV score. Run over 12 real decoded test windows: the LLM said **AGREE in 10 of 12 cases**, DISAGREE once, and failed to produce a parseable tag once — regardless of whether the tested concept's TCAV score was 0.99 or 0.07. Scoring is reported as a range rather than one number, because most decodes surface *multiple* candidate concepts and it's genuinely ambiguous which one a one-sentence verdict is "about": under a strict rule (must match the first-extracted concept) faithfulness is 25% (3/12); under a lenient rule (must match *any* tested concept) it's 58% (7/12). The number that doesn't depend on that ambiguity is the **AGREE-default bias itself** — inspecting the actual generated text shows the LLM's prose often echoes whichever concept happened to have the *highest* TCAV score among several tested, rather than the specific one its verdict claims to summarize, which is the same "irrelevant vs. disagrees" conflation from §3.5, now confirmed as a systematic pattern at n=12, not a single anecdote. Full outputs: `results/case2_rag_cav_loop_examples.json`, `results/case2_faithfulness_scorecard.json`.

### 3.7 Three RAG-LLM improvements

Three specific, scoped improvements to the RAG/LLM block were built and measured — not just proposed:

**1. CAV-aware query refinement (no training).** After the CAV loop identifies which literature-derived concept the model is *most* sensitive to (highest TCAV), a second, concept-steered retrieval query is issued and re-synthesized from those results, rather than only ever searching on the decoded label. Demonstrated on 4 real Case 1 examples: the top retrieved excerpt didn't change in any of the 4 (the 8-paper corpus is small enough that the single most relevant excerpt already dominates any related query — the same ceiling effect seen in the original hand-labeled retrieval eval), but the cross-encoder's relevance *score* for that excerpt improved in all 4, suggesting the technique would matter more with a larger, more diverse corpus. `results/case1_query_refinement_examples.json`.

**2. LoRA fine-tuning on the documented failure mode.** A 56-example synthetic dataset was built where each (prompt, gold-completion) pair explicitly teaches the AGREE/DISAGREE/UNCLEAR discrimination rule the base model gets wrong (§3.6). LoRA fine-tuning (`mlx_lm.lora`, 150 iterations, 8 layers) on this small set drove training loss from 2.8 to 0.03 — and fixed a real, separate problem outright: the base model, even when explicitly instructed to end every response with a `VERDICT:` tag, never actually did so (0/8 held-out prompts); the fine-tuned model did so reliably (8/8). But it did **not** learn the underlying discrimination logic: on those same held-out prompts it collapsed to predicting the single majority class (`UNCLEAR`, ~56% of the tiny training set) every time, and on real out-of-distribution prompts from the actual pipeline it was inconsistent — matching the original behavior in 2/4 cases, plausibly more honest in one (`left_hand`: shifted from confident AGREE to UNCLEAR given genuinely mixed concept evidence), and worse in one (`right_foot`: shifted from a defensible DISAGREE to AGREE). Honest reading: LoRA reliably fixes *format compliance* at this scale but needs a larger, more varied training set — not one rigid template — before it can be trusted to fix the *reasoning* bug. `results/case1_lora_before_after.json`.

**3. Domain-adaptive retrieval embedding fine-tuning.** The existing paper-level retrieval eval was already saturated (precision@5 = 1.00), leaving no room to show whether domain adaptation helps — so a harder, chunk-level benchmark was built instead: an LLM paraphrases one fact from a sampled chunk into a natural query, and the task is retrieving that *exact* source chunk out of the full ~880-chunk corpus (adjacent chunks overlap by 50 words, making this a genuine near-duplicate-disambiguation problem, not a saturated one). Off-the-shelf MiniLM: **45.5% top-1 / 69.1% top-3** accuracy on 55 held-out queries. After contrastively fine-tuning MiniLM on 112 in-domain (query, chunk) pairs (`MultipleNegativesRankingLoss`, 4 epochs, ~16 seconds of training): **56.4% top-1 / 74.5% top-3** — a genuine, non-trivial improvement (+11 / +5.4 points) from a small, cheap, in-domain fine-tune. Of the three ideas, this is the one with an unambiguous positive result. `results/case1_embedding_finetune_results.json`, fine-tuned model at `models/minilm_domain_finetuned`.

## 4. Discussion

### 4.1 Why the RAG/LLM step is not an appendage

Accuracy and CAV testing both answer questions a human already thought to ask (is the model right; does it use *this* pre-specified concept). Retrieval-augmented generation over real literature supplies concepts a human didn't have to think of — the papers already contain domain experts' claims about what should be represented and how. Routing those claims through an LLM to extract testable phrases, then checking them against the model with CAVs, turns the literature into an external, falsifiable audit of the model rather than a decorative citation list appended to a finished result. The convergent-validity result in §3.5 is the concrete evidence this works; the documented failure mode in the same section is evidence it isn't finished.

### 4.2 Limitations, stated plainly

- The gradient-vs-perturbation interpretability disagreement (§3.4) is unresolved.
- The RAG-CAV synthesis step conflates "concept irrelevant to this trial" with "model contradicts literature" — now *measured* at scale (§3.6: AGREE in 10/12 real cases regardless of the actual TCAV score), not just documented as a single anecdote. Not yet fixed; the LoRA attempt at fixing it (§3.7.2) improved format compliance but not the underlying reasoning at this data scale.
- The literature corpus (8 papers) is small; retrieval and generation quality both plausibly improve with a larger, more targeted corpus — consistent with the query-refinement experiment (§3.7.1) hitting a ceiling for exactly this reason.
- The phrase-to-concept mapping in the RAG-CAV loop is keyword-based, not embedding-based — a known simplification, discussed further in `interpretability-methods-notes.md`. Case 2's concept-direction method (§3.6) sidesteps this for the *direction* itself, but literature phrase→concept-name matching is still keyword-based in both cases.
- Case 2's forecasting used a frozen linear probe only; a fine-tuned or Case-3-style dynamical model might extend the usable horizon.
- The RAG-CAV loop's automated faithfulness check (§3.6) has its own scoring ambiguity when multiple concepts are tested per decode (which is common) — reported as a strict/lenient range rather than one number, since it's genuinely unclear which tested concept a one-sentence LLM verdict is "about."
- Case 3 (Bayesian SLDS/rSLDS) is designed, not implemented.

### 4.3 Future directions

**A three-modality extension of Case 2** (brain, text, and continuous HRF as a third contrastively-aligned view) is under active discussion — motivated by HRF's smooth, graded temporal structure being potentially more forecast-relevant than a discrete label, but requiring care: `y_hrf` is derived from full-run event convolution, so using it as a *forecast input* (rather than a same-window auxiliary target, as Case 1 already does safely) would leak future event timing unless restricted to the already-elapsed portion of the current window.

**Knowledge distillation as a Case 1 extension.** A natural next step for the "applied/efficient deployment" angle: train a much smaller student encoder (e.g. `d_model=32`, single layer — an order of magnitude fewer parameters than the 128-dim, 2-layer Transformer teacher) using Hinton et al. (2015)-style distillation, `L = α·CE(y_student, y_true) + (1-α)·T²·KL(softmax(y_teacher/T) ‖ softmax(y_student/T))`, with the already-trained Case 1 Transformer as a frozen teacher. The methodologically important comparison isn't student-vs-teacher (a small model is expected to underperform) but the **same small student trained two ways** — hard-label-only vs. with distillation — to isolate whatever the teacher's soft labels actually contribute from what the small architecture could reach on its own; reported with the same repeated-splits paired-Wilcoxon treatment already used for the GRU-vs-Transformer comparison (§3.3), plus a parameter-count/inference-latency comparison as the compression story. Fully buildable on existing infrastructure (`model_builder.py`'s architectures are already parametrized by width/depth; `engine.py` needs one new loss variant) — scoped but not yet built.

## 5. Resume-ready summary

See [case1-summary-report.md §11](case1-summary-report.md#11-resume-ready-bullet-points-google-xyz-format-accomplished-x-measured-by-y-by-doing-z) for XYZ/STAR-format bullets. The whole-project framing: this combines supervised and contrastive representation learning, four-method mechanistic interpretability, population-level statistical inference matching a peer-reviewed methodological standard, and a novel retrieval-interpretability integration (the RAG-CAV loop) — validated with real data, including honestly-reported negative results and a documented failure mode, not just successes.
