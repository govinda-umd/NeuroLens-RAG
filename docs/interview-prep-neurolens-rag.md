# NeuroLens-RAG — Interview Deep-Dive Prep

Living rehearsal document for the Amazon Applied Scientist (ML, Transportation) interview — deep dive on NeuroLens-RAG, plus likely extension questions. Built block by block; each section is (a) the thing to be able to say cleanly, (b) the follow-up questions already stress-tested, and (c) where the honest limits are, because "I don't know that yet, here's the plan" is a better answer than overclaiming.

Separate, still to prep: the Bayesian generative modeling thesis project (`Desktop/THESIS/GovindaThesis.pdf`) — not covered here.

---

## 1. The one-sentence purpose

Accuracy alone can't tell you whether a decoder learned something neurobiologically real or an incidental shortcut correlated with the label — NeuroLens-RAG is a framework for *testing* that distinction, not just reporting accuracy.

---

## 2. The data (say this before anything else — it grounds every design choice that follows)

- HCP Young Adult, MOTOR task, 100 subjects (200 as of the latest scale-up, not yet re-validated).
- `X`: a 300-channel **multivariate** time series (300 Schaefer-parcellated ROIs of the same BOLD signal — not "multisource," that word invites "which sources?"). Motion-regressed, detrended, z-scored per run.
- Windowed causally: 32 TRs, stride 2 (30-TR overlap between consecutive windows). HCP's fast TR is 0.72s, so 32 TRs ≈ 23 seconds — long enough to span a full ~12-second condition block plus margin.
- Per window, two targets, both defined at the window's **last** timepoint (the causal framing: "given the past ~23s, what's happening right now"), not a majority vote over the window even though a window can span a class boundary:
  - `y`: one of 6 categorical conditions (baseline, left/right hand, left/right foot, tongue).
  - `y_hrf`: a **5-channel continuous vector** (one channel per non-baseline condition), each channel = that condition's event timeline convolved with the canonical hemodynamic response function (standard GLM design-matrix regressor). Near-zero except during/just after that condition's block, because the HRF smears the signal forward in time. Baseline gets no channel — it's the implicit reference condition. **Important**: this is a single 5-dim vector at one timepoint, not a per-window time series — matters directly for Case 3's encoder design (a small MLP, not a sequence model).
  - Conditions occur in contiguous blocks (a subject performs one motor task at a time), so a single timepoint has one class, but a 32-TR window can straddle two blocks.
- Splits are subject-level, never window-level — overlapping windows from one subject are highly correlated, so a window-level split would leak near-duplicates across train/test.

---

## 3. Why three representation-learning paradigms (the core narrative)

Given the same three quantities per window — `X`, `y`, `y_hrf` — there are multiple structurally different claims about what a "good representation" should be faithful to, depending on which pair you relate and how:

1. **Relate `X` to `y` directly, as prediction.** Case 1: supervised multi-task decoding, joint classification + regression loss (`y` and `y_hrf` both used as direct prediction targets, off the same pooled features).
2. **Relate `X` to `y`, but through language, as alignment.** Case 2: embed the class's text description (frozen MiniLM + trainable projection), pull the brain window toward it in a shared space. Strictly richer claim than Case 1 — says the representation should respect the *semantic relationships between classes* (left/right hand share structure a bare class index can't express), not just which class. Trained supervised-contrastive (SupCon-style — true label picks the positive prototype), **not** self-supervised, **not** CLIP-style (CLIP's positives are naturally-paired, not label-selected).
3. **Relate `X` to `y_hrf`, as alignment, drop `y` entirely.** Case 3: self-supervised — no label anywhere in the loss. Claim: the representation should organize around concurrent physiology, and if it does, task structure should be recoverable afterward via a linear probe, not because it was trained to predict it.

Nice detail to have ready: `y_hrf` plays **two different roles** across the project — direct MSE regression target in Case 1, contrastive alignment target in Case 3. Same signal, two different claims about what "using" it means.

One-breath version: *given the same window, there are three different things worth being faithful to — the label, the label's meaning, and the concurrent physiology — one paradigm per choice, architecture held constant, to see which one(s) produce an interpretable representation.*

**Precision correction (caught during prep, don't say the old version):** Case 2 and Case 3 are **both sequence-to-vector alignment**, not "seq-to-scalar" (Case 2) vs "seq-to-seq" (Case 3). `y_hrf` is a single vector per window, same shape-class as a text embedding — no sequence-level target exists in the current build. The real distinguishing axes between Case 2 and Case 3 are (a) what the alignment target is, and (b) loss symmetry: Case 2 is asymmetric (brain→text only, against 6 fixed prototypes); Case 3 is symmetric in-batch-negative InfoNCE, possible *because* `y_hrf` is continuous and per-example rather than drawn from a small closed set.

---

## 4. Architecture choices (backbones)

Both paradigms compare the same two encoders — architecture held fixed, only the objective varies, so any representational difference is attributable to the objective, not a model-swap confound.

- **GRU** (hidden=128, 1 layer default): takes the raw 300-dim input directly, no projection needed. Pooled rep = final layer's final hidden state, `h_n[-1]`, `[B,128]`. Hypothesis under test: the final hidden state is a sufficient statistic for everything the recurrence has seen — causal by construction, at every layer, for the whole depth of the network.
- **Transformer** (d_model=128, 4 heads, 2 layers, ff=256, dropout=0.1): needs an explicit `Linear(300→128)` input projection (attention needs a fixed model dim) and explicit sinusoidal positional encoding (self-attention has no inherent order sense — the GRU gets order for free from recurrence). Pooled rep = final temporal token, `h[:,-1,:]`, chosen to mirror the GRU's pooling convention for a fair comparison (flagged as a first-pass choice — mean-pooling or a CLS/ViT-style token are open alternatives).
- **Unifying design choice**: every backbone exposes the same `forward_features(x) -> [B, hidden_dim]` interface — this is what let the CAV/TCAV code run unmodified across all three paradigms, including Case 3's post-hoc-probed model. Not an accident; it's why adding a third paradigm was cheap.

### Stress-tested follow-ups (verified against the actual code, not recalled from memory)

**Q: Does the Transformer use a causal attention mask?**
No — checked `model_builder.py` directly, `TransformerEncoder(h)` is called with no `mask` argument. Full bidirectional attention within each window.
- This does **not** leak information relative to the prediction task — "causal" here describes the *windowing scheme* (window only contains timepoints up to the target), not the internal attention pattern. Every timepoint in an already-fully-observed window is legitimately available.
- It does mean the OpenAI/GPT analogy doesn't quite hold — GPT's last-token readout is causal *because of* a causal mask (needed for autoregressive training). Ours has no such mask, so it's structurally closer to a BERT-style encoder reading out the last slot than to a GPT-style decoder.
- Real asymmetry with the GRU: the GRU's final state is causal at every layer, for the whole network depth; the Transformer's final token, by layer 2, has indirectly absorbed information mixed in from every other position at layer 1. So the comparison quietly bundles "recurrence vs. attention" together with "causal vs. bidirectional receptive field" — two different axes.
- **Now fixed**: `TransformerDecoder` takes a `causal: bool` constructor arg. When `True`, a lower-triangular mask (`nn.Transformer.generate_square_subsequent_mask`) is applied at every layer. Verified empirically, not just by reading the mask code: perturbed a future timestep (position 20) and measured the change at an earlier position's representation (position 10) — under `causal=False` the max change was 1.32 (real leakage, as expected for full attention); under `causal=True` it was exactly 0.0 (no leakage). Trains end-to-end without errors. Parameter count is unchanged by masking (304,907 either way) — this fixes the causal-vs-bidirectional confound specifically, it does **not** fix the 1.83× parameter-count gap with the GRU (separate issue, above). Both variants are queued into the capacity sweep alongside the GRU depth / Transformer width sweep, so the eventual comparison can isolate all three axes (recurrence vs. attention, causal vs. bidirectional, and raw capacity) rather than reporting one number that bundles them.

**Q: Is the GRU-vs-Transformer comparison fair — different layer counts, are parameters matched?**
No, and this is a real, currently open gap, not yet controlled for — checked exact parameter counts via `count_parameters()`:

| | GRU (1 layer) | Transformer (2 layers) |
|---|---|---|
| Total params | 166,539 | 304,907 |

**1.83× more parameters in the Transformer**, at the default config used for every result in this project. Honest statement: the Transformer-beats-GRU finding could be an architecture-family effect, a raw-capacity effect, or both — current experiments can't distinguish them. This is exactly why the model-capacity sweep (deeper GRU / narrower or param-matched Transformer) is already queued, sequenced after the 200-subject retraining.

---

## 5. Discriminative vs. generative vs. self-supervised, TCAV, and BNNs

**Q: Are the three cases discriminative or generative? Is the rSLDS idea generative?**
- Case 1: discriminative, textbook — directly parameterizes $p(y\mid x)$.
- Case 2: discriminative by loss mechanics — `cross_entropy(τ z_brain z_text^T, y)` *is* a softmax classification loss, just with semantically-initialized "weights" (text embeddings) instead of freely-learned ones. Uses `y` as supervision either way.
- Case 3: neither, precisely — never sees `y` (not discriminative w.r.t. the task), doesn't model a data likelihood over `X` either (not generative). Self-supervised contrastive representation learning is a genuine third category. Subtlety worth having ready: the InfoNCE loss *itself* is mechanically a discriminative classification task ("identify the true pair among distractors") — so Case 3 is self-supervised w.r.t. the task label, internally discriminative w.r.t. its own pretext task.
- rSLDS (deprioritized, not "Case 3"): generative, but precisely — an **unsupervised** generative model of `X` alone (no reference to `y` during fitting), whose discrete regimes are compared to `y` only after the fact. Different from a supervised generative *classifier* (Naive Bayes/GDA-style, explicitly modeling $p(X\mid y)p(y)$ for Bayes-rule classification) — don't conflate the two.

**Q: Does TCAV work on generative models? On BNNs? Are BNNs discriminative?**
TCAV needs exactly one thing: a differentiable, class-conditional scalar to take a directional derivative of. It's not restricted to discriminative models by definition — it's restricted to anything exposing that scalar. Generative/self-supervised models without one (Case 3, or a generative model like rSLDS) need a bolted-on linear probe first — the *same* `BrainWithPostHocClassifier` trick used for Case 3 would work for rSLDS too: fit a linear readout on the discrete-regime posterior probabilities against known labels, then run TCAV through that.

"Bayesian" and "discriminative/generative" are orthogonal axes, not the same question:

| | Point-estimate | Bayesian (weight posterior) |
|---|---|---|
| **Discriminative** ($p(y\mid x)$) | standard NN classifier | standard "BNN" — same $p(y\mid x)$, marginalized/sampled over weight posterior |
| **Generative** ($p(x\mid y)p(y)$ or unsupervised $p(x)$) | e.g. SLDS fit by MLE | thesis work — nonparametric Bayesian SBMs, MCMC |

A "BNN" as normally used is discriminative — Bayesian describes uncertainty over the *weights*, not whether the output is $p(y\mid x)$ or $p(x\mid y)p(y)$. TCAV works on a BNN exactly like any discriminative model (directional derivative at the posterior mean, or per weight-sample). Novel extension worth having ready: propagate weight-posterior uncertainty into the TCAV score itself — a credible interval from *model* uncertainty, layered on top of the bootstrap-resampling uncertainty this project already computes from *data* uncertainty. Draws directly on the thesis side, not a generic answer.

**Asset to state explicitly if it comes up**: real built experience on both halves of that 2×2 table — discriminative representation learning here, Bayesian generative modeling in the thesis. Most candidates only have one.

---

## 6. Evaluation / interpretability block

Two layers, answering genuinely different questions — keep this distinction sharp:

### 6.1 Attribution — "which input regions drove this decision?"

Four methods against the 7 Yeo resting-state networks (Visual, Somatomotor, Dorsal Attention, Salience/Ventral Attention, Limbic, Control, Default), for one decoded window:
- **Gradient-based** (Saliency, Integrated Gradients): on the continuous 300-ROI input, aggregated to 7 networks by summing. Saliency = `|∂output/∂input|` — fast, but noisy/gradient-saturation-prone. IG integrates the gradient along a path from a zero baseline to the real input, guaranteeing *completeness* (attributions sum exactly to the output difference between baseline and real input) — a real advantage Saliency lacks.
- **Perturbation-based** (exact Shapley, LIME): on a coarser 7-"player" abstraction (whole networks ablated, not individual ROIs). Not a convenience shortcut — it's specifically what makes **exact** Shapley possible: $2^7=128$ coalitions is exhaustively enumerable; $2^{300}$ is not. This is the design reason for choosing the network-level abstraction over ROI-level for these two methods.
- **Known, unresolved finding**: the two families disagree with each other more than within a family. Reported honestly as open, not smoothed over — matches a well-known reproducibility concern in the broader attribution literature (gradient- vs. perturbation-based methods often answer subtly different questions about "importance"), not evidence of a bug specific to this implementation.

### 6.2 CAV/TCAV — "does the model's decision depend on a concept a human named?"

Fundamentally different from attribution: attribution tells you *where*; CAV/TCAV tells you *whether a named idea* mattered — a question attribution can't even ask, since it never requires defining a concept.

Mechanism:
1. Define a concept via positive/negative examples (Case 1/3: labeled brain windows; Case 2: two text-prototype embeddings).
2. Fit a **linear probe** (logistic regression) separating them in the pooled representation. Normalized weight vector = the Concept Activation Vector. Probe accuracy reported as a diagnostic — low accuracy means the concept isn't linearly present, don't trust the direction.
3. TCAV score: gradient of the target class's logit w.r.t. the pooled representation, dotted with the CAV direction (directional derivative) — **fraction of held-out examples where that dot product is positive.**

**Is this "causal"?** No — it's a local sensitivity measure (linearized: "if nudged infinitesimally along this direction, would the logit increase"), not a full interventional experiment on the real system. It's a genuine local "what-if" *within the model's own function*, stronger than correlation, but not the same epistemic strength as an actual intervention. Have this distinction ready if pushed on the word "causal."

**Connects directly to the Case 3 ceiling finding (§5)**: TCAV's score is *binarized* (fraction positive, not mean magnitude) — that's exactly why it saturates once a representation gets separable enough that almost every example is on the positive side. Not a bug specific to this project; a property of the original Kim et al. definition, surfaced by testing it against a more separable representation than it was designed against.

**Three CAV-derivation routes, one per case, and why:**
- Case 1: labeled-example logistic regression on brain features directly — no shortcut, no shared embedding space to exploit.
- Case 2: **no labeled brain examples** — subtract two text-prototype embeddings in the shared space, pull the direction back through the transpose of the trained linear projection (`brain_projection`). Works *specifically* because that projection is linear — pulling a direction back through a linear map's transpose is the adjoint of that map, a standard trick that wouldn't work for an arbitrary nonlinear projection.
- Case 3: same labeled-example route as Case 1. It has an HRF encoder, not a text encoder — "hand" or "tongue" isn't naturally expressible as a difference between two HRF vectors the way it is between two text embeddings. The brain-example-free trick is a consequence of having a *language* modality specifically, not any second modality in general.

---

## 6.3 Follow-up: is the effector/laterality asymmetry really "low-level vs. high-level"?

Sharper reframe: the 6 class labels are the *conjunction* of two orthogonal factors (effector: hand/foot/tongue × side: left/right) — "left hand" = effector:hand ∧ side:left. Effector and laterality are two marginal factors of the *same* underlying design, not a low/high hierarchy — both are equally present in every training label. The finding is that the representation cleanly encodes one marginal and not the other, despite the model needing both to predict the joint class.

**Would more data fix laterality?** This is exactly why the 100→200 scale-up is sequenced first — cheapest hypothesis to rule out (statistical-power problem: signal exists but is hard to estimate reliably from few examples). Competing, more structural hypothesis it doesn't rule out: laterality may be genuinely more diffuse at Schaefer-300 ROI-level pooling than effector identity (strong somatotopic organization → large, spatially concentrated signal), in which case more data just gives a more confident estimate of the same weak effect. If the scale-up doesn't move the needle, structural connectivity (already on the roadmap) is the targeted next move — SC is inherently hemisphere-specific in a way pooled functional time series may not be, so it's a specific candidate fix for *this* finding, not just "add anatomy" in the abstract.

**Would more capacity fix laterality?** Argument against: if the model were capacity-bound, both factors should degrade somewhat, drawing on the same limited representational budget. What's observed is one factor at the ceiling (~1.0) and the other weak, using the *same* model — selective degradation of one factor while the other maxes out is more consistent with "this factor's signal is genuinely weaker in the input" than "the model ran out of room." If capacity were the bottleneck, effector wouldn't be at ceiling either.

**General low-level/high-level concept hierarchy (broader literature, not project-specific)**: real and foundational —
- Zeiler & Fergus (2014), *Visualizing and Understanding Convolutional Networks* — canonical depth-indexed hierarchy (early layers = low-level edges/textures, later layers = high-level semantic concepts).
- Bengio, Courville & Vincent (2013), *Representation Learning: A Review and New Perspectives* — the general theory claim that depth captures more abstract factors of variation.
- Olah et al., Distill: *Feature Visualization* (2017), *Zoom In: An Introduction to Circuits* (2020) — the mechanistic-interpretability lineage TCAV sits in directly.
- Tenney et al. (2019), *BERT Rediscovers the Classical NLP Pipeline* — same hierarchy in NLP (early layers = syntax, later = semantics/discourse).

**Honest gap this reveals in NeuroLens-RAG**: that hierarchy is normally explored via multi-depth probing. This project probes concepts at exactly one fixed point (final pooled representation, pre-classifier-head) — never asks whether laterality is more cleanly represented at an earlier GRU timestep or Transformer layer. Real, cheap extension if asked "what next": layer-wise / timestep-wise CAV probing, not just final-representation probing.

## 6.4 Follow-up: the N ≫ P concern (params vs. training examples)

Real, verified numbers, not hand-waved: **16,510 training windows** at 100-subject scale (observed directly from notebook 17's dataloader build) against **166,539 (GRU) / 304,907 (Transformer)** parameters — parameters exceed training examples by **10–18×**, inverted from the classical heuristic. ~33K windows expected at 200 subjects (not yet verified — train/val/test subject-split lists haven't been re-run against the new pool). Overlapping windows (30/32 TR overlap) make even the raw window count an overstatement of true independent information — sharpens, doesn't resolve, the concern.

**Why this doesn't necessarily mean the models are broken:**
- Weight sharing (GRU: identical transition weights reused across all 32 timesteps and every window) collapses *effective* degrees of freedom well below raw parameter count — classical parameter-counting doesn't capture this.
- Explicit regularization already in use: dropout (0.1, Transformer), weight decay (AdamW), early stopping via validation loss.
- **The strongest answer**: don't reason from the heuristic a priori, measure whether it's happening. Repeated-split bootstrap CIs (20 resamples, subject-level) are the direct empirical check — harmful overparameterization would show up as unstable/wide CIs or a train/test gap. Not observed (Case 1: [90.9, 93.4], tight and stable).

**References**:
- Classical heuristic's origin: Vapnik, *Statistical Learning Theory* (1998); Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (2009).
- **The single most important paper to know**: Zhang, Bengio, Hardt, Recht & Vinyals, *Understanding Deep Learning Requires Rethinking Generalization* (ICLR 2017) — neural nets can perfectly memorize random labels yet generalize well on real data trained normally.
- Double descent: Belkin, Hsu, Ma & Mandal, *Reconciling Modern Machine Learning Practice and the Bias-Variance Trade-off* (PNAS 2019); Nakkiran et al., *Deep Double Descent* (2019/2021).
- Deeper/theoretical: Bartlett, Long, Lugosi & Tsigler, *Benign Overfitting in Linear Regression* (PNAS 2020).
- Honest framing if pressed: this is a genuinely open area of theoretical ML research, not solved — "classical theory doesn't explain this, here's the empirical literature on why it doesn't catastrophically fail anyway," not "here's the theorem that says it's fine."

**Weight decay**: L2 penalty on parameter magnitude (decoupled form in this project's optimizer, `AdamW`) — Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (ICLR 2019): naive L2-in-the-loss isn't equivalent to true weight decay under Adam's adaptive per-parameter learning rates, hence the explicit fix. Concretely relevant, not just textbook trivia — it's the actual optimizer in the code.

**Weight sharing**: same parameters reused across multiple applications within a model (CNN filter at every spatial location; GRU transition weights at every timestep) — the concrete mechanism behind the effective-degrees-of-freedom argument above.

**Murphy's books (*Probabilistic Machine Learning*, Intro + Advanced)**: expect both weight decay (general regularization material) and weight sharing (CNN chapter, RNN chapter for the time-step version) to be covered — comprehensive enough texts that both are near-certain to be there, but no exact section numbers confirmed here; check the index rather than trust a specific page citation.

## 6.5 Follow-up: foundation models for fMRI

Active research direction: self-supervised pretraining (masked-timepoint/masked-patch reconstruction, MAE/BERT-style) on large aggregated resting-state fMRI corpora (UK Biobank scale, sometimes HCP), then fine-tuned for a downstream task. **BrainLM** is one concrete example recalled with reasonable confidence — flag this area as fast-moving enough to warrant a fresh literature check before treating any name as current best-in-class, rather than relying on recall.

**Real practical constraint, not an afterthought**: most such models are pretrained on a specific parcellation or voxel/surface resolution. Before fine-tuning, need to check whether the foundation model's expected input format is compatible with — or adaptable to — this project's Schaefer-300, HCP-preprocessed ROI time series. Not plug-and-play; input-format matching is a real first engineering question.

---

## 6.6 Extension ideas surfaced during prep (not yet built — candidates for `case2-3-design-plan.md` once interview prep concludes)

**Why mechanistic interpretability is harder for fMRI than for images/text.** For images/text, ground-truth concepts are close to free — a captioned "zebra" photo really contains a zebra, the generative process already organized itself along human categories. fMRI has no equivalent: there's no independently observable ground truth for what a voxel pattern "means," the signal-to-concept mapping is itself the open scientific question. Doshi-Velez & Kim, *Towards a Rigorous Science of Interpretable Machine Learning* (2017), motivate exactly this kind of fallback-to-proxy evaluation when ground truth isn't directly checkable. **This is the deeper justification for the RAG literature-grounding step** — task labels are the closest thing to ground truth fMRI research offers, and RAG supplies an independent, externally-sourced cross-check against domain literature precisely because there's no "just look at the image" option here.

**Temporal-axis probing (reframing "layer-wise probing" for this project's actual architecture).** With 1–2 layers, depth-wise probing has almost no resolution — but there's a temporal axis instead. GRU's `nn.GRU` exposes a full per-timestep hidden-state sequence (`output`), not just the final `h_n` currently returned by `forward_features`; Transformer likewise has per-position representations before final pooling. Probing CAV/TCAV at every timestep (not just the window's end) would answer a genuinely new question: *when* does a concept become linearly decodable — right at the window's end, or does it build up earlier? Cheap to build (optional full-sequence return, not a structural rewrite).

**Attention-head inspection, with the right caveat attached.** Feasible (`nn.MultiheadAttention` can return per-head weights). Real caveat: Jain & Wallace, *Attention is not Explanation* (2019) — attention weights often don't track gradient-based importance, and very different attention patterns can produce near-identical outputs; Wiegreffe & Pinter, *Attention is not not Explanation* (2019), complicates this further — an unsettled, live debate. Right use: attention inspection as **hypothesis-generating** ("this head attends heavily around t=15"), CAV/TCAV (already built, more rigorous) as the **validating** step at the identified timestep — not treating raw attention weights as the finding itself.

**Case 3: per-timestep alignment instead of window-endpoint-only.** Currently sequence-to-*vector* (window pooled to one vector, aligned to one HRF vector at the window's end). Extending to align the full per-timestep brain sequence against the full per-timestep HRF sequence would make Case 3 genuine sequence-to-sequence alignment (resolves the earlier "not actually seq2seq" correction — it would become true seq2seq under this redesign). Real costs: `y_hrf` needs extracting per-timestep, not just at the endpoint (straightforward `data_setup.py` change); the contrastive matrix grows from batch² pairs to batch²×window² pairs — a real computational jump.

**Case 2: pooling ablation (smaller scope than Case 3's redesign).** Mean-pooling or attention-pooling instead of last-token pooling, before alignment with the (still single, non-temporal) text prototype. Distinct in scope from Case 3's idea — a pooling swap, not a structural redesign of what's being aligned.

**Open-vocabulary-*like* CAV for Case 1 and Case 3, via soft class-decomposition (real gap-closer — Case 3 currently has NO open-vocabulary route at all, only the 5 fixed concepts).** Decouple "map an open claim onto something testable" from "test sensitivity in brain-representation space" — Case 2 only fuses these because it happens to have a shared brain-text embedding space; Case 1/3 don't need that space to do the first half. Reuse existing, already-built infrastructure (`contrastive.py`'s `CONDITION_DESCRIPTIONS` + `encode_condition_prototypes`, general-purpose MiniLM): embed an open claim, embed the 6 condition descriptions, cosine-similarity + softmax → a soft weighting over the 6 known classes (e.g. 0.5 right-hand, 0.3 left-hand) — no new training required. **Resolved (stress-tested, option 2 dropped)**: TCAV is only ever defined relative to one class at a time (a specific logit to differentiate + held-out examples of that same class). (a) **Combine scores**: run the existing, fully-specified per-class TCAV computation once per class, linearly combine the final scalars — well-defined, no ambiguity. (b) **Combine directions**: sum the classes' CAV directions first, then run one TCAV pass — genuinely underspecified, not just less simple: once directions are summed, there's no natural single choice of "whose logit, which held-out examples" left. A "virtual weighted logit" (weighted sum of the component classes' actual logits) is mathematically fine since gradients are linear, but the held-out-example-set question stays unresolved. **Use (a).** No literature precedent found for composing independently-fit CAVs this way via a soft text-derived weighting — closest adjacent ideas are Concept Bottleneck Models (Koh et al. 2020, trained end-to-end through the bottleneck, not post-hoc) and ACE (Ghorbani et al. 2019, auto-discovers concepts rather than composing predefined ones). Neither is quite this — treat as a genuinely open idea, not an established recipe. Feeds naturally from the existing literature claim-extraction step (`pipeline.py`), which already produces open claims consumed today only by Case 1's keyword-matching and Case 2's open-vocabulary route.

**Invert the RAG loop: corpus-first concept discovery, not just query-first verification (2026-08-22).** The existing loop is top-down: one decode → RSN attribution → search query → retrieve → LLM extracts a claim from *that* excerpt. A separate, complementary direction: sweep the *whole* corpus for testable claims with no decode or query involved at all — the extraction primitive already exists unchanged for this (`pipeline.py::build_concept_extraction_prompt` takes a bare excerpt, no query), it was just never run over more than a query's top-k retrieved chunks. Concretely: (1) a cheap keyword pre-filter over all 879 corpus chunks before spending any LLM call — deliberately a *broader* generic neuroscience-organization term list (somatotopic, homunculus, topographic, hemispheric, gradient, selectivity, effector, etc.), not the narrow 5-concept keyword list already used for phrase-to-concept mapping, specifically so the filter doesn't pre-decide the answer by only ever admitting chunks that were already going to map to a known concept. Measured empirically against the real corpus: the narrow list admits 301/879 chunks (34.2%), the broader list 432/879 (49.1%), with 136 chunks caught *only* by the broader filter — real, additional discovery surface. (2) Run extraction 3x per surviving chunk and require self-consistency **at the concept level, not exact-phrase level** (map each repeat's phrase independently, keep a concept only if it recurs in ≥2 of 3 repeats) — phrasing varies run to run even when the underlying claim is the same, so phrase-level exact-match consistency would be too strict. (3) Phrases that map to nothing in the known vocabulary are kept separately as a "discovery" list for manual review, rather than silently discarded. Timed on the real model (`mlx-community/Llama-3.2-3B-Instruct-4bit`): 3.3s/call, so 432×3 ≈ 72 minutes for the full sweep — run 2026-08-22, results in `results/corpus_claim_extraction_sweep.json`.

**Cross-paradigm consistency of a literature claim — only possible because of the 2026-08-22 bootstrap infrastructure.** A claim isn't just AGREE/DISAGREE against one trained model; it can be tested against all 3 cases' 30 independently-resampled-and-retrained models each (Case 1/2/3, GRU+Transformer, all checkpoints saved, all using the *same* 30 subject splits so the comparison is paired, not confounded by different subjects landing differently per case — see `results/case{1,2,3}_bootstrap_30resamples.json`). This turns a single-model verdict into a population-level statement: does a claim like "tongue representation is bilateral" hold regardless of which training objective produced the representation (supervised, supervised-contrastive, self-supervised), or is it an artifact of one paradigm's inductive bias? Directly motivated by an early real finding: Case 1's bootstrap already showed `tongue` and `baseline` flipping unstably between TCAV=0 and TCAV=1 across resamples under the `left_side`/`right_side` concepts (classes outside those concepts' own defining label sets) — plausibly not just a tie-breaking statistical artifact but a genuine reflection of tongue/orofacial muscles receiving more bilateral corticobulbar innervation than the strictly-contralateral corticospinal control of limb muscles. Testing this specific claim across all 3 paradigms would tell us whether that instability is architecture/training-paradigm-dependent or a stable property of the task itself.

**RAG for corpus growth, not just corpus search — a fourth, separate pipeline block (2026-08-22).** Distinguish four now-distinct RAG-adjacent components rather than treating "RAG" as one thing: (1) **corpus curation/growth** (new idea, not built) — decide which papers exist in the pool at all, e.g. querying the Semantic Scholar/PubMed APIs for papers related to an extracted claim or concept name and pulling in open-access PDFs; (2) **ingestion/indexing** (already built, needs zero new code) — `ingest_pdf_directory` treats "the corpus" as just whatever PDFs sit in a directory, so growth requires no pipeline change, only more files; (3) **query-time verification loop** (already built) — decode → attribution → query → retrieve → CAV-verify one specific prediction; (4) **corpus-wide extraction sweep** (being built now, see above) — mine every chunk for claims, independent of any query. Real, honest constraints on (1): fetching new papers means downloading files (needs explicit go-ahead each time), and paywalled literature stays paywalled — no CAPTCHA-bypassing or scraping around access controls, only open-access-first search via legitimate APIs.

**Concept-vector input attribution: given a CAV, where in the raw input does it show up? (2026-08-22)** TCAV's directional derivative measures sensitivity of a *class logit* to a concept direction in hidden-representation space — it doesn't say which part of the raw `[32, 300]` input window is driving alignment with that concept. But the concept-alignment score itself, `h(x) · v_C` for a fixed frozen CAV direction `v_C`, is just a differentiable scalar function of the input `x` — backprop it through the backbone to `x` and the result is a genuine input-level saliency map for a *concept*, not a class. No new architecture needed, only a different backward target than the class-logit gradient already used in `interpretability.py`. Real literature doing exactly this: Schrouff et al. 2021, *"Best of both worlds: local and global explanations with human-understandable concepts"* (Google) — combines TCAV-style global concept scores with local, per-input concept attributions, the closest direct match; Fel et al. 2023, *"CRAFT: Concept Recursive Activation FacTorization for Explainability"* — generalizes integrated-gradients-style attribution from class-level to concept-level targets; Ghorbani et al. 2019, *"Towards Automatic Concept-based Explanations"* (ACE) — the original TCAV follow-up, also localizes which parts of an input relate to a discovered concept. **Graph-native version, for a future SC-GNN block**: if structural connectivity ever gets added as a GNN block in front of the GRU/Transformer, the same idea has an established graph analog — GNNExplainer (Ying et al. 2019) and PGExplainer (Luo et al. 2020) attribute a GNN's output to specific nodes/edges/subgraphs; swapping their attribution target from "classification logit" to "CAV projection" is the direct structural-connectome version of "which ROI nodes and structural edges are responsible for this concept."

## 6.7 Terminology: representation, encoding, decoding, manifolds

- **Representation**: the internal numerical form a model computes from input for downstream use (the pooled 128-dim vector here). Not tied to a mechanism.
- **Encoder/encoding**: the function that *produces* a representation from raw input (GRU/Transformer backbones; MiniLM+projection; the HRF MLP).
- **Decoder/decoding — real terminology trap.** This project uses "decoder"/"decoding" in the **neuroscience sense**: predicting a task/behavioral variable *from* brain activity (`GRUDecoder`, "6-class movement decoding"). Different from the general ML/generative sense (representation → reconstructed input or generated sequence, e.g. an autoencoder/seq2seq decoder). Used loosely in front of someone thinking generatively, this invites confusion about why there's no generation happening. Have the disambiguation ready verbatim.
- **Manifolds**: a related, more geometric idea — high-dimensional real data lies near a much lower-dimensional curved surface within the ambient space; a good encoder "flattens" that manifold into a more linear coordinate system. **Directly operational here, not just abstract**: CAV's premise (a concept = a roughly linear direction) is implicitly a claim the manifold has been flattened enough for that concept — if not, a linear probe fails, which is exactly what probe accuracy checks.

---

## 7. RAG / verification loop block

**Retrieval side**: 8-paper neuroscience corpus, chunked, embedded with the same MiniLM family used elsewhere, dense cosine-similarity retrieval, optional cross-encoder reranking. Two-stage design has a real cost/accuracy reason: a bi-encoder embeds query and passage independently, so passage embeddings precompute once and search the whole corpus cheaply; a cross-encoder scores a query-passage pair *jointly* (full attention between them) — more accurate, but a fresh forward pass per pair, too expensive over the whole corpus, so applied only to re-rank the bi-encoder's narrowed candidate set. Standard retrieve-then-rerank pattern, motivated by a real tradeoff, not aesthetics.

Later measurably improved: domain-adaptive contrastive fine-tuning of the retrieval embedding model on in-domain query-passage pairs — top-1 chunk-retrieval accuracy 43.9%→61.0%, top-3 75.6%→85.4%, evaluated against an 880-chunk benchmark with real gold labels (not just "the pipeline runs").

**LLM's three roles**, per decoded window: stance labeling of a retrieved excerpt (supports/contradicts/unrelated), claim extraction (a specific testable phrase, e.g. "hand movement is contralateral"), final narrative synthesis.

**The verification loop — the actual novel contribution.** Literature-extracted phrase → mapped onto a testable concept (keyword match for closed vocabulary, or Case 2's open-vocabulary embedding route) → CAV direction fit/derived → TCAV score compared against the excerpt's stance. **Critical design decision: the AGREE/DISAGREE/UNCLEAR verdict is computed deterministically in code from (stance, TCAV score) — the LLM never decides it.** Unrelated excerpt → no comparison at all (genuine third outcome, not folded into disagreement); supporting + high TCAV → agreement; supporting + low TCAV → flagged disagreement. LLM's only job: narrate an already-determined verdict.

**Why this design — strongest thing to say about it**: not built around a hypothetical failure, built around a *measured* one. Freely judging agreement, the LLM defaulted to AGREE in 10/12 real cases regardless of the actual TCAV score — sycophancy-shaped, measured not guessed. Structural reason the LLM can't do this job even in principle (good answer to "why not just have the LLM reason about it directly"): it only ever sees *text* — no access to the model's internal representation, can't compute a directional derivative through a different model's weights. It can only be *told* the TCAV number; letting it also freely decide whether to trust that number reintroduces the exact measured failure.

**Honest gaps**: fix applied to Case 2's loop only — Case 1's loop still uses the older free-judgment design, not re-measured at the same scale. Corpus is small (8 papers). Phrase-to-concept mapping is keyword-based (not embedding-based) for the closed-vocabulary path.

**Generalizable lesson (relevant framing for a transportation/logistics ML role, not domain-specific)**: compute a decision deterministically from structured, checkable signals; restrict the LLM to narrating an already-decided outcome rather than trusting its free-form judgment on evidence it can't independently verify.

## 8. Statistical framework block

**Foundational principle, underlies everything below**: subject-level resampling, never window-level, everywhere (repeated splits, bootstrap CIs, the TCAV significance test). Overlapping windows within a subject aren't independent draws — window-level resampling would silently violate the i.i.d. assumption most resampling math relies on.

**Two genuinely different resampling procedures, worth keeping distinct:**
1. **Repeated-split resampling** (Case 1/2 architecture comparison): re-partition the subject pool into train/val/test, 20-40 times, retraining fresh each time. Answers "how much does which subjects end up where matter" — a real training-time variability question.
2. **Post-hoc bootstrap** (TCAV CIs): for one already-trained, fixed model, resample the test set's subject composition with replacement — no retraining. Cheaper, narrower — variability from test-set composition only, holding the trained model fixed. Does **not** capture training-time variability — explicitly flagged gap: Case 1's TCAV CI doesn't yet resample the direction-fitting/training data.

**Paired comparisons + Wilcoxon, not a paired t-test**: same random split used for both architectures within a repeat (not independent draws), enabling a signed-rank test on per-repeat differences rather than eyeballing overlapping CIs. Wilcoxon over t-test because it's non-parametric — no normality assumption on the differences (only symmetry around the median), uses ranks not raw magnitudes — matters with only 20-40 repeats, too few to lean on asymptotic normality.

**Adaptive stopping rule, not a fixed repeat count**: batches of 10, check whether running CI half-width has stabilized (below a threshold) before stopping, capped at 40. Avoids stopping before convergence or wasting compute past it. Also the concrete empirical answer to the N≫P worry (§6.4) — a harmfully overparameterized model would show an unstable, non-converging CI under this exact procedure; not observed.

**Bootstrap vs. permutation testing — precise distinction, terminology matters if pressed.** Bootstrap resamples *data* (with replacement) to estimate a statistic's sampling distribution — naturally a CI, though a CI that excludes a null value is itself a significance statement, so bootstrap *can* serve that role too. Classical permutation testing shuffles *labels/group assignments* under a null to build an explicit null distribution — naturally a p-value. Both can be pushed to do either job with adaptation; that's just the natural division of labor. This project's actual rank-bootstrap test is a hybrid worth naming precisely if pushed: bootstrap resampling, but producing a P(rank #1)-style statistic that functions like a significance statement — not a plain CI, not a classical permutation test either. Safe to just call it "bootstrap-based significance testing."

**A well-specified extension of the CI-for-TCAV gap (stress-tested, matches the documented plan precisely)**: fix the concept set up front; for each repeated train/val/test split (already retrained for macro-F1), retrain fresh and recompute CAV/TCAV on that repeat's model; get a distribution of TCAV scores *across repeats*, the same way macro-F1 gets one — not just a post-hoc bootstrap CI on one fixed model's test set. This is the complete version of the "resample the direction-fitting data too" gap already flagged in `population-level-evaluation-plan.md` §6. Real compute cost (retraining × concepts × repeats) — deferred alongside the rest of the paused sweep, but the design is fully specified and ready to build when that resumes.

**Reference anchor**: Misra & Pessoa (2025, *eLife*) — cited throughout as the direct methodological source (repeated subject-level resampling, paired non-parametric tests, bootstrap CIs), matching an established neuroimaging standard rather than an ad hoc evaluation scheme.

## 8.4 Case 1, full architecture pass (2026-08-20) — data, loss, backprop, block diagram

Session pivoted from resume-bullet drafting back to a from-scratch architecture review (Case 3 confusion prompted this) -- ViT/attention-paper style: full block diagram, loss functions with rationale and alternatives, then per-case CAV-RAG loop, then results. Case 1 done in full; Cases 2/3 and the CAV-RAG loop pass still to do.

**Notation convention adopted, extending Andrew Ng's course notation with a time index**: $x_{n,t}^{(m)}$ = ROI $n$'s signal at timestep $t$ in example (window) $m$. $X^{(m)} \in \mathbb{R}^{32 \times 300}$ is one window, $y^{(m)} \in \{0,...,5\}$ its scalar class label, $y_{hrf}^{(m)} \in \mathbb{R}^5$ its HRF vector, both read at the window's last timestep.

**Data facts, verified against the actual pipeline code, not recalled from memory**:
- No per-timepoint missing-value imputation. `validate_bundle()` in `02_data_complete.ipynb` hard-requires `np.isfinite(X).all()` -- any non-finite value anywhere fails the *entire run*, no partial exclusion.
- `valid_mask` is wired all the way through window construction (`data_setup.py` rejects any window where `valid_mask` isn't `True` for every timepoint) but is currently always `np.ones(...)` -- an unpopulated hook for future frame-level motion-scrubbing (the way the BGM/GRU precursor papers did with framewise-displacement thresholds), not active filtering right now. Honest gap, not silently missing.
- Classes are imbalanced (baseline dominates). Corrected via inverse-frequency class weighting in the CE loss: `weight[c] = N / (num_classes * count_c)` (`compute_class_weights` in `data_setup.py`) -- same formula as sklearn's `class_weight='balanced'`.

**Loss, exact math**: $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{MSE}$, $\lambda=0.1$.
$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_i w_{c_i} \log \hat y_{i,c_i}, \qquad \mathcal{L}_{MSE} = \frac{1}{5N}\sum_i \|\hat y_{hrf,i} - y_{hrf,i}\|_2^2$$

**Lambda sweep -- real correction, checked `notebooks/07_hyperparameter_sweep.ipynb` directly rather than trusting memory.** lambda=1.0 WAS tested (`LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.3, 1.0]`), contrary to initial recollection. At lambda=0.0 the HRF head is essentially untrained (MSE ~0.34, confirming the auxiliary loss is *necessary* not just helpful). GRU: classification flat across the whole range (0.908-0.912), HRF MSE improves monotonically with higher lambda -- lambda=0.1 was leaving free HRF quality on the table for GRU specifically, no classification cost to raising it. Transformer: real tradeoff -- lambda=0.1 is the classification sweet spot (0.925), lambda=1.0 gets the best HRF fit (MSE 0.019) at a small classification cost (0.918). lambda=0.1 stayed the default across the main results likely for cross-architecture comparability, not because it was shown optimal -- worth stating precisely, not implying it was simply the best value found.

**Kendall et al. (2018) uncertainty-weighted multi-task loss -- flagged by the user as worth trying later, real follow-up, not just interview color.** Learn per-task weights via a homoscedastic-uncertainty parameter trained jointly with the network: $\mathcal{L}(\theta,\sigma_1,\sigma_2) = \frac{1}{\sigma_1^2}\mathcal{L}_{CE} + \frac{1}{2\sigma_2^2}\mathcal{L}_{MSE} + \log\sigma_1 + \log\sigma_2$. Self-balancing: a task with persistently high loss gets interpreted as high uncertainty, its weight shrinks automatically. **TODO if picked up**: replace the fixed lambda_hrf=0.1 with this and compare against the manual-sweep results in `results/hyperparameter_sweep_results.json`.

**Multi-task-as-regularization, refined with a real citation.** User's own framing: fitting one task admits a set of possible representations; fitting multiple tasks shrinks that set to the intersection. Extended: HRF (continuous) additionally forces the representation to stay *graded* within a class, not collapse to one point per class. This maps onto **Neural Collapse** (Papyan, Han & Donoho, 2020, *Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training*) -- pure cross-entropy training provably collapses within-class representations toward their class mean as training converges (class means arrange into a simplex equiangular tight frame). The HRF term counteracts exactly this, since HRF varies continuously within a class and the representation must retain resolution to predict it.

**HRF backprop through a 5-dim output, confirmed correct.** Loss stays scalar regardless of output dimensionality. Gradient w.r.t. the 5-dim regression output: $\partial\mathcal{L}/\partial\hat y_{hrf,d} = \frac{2}{5}(\hat y_{hrf,d}-y_{hrf,d})$ per dimension, a 5-vector. Backprops through the linear head ($\hat y_{hrf}=Wh+b$) via $W^\top \cdot (\partial\mathcal{L}/\partial\hat y_{hrf}) \in \mathbb{R}^{128}$ -- a Jacobian-vector product where the Jacobian of a linear layer is just $W$ itself. At the shared representation $h$ (feeding both heads), gradients from the classification and regression paths *add* -- ordinary multivariable chain rule at a fan-in node in the computation graph: $\partial\mathcal{L}/\partial h = \partial\mathcal{L}_{CE}/\partial h + \lambda\,\partial\mathcal{L}_{MSE}/\partial h$.

**Design taxonomy, refined**: Case 1 is both seq-to-scalar (classification, y is a single class index) and seq-to-vector (HRF regression, 5-dim) simultaneously, through its two heads off one shared trunk -- ties back to the earlier seq-to-scalar/seq-to-vector distinction used for Case 2/3.

**Block diagram**: rendered via the visualize tool (title `case1_multitask_architecture`) -- input window -> swappable GRU/Transformer encoder (architecture held fixed across the comparison) -> pooled 128-dim representation h -> two heads (classifier, HRF regression) -> combined multi-task loss. Not reproduced here since it's a rendered artifact, not text.

## 8.4.5 Case 2, full architecture pass

**Brain path**: same encoder as Case 1 (shared code, `forward_features`), same 128-dim pooled representation, then diverges: `Linear(128->64)` projection + L2-normalize -> $z_{brain}$, a unit vector on the 64-dim hypersphere.

**Text path**: 6 fixed condition descriptions, embedded once offline by a frozen pretrained MiniLM (384-dim, weights never update), then a small *trainable* `Linear(384->64)` projection + L2-normalize -> $z_{text} \in \mathbb{R}^{6\times64}$, six fixed unit-vector prototypes.

**Combine**: $\text{logits} = \tau \cdot z_{brain} z_{text}^\top$ ($[batch,6]$ similarity matrix), $\tau = e^{\text{log\_temperature}}$ learned (init $1/0.07\approx14.3$, matching CLIP's init, clamped at 100 for numerical stability). Loss: $\mathcal{L} = \mathcal{L}_{CE}(\text{logits}, y)$ -- plain softmax CE, but over similarities to text prototypes rather than a freely-learned classifier's logits.

**Why not literal CLIP -- precise distinction, common confusion point.** CLIP is symmetric (unique caption per image, so both image->text and text->image directions run, each batch element its own target). Case 2's text side has only 6 fixed prototypes shared dataset-wide -- one prototype maps to hundreds of different brain windows, so there's no sensible text->brain direction the way CLIP has text->image. Only brain->text runs. And the positive target is selected via the window's *true label*, not natural pairing (CLIP) or self-supervised augmentation (SimCLR) -- that's what makes this supervised contrastive (SupCon-style), not CLIP-style, despite the surface resemblance.

**Why a learned temperature**: controls softmax sharpness over similarities -- too sharp, noisy/unstable early gradients; too smooth, the model never commits. Learning it (as CLIP does) self-calibrates over training instead of one more hand-tuned hyperparameter.

**Real alternatives**: fixed hand-tuned temperature (simpler, back to manual tuning); older margin-based losses (triplet/contrastive-with-margin, predate InfoNCE-style softmax contrastive losses, generally underperform once there are several negatives to contrast against -- exactly the situation here, 5 negative prototypes per positive). A literal symmetric CLIP loss isn't structurally available here at all -- that had to wait for Case 3, where the alignment target (HRF) is continuous and unique per window, not a fixed 6-item vocabulary.

**Block diagram**: rendered via the visualize tool (title `case2_contrastive_architecture`) -- two parallel branches (brain: encoder -> projection -> z_brain; text: frozen MiniLM -> trainable projection -> z_text) converging into a similarity computation, then the contrastive loss. Not reproduced here since it's a rendered artifact.

## 8.4.6 Case 2 deep-dive: contrastive learning theory, CLIP/SimCLR precision checks, real follow-up ideas

Dense two-round Q&A on contrastive learning mechanics. Key precision corrections and confirmed extension ideas, worth having exact for the interview.

**Softmax-CE gradient mechanics for the prototype embeddings, derived precisely.** For a brain example with true class $c$: $\partial\mathcal{L}/\partial z_{text,c} = \tau z_{brain}(p_c-1)$ (pulls true prototype toward $z_{brain}$); $\partial\mathcal{L}/\partial z_{text,k} = \tau z_{brain}\,p_k$ for $k\neq c$ (pushes other prototypes away). No direct prototype-prototype term exists in the loss anywhere -- apparent prototype separation is an emergent, indirect consequence of shared mediation through $z_{brain}$ across the dataset, not direct repulsion. Precision worth having if asked the exact mechanism.

**The actual CLIP loss** (for calibration against our asymmetric version): full $[N,N]$ similarity matrix over a batch, $\text{logits}[i,j]=\tau z_{img,i}\cdot z_{txt,j}$, symmetric CE along both axes using the identity as implicit labels, averaged: $\mathcal{L}=\frac12[\text{CE(rows)}+\text{CE(cols)}]$. Requires one unique $z_{text}$ per $z_{brain}$ -- structurally why our 6-shared-prototype design can't do this directly.

**Temperature has a real physics origin**, not an arbitrary name -- literal reuse of the Boltzmann distribution's temperature parameter ($p\propto e^{-E/T}$), entered deep learning via Hinton, Vinyals & Dean (2015) distillation, inherited by contrastive learning from there.

**Normalization to unit vectors is near-universal in this method family** for a real reason: makes the dot product = cosine similarity (bounded, scale-invariant), and removes a trivial shortcut (inflating similarity by growing vector norm rather than learning genuine directional alignment).

**Terminology, confirmed against the actual code** (`concepts_case2.py` operates on `brain_backbone.forward_features(x)`): "representation"/"features" = the 128-dim pre-projection $h$ (what CAV/TCAV actually probes, shared across all 3 cases). "Embedding" = the 64-dim post-projection $z$ vectors specifically, a distinct object. Not interchangeable in this project's own vocabulary.

**Real correction made mid-discussion**: an earlier claim that "there's no sensible text->brain direction" was too strong. A literal index-based CLIP loss doesn't apply (many brain windows share one prototype), but a **multi-positive symmetric loss** (Khosla et al. 2020, SupCon-style: for a given text prototype, every brain window with that true label is a positive, everything else negative) is architecturally buildable right now, frozen MiniLM and all -- no need to train a text encoder.

**The zero-shot capability the user got excited about is already built and validated, not a new thing to build.** `open_vocabulary_concept_direction` already embeds arbitrary unseen phrases via frozen MiniLM + trained projection and generalizes correctly (tongue phrase example, TCAV=0.686, P=1.000 across 1000 resamples) -- exactly because MiniLM is a general-purpose pretrained encoder. This is CLIP's zero-shot mechanism (shared embedding space + frozen general-purpose encoder handling novel categories) applied to concept-sensitivity testing rather than direct classification.

**Adjoint (transpose) pullback vs. inverse -- real, important correction.** `brain_projection.weight.T` used in the open-vocab CAV mechanism is the map's *adjoint* ($\langle Wh,z\rangle=\langle h,W^\top z\rangle$), not an inverse -- $W\in\mathbb{R}^{64\times128}$ isn't square and isn't invertible; going 64-dim back to 128-dim is fundamentally underdetermined (infinitely many $h$ map to the same $z$). The adjoint correctly pulls back a *direction* (what TCAV needs for a directional derivative) but does NOT reconstruct a specific point $h_{brain}$ "the text came from." A literal point reconstruction would need a Moore-Penrose pseudoinverse, giving one particular minimum-norm answer among infinitely many valid ones -- architecturally closer to the parked, out-of-scope generative HRF-to-brain direction than to anything currently built.

**Concrete follow-up experiments logged, not yet built:**
1. **Symmetric multi-positive loss** (Khosla et al. 2020 SupCon-style), MiniLM frozen -- safer, likely to work, direct extension of the existing architecture.
2. **Literal full-CLIP-style with a trained text encoder** (not frozen MiniLM) -- riskier, may just memorize the 6 texts and produce an informative negative result; worth trying and reporting honestly either way.
3. **Nonlinear (MLP + ReLU) projection head** instead of the current linear projections, per SimCLR (Chen, Kornblith, Norouzi & Hinton, 2020) -- their most-reused finding, that a nonlinear projection before the contrastive loss improves the quality of the pre-projection representation used downstream. Note: SimCLR's "benefits from larger batch size" finding does NOT straightforwardly apply to the *current* fixed-6-prototype design (negatives per example = 5, always, independent of batch size) but becomes directly relevant once experiment 1 or 2 is built, where negatives genuinely scale with batch size.
4. **Literal zero-shot classification test**: embed a genuinely novel condition never in the 6 training classes (e.g. a hypothetical "elbow movement") and check whether the brain encoder's representation lands closest to that new embedding -- distinct from the already-validated concept-sensitivity use of open-vocabulary CAV.
5. Keep both the current asymmetric version and any new symmetric version as separate, compared results (user's own instinct) -- don't just replace one with the other.

## 8.4.7 Case 3, full architecture pass

**The unlock**: Case 3's brain path is architecturally identical to Case 2's (same encoder, same `Linear(128->64)` + normalize). The entire difference is what's on the other side, and that one difference cascades into everything else.

**HRF path**: target is $y_{hrf}^{(m)} \in \mathbb{R}^5$, the *same window's own* HRF vector (the exact quantity already used as Case 1's auxiliary regression target). Small MLP built from scratch, no pretrained starting point (unlike MiniLM, there's no existing "HRF language model" to freeze): `Linear(5->32) -> ReLU -> Linear(32->64)`, then normalize -> $z_{hrf}$, 64-dim unit vector.

**Why Case 3 gets the symmetric loss and Case 2 structurally cannot -- the actual crux.** Case 2's text side: 6 fixed, repeated prototypes, many brain windows share one target, no sensible text->brain direction (no unique brain window per text). Case 3's HRF side: continuous, genuinely different per window even within the same class (depends on exact position within the movement block) -- for a batch of $B$ windows, $B$ distinct targets, literal CLIP-style batch-index InfoNCE applies naturally in both directions:
$$\text{logits} = \tau Z_{brain}Z_{hrf}^\top \quad (B\times B, \text{ not } B\times 6), \qquad \mathcal{L}=\tfrac12[\text{CE}(\text{logits},\mathbf{i})+\text{CE}(\text{logits}^\top,\mathbf{i})]$$
Case 3 gets the full symmetric CLIP loss for free, precisely because of what kind of target HRF is -- not a deliberate design choice to make it "more advanced."

**Why this makes it self-supervised, not supervised-contrastive like Case 2**: pairing is defined purely by temporal co-occurrence (window $m$ pairs with window $m$'s own HRF because they co-occurred, nothing else). The class label $y$ never enters this loss anywhere.

**Real architectural consequence**: no label used during training -> no classifier head at all -> nothing to evaluate accuracy against, no target-class logit for CAV/TCAV to differentiate against. This is exactly why `BrainWithPostHocClassifier` exists -- fit a linear probe on frozen features *after* training, purely for evaluation and to give CAV something to work with. Case 1 and Case 2 both have some label-derived readout baked into training; Case 3 structurally cannot.

**Block diagram**: rendered via the visualize tool (title `case3_selfsupervised_architecture`) -- parallel to Case 2's diagram structurally (brain branch identical), but the text branch is replaced by an HRF branch (same-window continuous target, small from-scratch MLP instead of frozen pretrained + projection), and the loss box is explicitly symmetric (both directions) rather than one-directional. Not reproduced here since it's a rendered artifact.

**Language correction, apply going forward**: don't say CLIP-style / SupCon-style training is "not possible" for Case 2 -- it's not built by default, but both are architecturally buildable (see 8.4.6's logged follow-ups) now that the mechanism is understood. "Not possible" overstates it.

**Case 2 vs Case 3, sharpest version of the distinction (refined from an initial user intuition, checked precisely)**: not that Case 2's trainable projection is parameter-starved (a 384x64 matrix has plenty of capacity to place 6 points flexibly) -- the real constraint is that MiniLM's input set is **finite by construction**: exactly 6 condition descriptions exist, so $z_{text}$ can only ever take one of 6 possible values, full stop, regardless of training. Case 3's HRF target isn't drawn from a finite set at all -- $y_{hrf}$ varies continuously per window, so $z_{hrf}$ takes a genuinely continuous range of values, and because the HRF encoder is a smooth function (an MLP, not a lookup table), nearby HRF vectors land near each other in $z_{hrf}$-space too -- a continuous embedding manifold, not a handful of discrete anchors. Discreteness vs. continuity of the *target space* is the precise structural reason, not encoder capacity.

## 8.5 Message-point critique and refinement (2026-08-19/20, IN PROGRESS, not yet finalized)

First attempt at a 5-message breakdown (mirroring BGM's process) was rejected by the user as disconnected — see full critique below, each point still worth having precise:
1. "accuracy is insufficient" as a message is field-level context (the premise of interpretability as a field), not specific to this project — cut.
2. The actual core finding is genuinely more modest than BGM's, and worth being honest about rather than oversold: with accuracy matched across all 3 paradigms, the whole distinguishing question is what CAV/TCAV found. The real finding is the effector/laterality asymmetry, replicated via 2 independent derivation methods (Case 1/2) -- not "everything is similar, nothing to report." Case 3 (self-supervised, tested as a third stress-test) did NOT cleanly replicate this -- it hit a ceiling-saturation problem (near-universal separability) that makes the significance test inconclusive for laterality specifically, a different failure mode than Case 1/2's clean "weak" result, not simply "didn't replicate."
3. The LLM/verification-loop point was disconnected without first establishing the loop's purpose. RESOLVED via the user's own origin story: the motivation wasn't an abstract "literature-grounding is good science" principle -- it was "I'd have to manually search and read literature to check whether my model's representations line up with anything real, so let RAG + LLM do that legwork for me." Building it, the LLM was found to be hallucinating support / agreeing regardless of evidence (sycophancy). The deterministic-verdict fix IS the "evaluation loop" -- verifying that the automation built to replace manual literature-checking could actually be trusted.
5. The engineering/local-pipeline point and the LLM point aren't two separate messages -- they're one arc: probe for concepts -> ground against literature -> verify that the verification step itself is reliable.
4. (Rigor) -- deferred by the user ("i am totally off here, we need to sit and figure out tomorrow").

Tentative summary line proposed, user response: "it makes some sense to me, maybe once we write down the bullet points everything falls into place" -- not yet fully validated:
*Built a system that doesn't just decode brain activity, but checks whether the specific concepts a model appears to rely on are actually supported by real published findings, and made sure that check itself could be trusted rather than just sounding plausible.*

**Second attempt, explicitly anchored to the GRU/PLOS 2021 precursor project** (see `docs/interview-prep-precursor-projects.md`) as "what NeuroLens-RAG does beyond that 2021 baseline" -- proposed, not yet reacted to by the user before the session pivoted to a full architecture deep-dive (see below):
1. Scope -- extended a single decoding paradigm (2021 GRU classifier) into three deliberately different representation-learning paradigms on the same task.
2. Core finding, replicated then honestly stress-tested -- the effector/laterality asymmetry via 2 methods, then a genuine limit found (not confirmed or refuted) when stress-tested against a third.
3. Region-level to concept-level validation -- the direct upgrade from the 2021 paper's network-lesion check to testing a specific, human-named concept.
4. The verification loop as one arc -- automate manual literature-checking, find the automation isn't reliable, fix it.
5. Statistical rigor -- population-level validation for architecture comparisons specifically (distinct from message 2's finding-level rigor).

**Session pivot (2026-08-20)**: user reported difficulty understanding Case 3 clearly and asked to step back from resume-bullet drafting entirely, in favor of a full from-scratch architecture walkthrough of all 3 cases (attention/ViT-paper style: precise architecture diagrams, loss functions with rationale and alternatives, then the CAV-RAG loop per case, then results) -- see next section. Message-point refinement above is paused, not abandoned; resume once the architecture pass rebuilds a clearer foundation.

## 8.6 Interpretability block, rebuilt from scratch (2026-08-20)

CAV/TCAV was covered at a high level earlier (section 6/7 above) but didn't fully land once Case 2's derivation variant came up. Rebuilt here as the core mechanism, using Case 1 (direct labeled-example probe, no extra derivation tricks) as the clean running example, before revisiting how Case 2/3 get their *direction* differently while reusing this exact testing machinery.

**The question CAV/TCAV answers**: not "does the model classify tongue windows correctly" (accuracy) -- "does the model's decision depend on the concept of tongue-movement the way a human defines it, or does it just correlate with something else."

**The six steps**:
1. Define the concept via labeled examples: positive (true tongue windows) vs. negative (everything else). For each, run the *already-trained* model forward and take its pooled representation $h \in \mathbb{R}^{128}$ -- the model's internal state, not the raw input.
2. Fit a linear probe: ordinary logistic regression on $\{(h_i, \text{concept present?})\}$. Accuracy is a diagnostic, not the point -- low accuracy means the concept isn't linearly present at all, don't trust anything downstream.
3. The CAV is the probe's weight vector, normalized: $v_C = w/\|w\|$, a unit vector, geometrically perpendicular to the probe's decision boundary, pointing toward "more concept."
4. Take held-out examples that are *truly* the target class. Compute the gradient of that class's logit w.r.t. the representation: $g = \partial(\text{logit}_{target})/\partial h$ -- "if h moved this way, how much would the logit move."
5. Directional derivative: $g \cdot v_C$. Positive means nudging $h$ toward "more concept" would increase the target logit.
6. TCAV score = fraction of held-out examples where step 5 is positive. Near 1.0 = strong, consistent dependence. Near 0.5 = no real relationship.

**Concrete anchor with real project numbers**: tongue concept -- probe accuracy ~0.99 (cleanly linearly separable), TCAV ~1.0 (every held-out example's directional derivative positive). Laterality -- probe accuracy still decent (concept is *findable*), but TCAV only 0.44-0.74 (the model's actual decision is far less consistently sensitive to it). The gap between "concept is linearly findable" (probe accuracy, step 2) and "the decision actually depends on it" (TCAV, steps 4-6) is the entire reason this is two separate steps, not one -- a probe could succeed while TCAV reveals the model doesn't actually use that direction to decide.

**Block diagram**: rendered via the visualize tool (title `cav_tcav_mechanism`) -- two parallel paths (labeled examples -> probe -> CAV direction; held-out target examples -> representation -> gradient) converging into a directional-derivative dot product, then aggregated into the TCAV score. Not reproduced here since it's a rendered artifact.

**Still to cover**: how Case 1/2/3 get the CAV *direction* differently (direct probe / text-arithmetic pullback / post-hoc-classifier probe) while reusing this exact testing machinery unchanged -- next step once the core mechanism above is confirmed solid.

## 8.6.1 Follow-up precision round: multi-class concepts, nonlinear probes, RAG-loop generality, sample size, "Case 4"

**Multi-class concepts (e.g. right_side = right_hand + right_foot), confirmed against `run_concept_analysis`**: one combined CAV direction from the pooled positive examples, but TCAV *scoring* runs separately per class -- right_hand's own held-out examples/logit tested against the direction, right_foot's own examples/logit tested separately, two distinct numbers not one average. Matches the actual Case 3 output (`right_side` scored 1.0 for right_hand AND 1.0 for right_foot as separate entries). "Dependence on the concept" means checking whether *both* component classes come out high -- if only one did, that's informative (direction captured one effector's laterality but not the other's).

**CAV probes must be linear -- hard requirement of the method, not a style choice.** A CAV needs a single well-defined direction to extract; a nonlinear (MLP) head has no equivalent. This is also *why* probe accuracy is a meaningful diagnostic of representation quality specifically -- a nonlinear head could recover information from a badly-organized representation too, which would break the diagnostic. Legitimate separate use for a nonlinear head: report it purely as an additional comparison metric (e.g. "linear probe 91.6%, MLP probe 93.2%") -- the gap is itself informative (task-relevant information present but not linearly accessible) -- but it's not usable for CAV/TCAV.

**The RAG-CAV loop is architecture-agnostic by design, confirmed.** It only needs *some* model exposing `forward_features` + `classifier`; doesn't matter whether that's Case 1, either new Case 2 variant, Case 3, or a future case. This is exactly why the consistent interface convention mattered -- it's what made Case 3 cheap to add, and will make the new Case 2 variants equally cheap.

**Sample size for a stable CAV -- no clean formula, but a better empirical check already half-exists.** Classical logistic-regression heuristic: events-per-variable (EPV) >= 10 (Peduzzi et al. 1996) -- for a 128-dim probe, roughly 1,280+ positive examples by that rule. Caveat: built for *unregularized* logistic regression used *inferentially* (reliable per-coefficient p-values), doesn't map cleanly onto a regularized (sklearn default L2) probe used to extract a *direction*. More rigorous, partially-already-built answer: bootstrap-resample the *training* data used to fit the probe, check whether the resulting CAV direction and downstream TCAV scores stay stable across resamples -- this is precisely the "resample the direction-fitting data too" gap already flagged in `population-level-evaluation-plan.md` §6, and it's a direct empirical answer rather than a borrowed textbook rule.

**"Case 4" (generative HRF-from-brain) -- real pushback given, not yet a genuinely new case.** As described (predict $y_{hrf}$ from $X$), this is exactly what Case 1's existing HRF regression head already does -- deterministic point estimate, MSE-trained. For this to be a genuinely distinct generative case, it needs to model the *distribution* over $y_{hrf}$ given $X$ (samplable, uncertainty-aware -- e.g. a VAE/flow/diffusion-style decoder), not a point prediction. Same direction as the already-parked, explicitly out-of-scope HRF-to-brain generative decoder in `case2-3-design-plan.md`, just the reverse mapping. The post-hoc-classifier pattern (freeze representation, fit linear probe) would apply unchanged regardless of whether this gets built.

## 8.7 Implementation sprint (2026-08-21), scoped given 3 days to the interview

Full request was: layer-count sweep for GRU/Transformer, full bootstrap-resampling evaluation, both new contrastive paradigms (SupCon-style + literal CLIP-style), and CAV testing on the representations. Given the interview is 3 days out (today the 21st, interview ~24th), scoped explicitly rather than attempting everything -- communicated to the user, not silently decided:

**Priority order:**
1. Build the SupCon-style multi-positive symmetric loss for Case 2 (frozen MiniLM, no architecture change) -- DONE, `src/neurolens/contrastive.py` (`supcon_text_to_brain_loss`, `symmetric_supcon_loss`, `run_epoch_supcon`, `train_contrastive_supcon`). Kept fully separate from the original asymmetric `train_contrastive` so both can be trained and compared, not one replacing the other.
2. Quick, honest single-split comparison (5 epochs, matching the standard Case 2 training regime) + fixed 5-concept CAV/TCAV test on the new SupCon model, compared against the same test on the original asymmetric model -- RUNNING (`case2_supcon_comparison.py`, background), results to `results/case2_supcon_vs_asymmetric_results.json`. Smoke-tested first (1 epoch, macro F1 0.875, no errors) before committing to the full run.
3. Literal full-CLIP variant with a trained text encoder -- NOT STARTED, attempt only if 1-2 land cleanly; genuinely riskier (reopens the "only 6 unique texts" data-scarcity concern from earlier), and an honest negative result there would itself be a legitimate finding given this project's track record.
4. Layer-count sweep (deeper GRU, more Transformer heads/layers) and the full bootstrap-resampling treatment across all variants -- DEFERRED, documented as future work, not attempted before the interview. Most expensive, least interview-narrative-critical: the interview needs "built X, found Y," not full production-grade statistical treatment on every variant.

**SupCon loss, exact formulation (Khosla et al. 2020, "L_out" multi-positive variant)**: for text prototype row $c$ in the $[\text{num\_classes}, B]$ text-to-brain logits, every brain example in the batch with true label $c$ is a positive. Log-softmax taken over the full row (all $B$ brain examples as candidates), averaged only over that row's positive columns, then averaged across classes present in the batch. Combined with the existing brain-to-text cross-entropy, averaged: $\mathcal{L} = \tfrac12[\mathcal{L}_{b2t} + \mathcal{L}_{t2b}]$.

## 11. Full implementation results and fresh resume draft (2026-08-22)

**Supersedes §9's draft v1 below** — that draft used single-split numbers and pre-dates all of this. Kept for history, not for use.

### 11.1 Bootstrap infrastructure and architecture comparison

GRU defaulted to 2 layers (was 1) in `model_builder.py`, closing the GRU/Transformer parameter mismatch from 1.83x to 1.15x. 30 repeated-split bootstraps trained with checkpoints saved (not discarded) for all three cases, all on the *same* 30 subject partitions (`results/case{1,2,3}_bootstrap_30resamples.json` — capped at 30 of an originally-planned 100 for time), enabling paired comparisons rather than eyeballing overlapping CIs.

| Case | GRU mean F1 (95% CI) | Transformer mean F1 (95% CI) | Winner (paired Wilcoxon) |
|---|---|---|---|
| 1 (supervised) | 0.906 [0.876, 0.931] | 0.922 [0.902, 0.945] | Transformer, p<0.0001 |
| 2 (supervised-contrastive) | 0.900 [0.866, 0.927] | 0.918 [0.898, 0.942] | Transformer, p<0.0001 |
| 3 (self-supervised) | 0.928 [0.903, 0.946] | 0.922 [0.903, 0.944] | GRU, p=0.0004 |

**Real finding**: architecture ranking flips by paradigm — Transformer wins when labels drive training directly (Cases 1, 2), GRU edges ahead in the one paradigm where the model never sees a label during representation learning (Case 3).

**Split-construction mechanics, precisely** (matches `notebooks/12_population_level_evaluation.ipynb`'s established convention, not reinvented): **repeated random re-partitioning, not literal with-replacement bootstrap** — `data_setup.py`'s window index is keyed by unique `(subject, run)` pairs, so a subject drawn twice under true bootstrap resampling wouldn't actually contribute its windows twice; the infrastructure de-duplicates by construction. Each of the 30 resamples reshuffles the same 90-subject pool (10 more permanently reserved for hyperparameter decisions only) into a fresh 65/13/12 train/val/test partition, seeded `SEED_BASE=2000 + resample_index` (deliberately disjoint from notebook 12's own `SEED_BASE=1000`, so the two studies' splits never collide). Model-init seed fixed at 42 across every resample — all variability comes from which subjects land in which split, not from a different random initialization. Case 2 and Case 3's bootstraps reuse Case 1's *exact* 30 splits (loaded from `case1_bootstrap_100resamples.json`, never redrawn), so a given resample index means the same subjects in all three cases — this is what makes the architecture-comparison table above and the concept comparison in §11.3 genuinely paired, not just three separately-resampled studies compared informally. Capped at 30 of an originally-planned 100 for time (~68-80s per resample across both architectures); checkpoints saved for every resample of every case, not discarded, specifically to make per-resample concept testing (§11.3) possible later without retraining.

### 11.2 Case 2 loss naming correction

`contrastive.py`'s `supcon_*` functions renamed to `multi_positive_*` / `symmetric_multi_positive_prototype_loss` — **not** literal SupCon (Khosla et al. 2020): anchors are the 6 text prototypes, never a brain-to-brain pair (`z_brain` never multiplies `z_brain`). Only the multi-positive averaging trick is borrowed. Also added a literal CLIP-style variant (`clip_loss`/`train_contrastive_clip`, naive in-batch negatives) — surprisingly scored *higher* test macro-F1 (0.9165) than both the original asymmetric baseline (0.9080) and the multi-positive loss (0.9137) on the fixed 100-subject split, contradicting the a-priori false-negative-penalty hypothesis (plausibly because batch size 64 across only 6 classes still leaves the softmax dominated by genuinely-different-class negatives most of the time).

**Exact formulation, multi-positive class-prototype loss** ($B$=batch size, $K$=6 classes, $\mathbf{z}_\text{brain}\in\mathbb{R}^{B\times d}$, $\mathbf{z}_\text{text}\in\mathbb{R}^{K\times d}$, both L2-normalized, $\tau$=temperature):

Similarity matrix $S\in\mathbb{R}^{B\times K}$, $S_{ic} = \tau \cdot \mathbf{z}_{\text{brain},i}\cdot\mathbf{z}_{\text{text},c}$.

Brain→text (unchanged from the original asymmetric loss): $\mathcal{L}_{b2t} = -\frac{1}{B}\sum_i \log\frac{\exp(S_{i,y_i})}{\sum_c \exp(S_{ic})}$ — plain $K$-way cross-entropy.

Text→brain (the new piece): for class $c$, let $P(c)=\{i : y_i=c\}$. Softmax column $c$ over all $B$ windows: $p_{ci} = \exp(S_{ic})/\sum_j \exp(S_{jc})$. Average the negative log over $c$'s positives only: $\mathcal{L}_c = -\frac{1}{|P(c)|}\sum_{i\in P(c)} \log(p_{ci})$, then average over classes present in the batch: $\mathcal{L}_{t2b} = \frac{1}{|C|}\sum_{c\in C}\mathcal{L}_c$.

Total: $\mathcal{L} = \frac{1}{2}(\mathcal{L}_{b2t}+\mathcal{L}_{t2b})$.

**Exact formulation, literal CLIP variant** (Case 2's `clip_loss` and Case 3's `symmetric_contrastive_loss` are the same math, different pairing source): gather one target embedding per sample — `z_text_per_sample[i]` = the text prototype of window $i$'s own class (Case 2), or `z_hrf[i]` = window $i$'s own concurrent HRF encoding (Case 3, genuinely one-to-one, no collisions). $\text{logits} = \mathbf{z}_\text{brain}\,\mathbf{z}_\text{target}^\top \cdot \tau \in \mathbb{R}^{B\times B}$. $\mathcal{L} = \frac{1}{2}\big(\text{CE}(\text{logits}, \text{arange}(B)) + \text{CE}(\text{logits}^\top, \text{arange}(B))\big)$ — the diagonal is the only positive, symmetric in both directions.

### 11.3 Extended concept set and the systematic 8-concept x 3-case CAV/TCAV comparison

3 new concepts added (`EXTENDED_CONCEPT_DEFINITIONS`), each grounded in real motor anatomy: `movement_vs_rest` (standard task-vs-rest GLM contrast), `limb_vs_orofacial` (corticospinal vs. corticobulbar pathway), `upper_vs_lower_limb` (hand vs. foot, isolated from tongue/baseline). Full sweep: 8 concepts x 3 cases x 2 architectures x 30 resamples = 1,440 model-concept evaluations, ~5 minutes wall-clock, CPU only (`results/all_cases_cav_sweep_8concepts.json`).

| Concept | Case 1 | Case 2, text-derived | Case 3 |
|---|---|---|---|
| hand | 1.000 | 0.630±0.31 | 1.000 |
| foot | 1.000 | 0.707±0.29 | 1.000 |
| tongue | 1.000 | 0.653±0.35 | 1.000 |
| right_side | 1.000 | 0.332±0.15 | 1.000 |
| left_side | 1.000 | 0.325±0.17 | 1.000 |
| movement_vs_rest | 0.987 | 0.598±0.21 | 0.903 |
| limb_vs_orofacial | 1.000 | 0.631±0.26 | 0.900 |
| upper_vs_lower_limb | 1.000 | 0.535±0.27 | 1.000 |

**The headline finding: Case 2's weakness was a derivation-method artifact, not a representation-quality gap.** Confirmed by fitting a logistic-regression probe on Case 2's frozen features via the *exact same method* as Case 1/3 — `case3.py`'s `fit_post_hoc_classifier` reused completely unmodified, since `ContrastiveModel` shares the same `.brain_backbone`/`.brain_projection` attribute names as `BrainHRFModel` (`results/case2_fitted_probe_cav_sweep.json`). Result: probe accuracy 0.997-0.999, TCAV 0.92-1.00 across all 8 concepts — essentially identical to Case 1/3. The mechanism, understood not just observed: MiniLM's sentence embeddings organize primarily around the concrete noun (hand/foot/tongue is the dominant semantic axis), so averaging-and-subtracting to isolate "laterality" (a weak, secondary axis) fights the text space's actual geometry and can flip sign (`right_side`/`left_side` scored 0.33 overall, 0.17-0.21 for Transformer specifically — systematically below chance, not noise), while concepts aligned with the dominant axis (hand/foot/tongue) degrade only moderately.

**Conclusion for the resume**: all three paradigms converge to near-perfectly interpretable representations when evaluated correctly. The interesting result is methodological — CAV-derivation method matters independently of representation quality, and this was caught and diagnosed via a targeted controlled experiment, not assumed — not a paradigm-superiority result.

### 11.4 Corpus-wide literature extraction sweep

879 total chunks in the 8-paper corpus. A broad keyword pre-filter (deliberately wider than the existing 5-concept keyword list, to preserve discovery) admits 432/879 chunks (49.1%) vs. 301/879 (34.2%) for the narrow list — 136 chunks caught only by the broader filter. 3x self-consistency extraction per surviving chunk using the local `mlx-community/Llama-3.2-3B-Instruct-4bit` model (`build_concept_extraction_prompt` — already existing, already query-independent, just never previously run over more than one query's top-k retrieved chunks) — 36 minutes wall-clock for 432x3=1,296 calls (`results/corpus_claim_extraction_sweep.json`).

**Exact pre-filter keyword list used** (case-insensitive substring match against raw chunk text, distinct from and broader than `concepts.py`'s narrow `LITERATURE_CONCEPT_KEYWORDS`): `somatotop, homuncul, topograph, hemispher, ipsilateral, contralateral, lateraliz, lateral, asymmetr, bilateral, unilateral, limb, digit, finger, toe, hand, foot, feet, tongue, orofacial, articulat, effector, gradient, selectiv`. An earlier, even-broader attempt (adding `cortex, cortical, motor, premotor, sensorimotor, represent, activat, movement, motion`) admitted 85.2% of chunks — useless as a filter, since those terms appear in nearly every sentence of any fMRI paper; dropped before the real run.

Result: 58/432 chunks (13.4%) produced a consistent (>=2/3 repeats) claim mapping to a known concept — real per-concept literature-support counts: `right_side`/`left_side`=49 chunk-hits each, `tongue`=30, `hand`=28, `foot`=9 (thinnest support in this corpus). 13 unique unmapped "discovery" phrases surfaced; most are irrelevant noise from the Yeo et al. 2011 resting-state paper, but several from Meier et al. 2008 and Ehrsson et al. 2003 make a real, citable point: motor cortex somatotopy is "blurred, overlapping," with a "core and surround organization," not a clean discrete map — a legitimate reason CAV directions might show partial rather than perfectly clean separation, if ever asked.

### 11.5 Fresh resume draft (2026-08-22, IN PROGRESS)

Purpose-first summary line, correcting an earlier draft that listed 4 system components without saying why they exist together (user feedback: "the purpose is not coming out clearly... it reads like there are four parts, but what are they doing together?").

**Summary**: *Built a system to test whether a brain-decoding model is right for the right reasons — that its accuracy reflects concepts a human would actually recognize, like which body part moved or which side, rather than an unverifiable shortcut — training three different representation-learning approaches on the same task, validating every result across repeated resamples, and checking findings against independent neuroscience literature.*

**Bullets, 1-5 all provisionally accepted this pass** (1-4 map to the summary's 4 clauses; 5 is a genuine additional contribution from this week's extraction-sweep work — open question, not yet resolved: does the summary need a small edit to gesture at it, or does it stand as a bonus beyond the 4-clause summary, matching how BGM ended up with 5 bullets under its own summary):

1. Isolated the effect of training objective from architecture by parameter-matching two backbones (GRU and Transformer, tuned to within 1.15x parameter count) and training each on three deliberately different objectives — a supervised multi-task decoder, a supervised-contrastive aligner, and a self-supervised contrastive co-embedding — so that any difference in how interpretable the resulting representation turned out to be could be attributed to the training objective, not an incidental architecture advantage.
2. Tested every paradigm's representation against 8 human-recognizable concepts — which body part, which side, movement versus rest — and confirmed all three pass equally well, a result that held up only after catching and correcting a flaw in one paradigm's own test that had initially made it look far weaker than it actually was.
3. Validated every result across 30 independently retrained subject splits per paradigm rather than a single train/test division, using paired significance tests on those same splits to surface a genuine, non-obvious pattern: which architecture wins depends on the training paradigm, not a fixed ranking — Transformer ahead under both label-driven objectives, GRU ahead under the self-supervised one, every comparison significant (p<0.001).
4. Checked whether the tested concepts were more than an internally-chosen list by building a literature-verification loop that retrieves relevant neuroscience papers and computes each verdict deterministically in code from the excerpt's stance and the model's own measured concept-sensitivity score — never letting the language model decide the verdict itself, a design forced by a measured failure rather than a hypothetical one: asked to judge agreement freely, the same model defaulted to agreeing regardless of the actual score in 10 of 12 real cases.
5. Extended the verification loop from checking literature about one decode at a time to mining the entire corpus independently of any prediction, using a keyword pre-filter deliberately broader than the known concept list — preserving 136 chunks a narrower filter would have missed — and requiring each claim to recur across 3 independent extraction passes before trusting it, surfacing real per-concept literature-support counts and confirming the tested concepts were grounded in actual research, not just internally assumed.

## 9. Resume points — draft v1 (2026-08-18, to revisit after the BGM thesis pass)

Restructured from the existing 7 disconnected bullets into one throughline, per the "reads like ATS keyword stuffing" complaint. Uses current verified numbers (91.8%, not the resume's existing 90.8% — replace wholesale, don't merge).

**Framing line** (subtitle under project title, or fold into bullet 1): *Testing which neural representation-learning paradigms produce human-interpretable brain decoders, grounded against independent scientific literature rather than assuming accuracy implies understanding.*

**Final v1, XYZ format** (Accomplished X, as measured by Y, by doing Z — result first, not method first):

1. Validated that three fundamentally different training objectives converge on comparable decoding accuracy, as measured by macro-F1 within 1 point across all three (91.6-92.0%) on a common 100-subject fMRI task, by designing and training matched supervised, supervised-contrastive, and self-supervised (label-free) representation-learning paradigms on identical architecture.
2. Demonstrated a reproducible effector-identity vs. body-laterality representational asymmetry, as measured by near-ceiling TCAV concept sensitivity for effector concepts against markedly weaker sensitivity for laterality concepts, by building a 4-method attribution + CAV/TCAV interpretability suite and independently deriving concept directions three separate ways.
3. Established that a Transformer architecture significantly outperforms a recurrent baseline, as measured by paired Wilcoxon signed-rank p<10⁻⁸ across repeated subject-level resamples, by applying population-level statistical validation matching an established peer-reviewed neuroimaging methodology.
4. Eliminated a measured LLM sycophancy failure, as measured by moving from 10/12 free-form judgments defaulting to agreement regardless of evidence to a fully evidence-grounded verdict, by redesigning the retrieval pipeline to compute the evidentiary decision deterministically and restricting the LLM to grounded narration.
5. Improved domain-specific retrieval accuracy by 17 points and diagnosed the real limits of fine-tuning, as measured by top-1 chunk-retrieval accuracy (43.9%->61.0%) and a fine-tuned model fixing output-format compliance (0/8->8/8) without fixing underlying reasoning, by contrastively fine-tuning the retrieval embedding model and running a targeted fine-tuning experiment on the generation model's failure mode.

**Not in the bullets, but keep verbally ready**: Case 3's number is single-split, not yet repeated-split-CI-backed like the other two. Bullets are draft-length, not resume-line-length — trimming is a later pass once seen against actual page real estate.

## 10. Still to build out (placeholders — continue block by block)
- [ ] Statistical framework block (repeated splits, bootstrap CIs, paired Wilcoxon — why this design, not simpler alternatives)
- [ ] The Case 3 "did we do justice to the TCAV comparison" self-critique, rehearsed out loud
- [ ] Likely extension questions: SC-graph/GNN input, generative HRF→brain decoder — framed as design-level answers backed by real GNN/graph-spectral and Bayesian-generative-modeling background, not as things to build in the remaining days
- [ ] Resume-number reconciliation (90.8%/91.8% Case 2 macro-F1 discrepancy — decide which number to say)
