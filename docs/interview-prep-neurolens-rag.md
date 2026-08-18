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

## 7. Still to build out (placeholders — continue block by block)

- [ ] RAG + verification loop block (why the deterministic verdict, the sycophancy bug)
- [ ] Statistical framework block (repeated splits, bootstrap CIs, paired Wilcoxon — why this design, not simpler alternatives)
- [ ] The Case 3 "did we do justice to the TCAV comparison" self-critique, rehearsed out loud
- [ ] Likely extension questions: SC-graph/GNN input, generative HRF→brain decoder — framed as design-level answers backed by real GNN/graph-spectral and Bayesian-generative-modeling background, not as things to build in the remaining days
- [ ] Resume-number reconciliation (90.8%/91.8% Case 2 macro-F1 discrepancy — decide which number to say)
