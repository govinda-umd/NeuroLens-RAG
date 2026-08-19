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

## 9. Still to build out (placeholders — continue block by block)
- [ ] Statistical framework block (repeated splits, bootstrap CIs, paired Wilcoxon — why this design, not simpler alternatives)
- [ ] The Case 3 "did we do justice to the TCAV comparison" self-critique, rehearsed out loud
- [ ] Likely extension questions: SC-graph/GNN input, generative HRF→brain decoder — framed as design-level answers backed by real GNN/graph-spectral and Bayesian-generative-modeling background, not as things to build in the remaining days
- [ ] Resume-number reconciliation (90.8%/91.8% Case 2 macro-F1 discrepancy — decide which number to say)
