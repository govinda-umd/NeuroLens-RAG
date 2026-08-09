# Interpretability Methods for RSN Inference — Notes

> Documents what v1 implemented and why, an empirical finding worth tracking, a survey of methods **not** implemented (for future reference), and Been Kim's concept-based interpretability line, which is a distinct and arguably more promising direction than feature/region attribution. Companion to [`06_interpretability_rsn.ipynb`](../notebooks/06_interpretability_rsn.ipynb) and [decoded-state-to-text-report.md](decoded-state-to-text-report.md).

## 1. What v1 implemented, and why

Four methods, chosen because they span the two fundamentally different ways to attribute a neural network's decision, and because all four are well-established (not bespoke):

| Method | Family | Library | Operates on |
|---|---|---|---|
| Saliency | gradient-based | Captum | continuous 300-ROI input directly |
| Integrated Gradients | gradient-based | Captum | continuous 300-ROI input directly |
| Exact Shapley values | perturbation/coalition-based | hand-implemented | 7-network coalition space |
| LIME | perturbation/coalition-based | `lime` | 7-network coalition space |

**Why pool to 7 Yeo networks instead of 300 ROIs or 32×300 (timestep, ROI) cells?** Gradient-based methods (Saliency, IG) work natively on the continuous input and are aggregated to network level only for reporting (sum |attribution| across the window's 32 timesteps, then across each network's ROIs). But perturbation-based methods scale with the number of "players" being ablated: exact Shapley values require enumerating all `2^n` coalitions, and LIME needs enough samples to fit a stable local linear surrogate. At `n=300` ROIs, neither is tractable without approximation (KernelSHAP sampling, LIME on a huge feature space). At `n=7` networks, `2^7=128` coalitions can be enumerated **exactly** — no sampling approximation needed for Shapley — and LIME's surrogate only has to fit 7 features. This is a deliberate resolution trade-off: you get exact, fast, network-level attribution instead of approximate, slow, ROI-level attribution. Revisit ROI-level (approximate) Shapley/LIME later if network-level granularity turns out to be too coarse.

## 2. An empirical finding worth tracking

Running all four methods on every test-set window (`06_interpretability_rsn.ipynb`) surfaced a **two-cluster disagreement pattern**, not one consensus. Measured twice — at 5-subject and 20-subject scale — and the pattern persists, though it softens as the model gets better:

| | 5 subjects (v1) | 20 subjects (v2) |
|---|---|---|
| Saliency ↔ IG top-1 agreement | 90.6% | 97.0% |
| Shapley ↔ LIME top-1 agreement | 94.9% | 97.0% |
| **Cross-family top-1 agreement** | **~65%** | **~78%** |
| Cross-family rank correlation | ~0.39 | ~0.43–0.47 |
| Mean Default-network attribution (gradient methods) | ~21% | ~18% |
| Mean Default-network attribution (perturbation methods) | ~10% | ~8% |

The disagreement has a specific, stable shape: gradient-based methods consistently put more attribution on the **Default** network than perturbation-based methods do, while perturbation-based methods concentrate more on **SomMot**. For MOTOR-task decoding, SomMot dominance is the neuroanatomically expected answer — the perturbation-based methods' result is the more literature-plausible one here. The gradient methods' extra Default-network mass could be a genuine finding (auxiliary HRF regression pulling in DMN-related temporal structure?) or a known gradient-method artifact — vanilla gradients and even path-integrated gradients can reflect local loss-landscape curvature near the input/baseline rather than true causal importance, especially for inputs (like z-scored fMRI) where "zero" isn't a semantically neutral baseline the way black pixels are for images. **Cross-family agreement improving with more data and a more confident model is itself informative** — it suggests at least part of the disagreement was driven by the 5-subject model's genuine uncertainty (wrong or low-confidence predictions have less "true" attribution structure for either method family to recover), rather than purely a fixed methodological gap between the two families. **Still not resolved — flagged as an open question before trusting either family's attribution unquestioningly in a downstream RAG query.**

## 2.1 GRU vs. Transformer: do they represent concepts equally meaningfully?

Raw decoding accuracy (macro-F1) says whether a model gets the label right, not whether its internal representation is organized the way a human would expect. Running the same TCAV analysis (§4 label-derived concepts) on **both** architectures, trained on the official 100-subject split, tests that directly — a genuinely different question from "which model scores higher."

**Result**: both architectures show the same clean, near-perfect effector-concept structure (`hand`, `foot`, `tongue`: TCAV 1.0 for their own classes, 0.0 elsewhere, probe accuracy 0.985–0.999 for both) — real evidence that *both* models organize hand/foot/tongue movement information along genuinely separable, concept-aligned directions, not just an artifact of one architecture. Transformer's probe accuracy is consistently slightly higher across all 5 concepts (0.996–0.999 vs. GRU's 0.985–0.997) — a small but consistent edge in how linearly separable these concepts are in its representation, independent of the raw decoding-accuracy comparison.

**A genuinely useful accidental finding**: the laterality concepts' extrapolation artifact (§4 — a CAV fit only on lateralized classes extrapolating a direction onto `baseline`/`tongue`, which were never in its training set) **flipped polarity between the 20-subject and 100-subject runs**. At 20 subjects, `baseline`/`tongue` scored 1.0 on `right_side`; at 100 subjects, both architectures score them 1.0 on `left_side` instead — the exact opposite direction. A real neuroscience effect should be stable in direction across independent training runs on different data; a coin-flip-like polarity switch is strong additional evidence (not just the original suspicion) that this specific result is a linear-probe extrapolation artifact, not a genuine finding about how these models represent baseline/tongue laterality. Worth remembering as a concrete example of why out-of-training-distribution CAV scores need this kind of stability check before being trusted.

## 3. Methods surveyed but not implemented (candidates for later)

- **SmoothGrad** (Smilkov et al. 2017) — averages Saliency over many noise-perturbed copies of the input; mitigates the well-known noisiness of raw gradient maps. Cheap add-on to the existing Saliency implementation.
- **DeepLIFT** (Shrikumar et al. 2017) — propagates *differences* in activation from a reference input through the network via modified backpropagation rules; closely related to Integrated Gradients but avoids IG's need to numerically integrate along a path.
- **GradientSHAP** — combines ideas from SmoothGrad, IG, and Shapley values into a single gradient-based estimator of Shapley-like attributions; available in Captum, would sit as a bridge between our two method families.
- **Layer-wise Relevance Propagation (LRP)** — propagates a "relevance" score backward through the network layer by layer via conservation rules, rather than through raw gradients.
- **Attention-weight inspection** — the Transformer already computes real self-attention weights over the 32-timestep window; in principle these could be read out directly as a measure of "which timepoints mattered." **Caveat worth flagging explicitly**: attention weights as explanations are contested in the literature — Jain & Wallace (2019), "Attention is not Explanation," showed attention weights often don't correlate with gradient-based importance and can be adversarially manipulated while preserving output; Wiegreffe & Pinter (2019), "Attention is not not Explanation," pushed back, arguing attention can still be meaningful under the right testing protocol. Don't treat raw attention weights as a free fifth interpretability method without accounting for this debate.
- **TimeSHAP** (Bento et al. 2021) and related window/event-based SHAP variants — extend Shapley-value attribution to sequential and recurrent models, explicitly attributing importance to *time steps* (and their interactions with features), not just to features. More directly suited to "which timepoint within the window mattered" than our current network-only pooling, which collapses the time axis for the perturbation-based methods.
- **Counterfactual explanations** — search for the minimal perturbation to the input (spatially, temporally, or both) that would flip the decoded class; complements attribution-style methods by answering "what would need to change," not just "what mattered."
- **Model distillation / surrogate fidelity** — fit a simple interpretable model (e.g. logistic regression on network-averaged ROI features) to approximate the Transformer's decisions, and measure how well it tracks the real model (fidelity) as an independent sanity check on any attribution method's plausibility.
- **Finer-than-network perturbation attribution** — approximate KernelSHAP or LIME at ROI (300) or (timestep, ROI) (32×300) resolution, once network-level granularity proves too coarse for a given question; would need sampling rather than exact enumeration.

## 4. Been Kim's concept-based interpretability line

A meaningfully different research direction from everything above, worth tracking separately because it targets a specific critique: raw feature/ROI/network attribution answers "which input dimensions mattered," but humans reason in terms of higher-level **concepts**, not raw features — a clinician doesn't think in voxels, and (by analogy) a neuroscientist doesn't natively think in terms of "ROI #147," but in terms of named functional patterns, networks, or cognitive constructs. This is the throughline of Been Kim's work (Google DeepMind / prior Google Brain):

- **TCAV — Testing with Concept Activation Vectors** (Kim et al., 2018) — define a concept (e.g., "striped," in the original image-classification examples) via a small set of positive/negative example inputs, train a linear probe in one of the model's hidden layers to separate them, and use the resulting direction (the Concept Activation Vector) to test how sensitive a given prediction is to that concept — without needing the concept to be a literal input feature.
- **ACE — Automatic Concept-based Explanations** (Ghorbani, Wexler, Zou, Kim, 2019) — removes the need for hand-labeled concept examples by automatically discovering candidate concepts via clustering of internal representations (e.g. segments of training examples), then scoring their importance with TCAV.
- **ConceptSHAP** (Yeh et al., 2020) — applies Shapley-value machinery to *discovered concepts* rather than raw features, giving a principled completeness/importance score per concept.
- **Concept Bottleneck Models** (Koh, Nguyen, Tang, Mussmann, Pierson, Kim, Liang, 2020) — architects the model itself to predict a set of human-interpretable concepts as an explicit intermediate bottleneck *before* the final prediction, so concepts are causally load-bearing for the model's decision rather than a post-hoc probe on an already-trained black box.

**Why this matters for NeuroLens-RAG specifically**: instead of (or in addition to) "SomMot network, 40% attribution," a concept-based approach could let you define concepts as known functional signatures — e.g. "bilateral motor coordination," "unimanual vs. bimanual movement," or literature-derived co-activation patterns spanning multiple networks — extract CAVs from the Transformer's pooled 128-dim representation (the natural probe point, right before the classification/HRF heads), and test whether a given decode is sensitive to those concept directions. That would produce a semantically richer, more literature-friendly RAG query than a raw network-percentage breakdown. **A pragmatic v1 of exactly this is now implemented** — see §4.1 — using literature-derived (not hand-labeled or auto-discovered) concepts, constrained to the ones expressible via existing task labels.

### 4.1 A NeuroLens-RAG-specific variant: literature-derived concept hypotheses

TCAV, ACE, and Concept Bottleneck Models all need concepts to come from *somewhere* — either a human hand-labels them, or they're auto-discovered by clustering the model's own internal representations. NeuroLens-RAG has a third option neither of those does: **use the RAG retrieval step itself as the concept-hypothesis generator.**

Once Level 2 retrieval exists (`src/neurolens/pipeline.py`), every decoded window already produces a set of retrieved literature excerpts labeled supports/contradicts/unrelated (per the structured prompt in `build_llm_prompt`). Those supporting/contradicting excerpts are themselves candidate concept descriptions written by domain experts (the papers' authors) — e.g. an excerpt discussing "bilateral coordination during unimanual tasks" directly hands you a concept phrase to formalize as a CAV, rather than requiring a human to guess useful concepts up front or hoping unsupervised clustering finds something semantically nameable. Concretely, the loop would be:

```
decoded window → RSN attribution → RAG retrieves literature
        → LLM extracts candidate concept phrases from supporting/contradicting excerpts
        → each phrase becomes a hand-labeled-by-proxy concept (build a small positive/negative
          example set for it, e.g. via further retrieval or existing task labels)
        → train a CAV for that concept in the Transformer's pooled representation
        → test decode sensitivity to the concept direction
        → feed the CAV result back into the generated text or the next retrieval round
```

This turns the literature into a source of testable hypotheses about *what the model might be representing*, rather than only a source of text to compare the decode against — the RAG and interpretability halves of the pipeline stop being sequential stages and become a loop.

**Both stages are now implemented.** The TCAV mechanism itself was validated first on label-derived concepts (`src/neurolens/concepts.py`, [`09_concept_vectors.ipynb`](../notebooks/09_concept_vectors.ipynb)) — concepts built from the true movement labels already on disk (`hand`, `foot`, `tongue`, `right_side`, `left_side`). Effector concepts showed clean, exactly-as-expected TCAV scores (1.0 for their own classes, 0.0 elsewhere, >0.99 probe accuracy); laterality concepts surfaced a real extrapolation limitation (§2.1) that turned out to be reproducible evidence of a genuine artifact, not a one-off.

**The full literature-derived loop is now built**: `concepts.py::map_phrase_to_known_concept` + `explain_literature_concept`, orchestrated end-to-end by `pipeline.py::explain_decoded_window_with_cav_loop`, demonstrated on real examples in [`14_rag_cav_loop.ipynb`](../notebooks/14_rag_cav_loop.ipynb). This answers the previously-open question ("how to turn a retrieved concept phrase into a labeled example set without heavy manual curation") pragmatically rather than in full generality: an LLM extracts a candidate concept phrase from a retrieved excerpt, and the phrase is mapped via keyword matching onto one of the concepts we can *already* build labeled examples for from existing task labels — not a solution for arbitrary literature claims, but a real, working v1 for the claims that happen to be about effector or laterality, which is exactly what this corpus's motor-cognition papers (Ehrsson et al. 2003, Meier et al. 2008) discuss.

**Real results, three examples**: a `left_hand` decode showed the textbook-correct convergent pattern — TCAV=1.0 for both `hand` and `left_side` (matching Ehrsson's contralateral-representation claim), TCAV=0.0 for `right_side` and `tongue`. A `baseline` decode correctly showed TCAV=0.0 for an extracted hand-related concept, sensibly uninvolved. **A genuine, documented limitation surfaced on the third example**: for a `left_foot` decode, the loop extracted and tested a `hand`-related phrase (because that's what a retrieved excerpt happened to discuss, not because it was anatomically relevant to a foot decode) and got TCAV=0.0 — a correct, expected null result (a foot decode shouldn't depend on hand-specific directions) — but the LLM's final synthesis mischaracterized this as "a discrepancy between the literature's claim and the decoded result," conflating "this concept doesn't apply to this trial type" with "the model disagrees with the literature." **This is a real v1 gap, not swept under the rug**: the loop currently tests whatever concepts get extracted from retrieved excerpts without checking whether they're anatomically relevant to the *actually decoded* class, and the synthesis prompt doesn't explicitly guard against this specific confusion. A v2 fix would either filter tested concepts to ones plausible for the decoded class, or make the synthesis prompt explicitly distinguish "irrelevant to this trial" from "contradicts the literature."

## 5. Parking lot: Level-3 brain-representation → LLM-embedding integration

Noted for a dedicated future discussion, not analyzed here: directly projecting the Transformer's learned brain representation (e.g. its pooled 128-dim hidden state) into a pretrained LLM's embedding space via a learned adapter, so the LLM can generate text conditioned on brain activity end-to-end, rather than going through the template → retrieval → grounded-generation pipeline (Levels 0–2) described in [decoded-state-to-text-report.md](decoded-state-to-text-report.md). That report's §3(b) already surveys the closest field precedents (Tang et al. 2023; Horikawa 2025 "Mind Captioning") — revisit this note once ready to scope out what NeuroLens-RAG's own version would need.

## References

- [Axiomatic Attribution for Deep Networks (Integrated Gradients), Sundararajan et al. 2017](https://arxiv.org/abs/1703.01365)
- [SmoothGrad: removing noise by adding noise, Smilkov et al. 2017](https://arxiv.org/abs/1706.03825)
- [Learning Important Features Through Propagating Activation Differences (DeepLIFT), Shrikumar et al. 2017](https://arxiv.org/abs/1704.02685)
- [A Unified Approach to Interpreting Model Predictions (SHAP), Lundberg & Lee 2017](https://arxiv.org/abs/1705.07874)
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME), Ribeiro et al. 2016](https://arxiv.org/abs/1602.04938)
- [Attention is not Explanation, Jain & Wallace 2019](https://arxiv.org/abs/1902.10186)
- [Attention is not not Explanation, Wiegreffe & Pinter 2019](https://arxiv.org/abs/1908.04626)
- [TimeSHAP: Explaining Recurrent Models through Sequence Perturbations, Bento et al. 2021](https://arxiv.org/abs/2012.00073)
- [Interpretability Beyond Feature Attribution: TCAV, Kim et al. 2018](https://arxiv.org/abs/1711.11279)
- [Towards Automatic Concept-based Explanations (ACE), Ghorbani, Wexler, Zou, Kim 2019](https://arxiv.org/abs/1902.03129)
- [On Completeness-aware Concept-Based Explanations (ConceptSHAP), Yeh et al. 2020](https://arxiv.org/abs/1910.07969)
- [Concept Bottleneck Models, Koh et al. 2020](https://arxiv.org/abs/2007.04612)
- [Captum: Model Interpretability for PyTorch](https://captum.ai/)
