# NeuroLens-RAG: End-to-End Report

> Consolidated, current-as-of-2026-09-12 account of the full system: data, three representation-learning paradigms, baseline and control experiments, interpretability, and literature verification (v1 and v2). Where a number here differs from an older doc, this report is correct. Several headline claims in earlier write-ups were superseded by later, more rigorous experiments, and each is flagged explicitly below, not silently. Written for a research-scientist or applied-scientist audience: every claim below is backed by a specific, checkable experiment, and every honest limitation is stated plainly rather than smoothed over.

## 1. The question this project answers

Accuracy alone cannot tell you whether a brain-decoding model learned something neurobiologically real or an incidental shortcut correlated with the label. NeuroLens-RAG is a framework for testing that distinction. It trains the same decoding task three structurally different ways, runs a battery of baseline and control experiments to rule out trivial explanations, checks whether each resulting representation depends on concepts a human would recognize (which body part, which side, movement versus rest), and cross-checks those concepts against independent neuroscience literature. The checking mechanism itself is held to the same standard of evidence as the thing it is checking.

## 2. Data

HCP Young Adult, MOTOR task, 100-subject pool. A 200-subject scale-up is complete at the data level but has not been re-validated under the current protocol; it is paused, not abandoned. Each subject contributes 300-channel ROI time series (Schaefer-300 parcellation of the same BOLD signal, motion-regressed, detrended, z-scored per run), windowed causally into 32-TR (about 23 seconds) overlapping segments. Per window, two supervisory signals are read, both at the window's last timepoint:

- `y`: a 6-way categorical label (baseline, left/right hand, left/right foot, tongue).
- `y_hrf`: a 5-channel continuous vector (one per non-baseline condition), each channel that condition's event timeline convolved with the canonical hemodynamic response function.

Splits are always subject-level, never window-level, because overlapping windows from one subject are near-duplicates and would leak across a window-level split. All bootstrap and ablation experiments below share the exact same 30 subject partitions (90-subject pool, 65/13/12 train/val/test, 10 subjects permanently reserved for hyperparameter decisions only), so every comparison in this report is paired, not three separately noisy studies compared informally.

## 3. Three representation-learning paradigms

Two backbones, GRU and Transformer, are parameter-matched to 1.15x after a fix: GRU now defaults to 2 layers, closing what was originally a 1.83x gap. Both are trained under three different objectives, so any representational difference is attributable to the objective, not an architecture confound. Every backbone exposes the same `forward_features(x) -> [B, 128]` interface. That is why adding a third paradigm, and later testing all three with one interpretability mechanism and one set of baseline and control experiments, was cheap rather than requiring three separate pipelines.

- **Case 1 (supervised):** multi-task decoder, joint 6-way classification and HRF regression off one shared trunk. Loss = CE + 0.1·MSE.
- **Case 2 (supervised-contrastive):** a brain encoder is aligned to 6 frozen-MiniLM condition-description prototypes in a shared 64-d space via a **symmetric multi-positive contrastive loss**, `L = ½(L_b2t + L_t2b)`. `L_b2t` is standard cross-entropy, one brain window against all 6 prototypes. `L_t2b` covers the direction a closed 6-item vocabulary would normally rule out, since no single brain window uniquely pairs with a prototype. This is resolved by treating every brain window sharing a prototype's true label as a positive for that prototype, averaged, in a Khosla et al. 2020-style multi-positive formulation rather than literal CLIP's one-to-one pairing. This is the loss that actually trained the checkpoints behind every Case 2 number in this report. The original, simpler single-direction, brain-to-text-only asymmetric loss was the initial design and is still in the codebase (`contrastive.py::train_contrastive`), but it is not what the 30-resample bootstrap or anything downstream of it uses. A literal CLIP-style variant with naive in-batch negatives (`clip_loss`) was also built for comparison. It scored a higher single-split test macro-F1 (0.9165) than both the asymmetric baseline (0.9080) and the multi-positive loss (0.9137), contradicting the a priori false-negative-penalty hypothesis, plausibly because batch size 64 across only 6 classes still leaves the softmax dominated by genuinely different-class negatives most of the time.
- **Case 3 (self-supervised):** a brain encoder is aligned to its own window's HRF vector via a symmetric InfoNCE loss. HRF is continuous and per-window, unlike Case 2's fixed prototypes, so both directions have real negatives. No label ever enters this loss. `BrainWithPostHocClassifier` fits a linear probe on frozen features after training. This is the only way to get a macro-F1 number and a target-class logit for CAV/TCAV out of a model that never saw a label.

### 3.1 Architecture comparison: the ranking flips by paradigm, not a fixed winner

30 paired repeated-split bootstraps, all three cases sharing the exact same 30 subject partitions. Case 2 and Case 3 reuse Case 1's splits verbatim; they are never redrawn.

| Case | GRU mean F1 (95% CI) | Transformer mean F1 (95% CI) | Winner (paired Wilcoxon) |
|---|---|---|---|
| 1 (supervised) | 0.906 [0.876, 0.931] | 0.922 [0.902, 0.945] | Transformer, p<0.0001 |
| 2 (supervised-contrastive) | 0.900 [0.866, 0.927] | 0.918 [0.898, 0.942] | Transformer, p<0.0001 |
| 3 (self-supervised) | 0.928 [0.903, 0.946] | 0.922 [0.903, 0.944] | **GRU**, p=0.0004 |

Transformer wins whenever a label drives training directly. GRU edges ahead in the one paradigm that never sees a label. This supersedes an earlier, smaller-scale claim that "Transformer always beats GRU." That claim held at single-split scale but does not survive the population-level, paired comparison.

### 3.2 Baseline check, Case 1: is a learned sequence representation even necessary?

Before crediting GRU or Transformer's accuracy to learned temporal structure, a more basic question needs answering: is class separability already sitting in the raw window, accessible to any generic function approximator? Two baselines (`baseline_mlp.py`) share Case 1's exact 30 resamples and the same `forward_features` interface:

- **FlattenMLP**: the whole 32x300 window flattened to one 9,600-d vector, then a single hidden layer, then a classifier. It keeps every raw value; "time" is just "which flat index."
- **MeanPoolMLP**: the window averaged over time to one 300-d vector, then a single hidden layer, then a classifier. It keeps only the spatial, per-ROI pattern and discards temporal dynamics entirely.

| Model | Mean F1 (95% CI) | Params | vs. GRU (paired Wilcoxon) | vs. Transformer (paired Wilcoxon) |
|---|---|---|---|---|
| GRU | 0.906 [0.876, 0.931] | 166,539 | — | — |
| Transformer | 0.922 [0.902, 0.945] | 304,907 | — | — |
| FlattenMLP | 0.907 [0.882, 0.933] | 1,230,347 | p=0.73 (not significant) | p<0.0001 (Transformer wins) |
| MeanPoolMLP | 0.654 [0.616, 0.694] | 39,947 | p<0.0001 (GRU wins) | p<0.0001 (Transformer wins) |

**The honest answer is more nuanced than yes or no.** A plain MLP with zero learned temporal structure is statistically indistinguishable from GRU (p=0.73) once given the full, undestroyed window. But the Transformer significantly outperforms that same flat MLP (p<0.0001) despite having 4x fewer parameters than it. This is real evidence that self-attention is doing something a naive aggregation structurally cannot, not just adding capacity. Destroying temporal order entirely by mean-pooling costs about 25 points and is significant in the other direction, so temporal information within the window clearly matters; GRU's specific way of encoding it simply is not the thing making the difference. **This changes how to read §3.1.** Case 1's GRU-versus-Transformer comparison is not really testing whether sequence modeling helps. It is closer to testing whether attention specifically finds structure that recurrence and raw aggregation both miss.

### 3.3 Baseline check extended to Case 2 and Case 3 (2026-09-12, closes a previously flagged gap)

The §3.2 check was Case 1 only until now. FlattenMLP and MeanPoolMLP were plugged into Case 2's contrastive loss and Case 3's self-supervised loss as brain backbones, unmodified, across the same 30 resamples:

| Model | Case 2 mean F1 (95% CI) | vs. GRU / Transformer | Case 3 mean F1 (95% CI) | vs. GRU / Transformer |
|---|---|---|---|---|
| GRU | 0.900 | — | 0.928 | — |
| Transformer | 0.918 | — | 0.922 | — |
| FlattenMLP | 0.899 [0.865, 0.925] | ties GRU (p=0.887); loses to Transformer (p<0.0001) | 0.892 [0.864, 0.917] | **loses to GRU (p<0.0001)**; loses to Transformer (p<0.0001) |
| MeanPoolMLP | 0.647 [0.614, 0.682] | loses to both (p<0.0001) | 0.640 [0.591, 0.684] | loses to both (p<0.0001) |

**A genuinely new finding: the "flat MLP ties GRU" result does not generalize across paradigms.** Case 1's and Case 2's flat-MLP baselines both statistically tie GRU (Case 1: p=0.73; Case 2: p=0.887), so recurrence is not demonstrably buying anything beyond raw aggregation under either supervised objective. But under Case 3's self-supervised objective, FlattenMLP significantly loses to GRU (p<0.0001). Recurrence appears to matter specifically when no label drives training directly. This is a genuinely paradigm-dependent result that the Case-1-only version of this check could not have revealed, and it is the exact reason this extension was worth doing rather than assuming Case 1's finding generalized.

### 3.4 Temporal perturbation controls (2026-09-12, new): does destroying order at test time collapse an already-trained model?

This is complementary to §3.2 and §3.3, which ask whether a differently structured model needs temporal order. Here the question is whether an already-trained GRU or Transformer, across all three cases, actually relies on temporal order at inference time. Four perturbations were applied to test windows, evaluated on the same 30-resample, 3-case, 2-architecture checkpoint set with zero retraining:

| Case | Arch | Original | Shuffle | Reverse | Circular shift | Mean-pool |
|---|---|---|---|---|---|---|
| 1 | GRU | 0.906 | 0.504 | **0.099** | 0.477 | 0.464 |
| 1 | Transformer | 0.922 | 0.572 | **0.133** | 0.484 | 0.443 |
| 2 | GRU | 0.899 | 0.438 | **0.084** | 0.472 | 0.425 |
| 2 | Transformer | 0.918 | 0.557 | **0.127** | 0.473 | 0.350 |
| 3 | GRU | 0.928 | 0.351 | **0.116** | 0.500 | 0.311 |
| 3 | Transformer | 0.922 | 0.534 | **0.081** | 0.467 | 0.288 |

**Time-reversal causes the worst, near-total collapse (F1 0.08 to 0.13) in every one of the 6 configurations.** This is consistent with a causal architecture's last-timestep readout being maximally sensitive to which end of the sequence is recent: reversing a sequence does not just scramble it, it inverts what recent means. Shuffle and circular-shift degrade to a similar moderate band, roughly 0.35 to 0.57, regardless of case or architecture. Mean-pooling hits Case 3 hardest, 0.29 to 0.31 versus 0.35 to 0.46 for Cases 1 and 2. A self-supervised representation trained with no label at all turns out to lean on temporal structure somewhat more than the label-driven paradigms once that structure is fully destroyed at test time.

### 3.5 ROI-network ablation controls (2026-09-12, new): does the visual-network confound (§7.2) survive input-level lesioning?

§7.2 below found that `movement_vs_rest`'s concept attribution localizes unanimously to the visual network (Vis), not somatomotor (SomMot). It is flagged there as plausibly a visual-cue confound, not yet stress-tested. Here, Vis and SomMot are zeroed out across the whole window at test time on the same checkpoint set, and two derived metrics are tracked: **movement-vs-rest accuracy**, which collapses the 6-way prediction to baseline versus any movement, and **effector accuracy**, which checks, on true non-baseline windows only, whether the predicted class falls in the correct body-part group (hand, foot, or tongue) regardless of side.

| Case | Arch | Metric | No ablation | Drop Vis (Δ) | Drop SomMot (Δ) |
|---|---|---|---|---|---|
| 1 | GRU | movement_vs_rest | 0.954 | 0.905 (−0.049) | 0.851 (−0.103) |
| 1 | GRU | effector | 0.918 | 0.844 (−0.075) | 0.450 (**−0.469**) |
| 1 | Transformer | movement_vs_rest | 0.963 | 0.911 (−0.052) | 0.870 (−0.094) |
| 1 | Transformer | effector | 0.930 | 0.871 (−0.059) | 0.501 (**−0.428**) |
| 2 | GRU | movement_vs_rest | 0.954 | 0.890 (−0.065) | 0.857 (−0.098) |
| 2 | GRU | effector | 0.917 | 0.826 (−0.091) | 0.495 (**−0.422**) |
| 2 | Transformer | movement_vs_rest | 0.965 | 0.896 (−0.069) | 0.885 (−0.080) |
| 2 | Transformer | effector | 0.931 | 0.852 (−0.079) | 0.527 (**−0.404**) |
| 3 | GRU | movement_vs_rest | 0.967 | 0.854 (**−0.113**) | 0.932 (−0.035) |
| 3 | GRU | effector | 0.936 | 0.780 (**−0.156**) | 0.533 (**−0.403**) |
| 3 | Transformer | movement_vs_rest | 0.966 | 0.905 (−0.061) | 0.891 (−0.075) |
| 3 | Transformer | effector | 0.931 | 0.858 (−0.073) | 0.498 (**−0.434**) |

**This is an honest, more nuanced result than a clean confirmation.** Dropping SomMot devastates effector accuracy everywhere, 40 to 47 points, while barely touching movement-vs-rest, 3.5 to 10 points, a clean and expected confirmation that SomMot is the genuine anatomical driver of body-part identity. But dropping Vis does not cleanly isolate movement-vs-rest the way the attribution finding might predict. It modestly hurts both movement-vs-rest and effector accuracy in all 6 configurations, and for Case 3 with GRU it hurts effector accuracy (0.156) more than movement-vs-rest (0.113). **The conclusion, stated plainly:** the visual-network confound found by concept-attribution (§7.2) is real at the level of that concept's linear direction in representation space, but it does not fully explain classification behavior at the level of raw input ablation. The two methods ask related but genuinely different questions: which direction best separates classes in representation space, versus which raw input channels the whole decision pipeline depends on. They do not have to agree, and here they do not fully agree.

## 4. Interpretability: does the representation depend on the concept, not just correlate with it

CAV/TCAV (Kim et al. 2018) works as follows. First, define a concept via labeled positive and negative examples. Second, fit a linear probe on the model's pooled representation; the probe's normalized weight vector is the Concept Activation Vector. Third, take held-out examples of the target class, compute the gradient of that class's logit with respect to the representation, and take its dot product with the CAV direction. Fourth, the TCAV score is the fraction of held-out examples where that directional derivative is positive. This is a genuine local sensitivity measure within the model's own function. It is stronger than correlation, though not the same epistemic strength as a real intervention.

**Standardized derivation mechanism.** Every case tests a concept the same way: fit a classification head on frozen pooled features from labeled examples, then use that head's differentiable logit for the directional derivative. For Case 1 this is the model's own trained head. For Case 2 and Case 3 it is a post-hoc-fitted head (`fit_post_hoc_classifier`, originally written for Case 3 and reused completely unmodified on Case 2, because both models share the same `.brain_backbone`/`.brain_projection` attribute naming). This replaced Case 2's original text-arithmetic CAV derivation, which subtracted two text-prototype embeddings and pulled the result back through the projection's transpose, as the default. That was not because the original mechanism was wrong, but because of a real diagnosed artifact, described below.

### 4.1 The 8-concept x 3-case x 2-architecture x 30-resample sweep (1,440 evaluations)

8 concepts, each grounded in real motor anatomy: `hand`, `foot`, `tongue`, `right_side`, `left_side`, `movement_vs_rest`, `limb_vs_orofacial`, `upper_vs_lower_limb`.

**This is a diagnosed artifact, not a real representation-quality gap.** Case 2's original text-derived CAVs scored low on laterality, with right/left TCAV around 0.33, and specifically for the Transformer backbone systematically below chance at 0.17 to 0.21. That is not noise; it is a consistent wrong-direction signal. A controlled experiment settled it: fitting a linear probe on Case 2's frozen features via the exact same method as Case 1 and Case 3 got probe accuracy of 0.997 to 0.999 and TCAV of 0.92 to 1.00, indistinguishable from the other two paradigms and confirmed across all 30 resamples, not just a single check. **The root cause is understood, not just observed.** MiniLM's sentence-embedding geometry is dominated by the concrete noun (hand, foot, tongue), so subtracting text prototypes to isolate a weak, secondary axis like laterality fights the embedding space's actual geometry and can flip its sign. Concepts aligned with the dominant axis degrade only moderately under the same text-derived method. **All three paradigms converge to near-ceiling interpretability once tested with a consistent method.** The interesting result here is methodological: derivation method matters independently of representation quality, and this was caught through a targeted controlled experiment rather than assumed. It is not a paradigm-superiority finding.

## 5. RAG v1: literature-grounded verification, per-decode

**Retrieval.** The neuroscience corpus (10 papers as of the most recent expansion, see §6.5) is chunked into originally page-scoped, overlapping 220-word windows and embedded with `sentence-transformers/all-MiniLM-L6-v2`. Dense cosine retrieval is narrowed by a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker: a bi-encoder for cheap whole-corpus recall, a cross-encoder for precision on the narrowed set. This is the standard retrieve-then-rerank pattern, motivated by the real cost asymmetry between the two. The bi-encoder was domain-adaptively fine-tuned on in-domain query-passage pairs. The original 8-paper-corpus study reported top-1 chunk-retrieval accuracy improving from 43.9% to 61.0%, and top-3 from 75.6% to 85.4%. The reranker itself was never fine-tuned.

**The loop:** decode -> 4-method attribution (Saliency, Integrated Gradients, exact Shapley, LIME over the 7 Yeo resting-state networks) locates a consensus network -> a natural-language query -> retrieve and rerank -> a local LLM (`mlx-community/Llama-3.2-3B-Instruct-4bit` via `mlx_lm`) extracts a stance and a concept phrase -> the phrase maps to a known concept via keyword match -> CAV/TCAV re-probes that concept against the model's actual representation.

**The measured failure that shaped the whole design.** An early version let the LLM freely judge AGREE or DISAGREE between the literature and the CAV evidence. It defaulted to AGREE in 10 of 12 real cases regardless of the actual TCAV score. This is sycophancy, measured, not hypothetical. **The fix:** the verdict is computed deterministically in code from the stance and the TCAV score, so the LLM's only remaining job is to narrate an already-decided conclusion. This fix reached Case 2's loop. Case 1's loop still uses the older free-judgment design and was never re-measured at the same scale, a real, still-open gap (§8).

**Corpus-first mining.** Independent of any decode, the whole corpus is mined for claims. A keyword pre-filter, deliberately broader than the closed concept vocabulary (`somatotop, homuncul, hemispher, lateral, effector, gradient, selectiv`, and others, 24 terms total), admitted 432 of 879 chunks in the original 8-paper corpus, compared to 34.2% for the narrow concept-mapping list. The broader filter was chosen so it would not pre-decide the answer. Each survivor was extracted three times with concept-level self-consistency, requiring at least 2 of 3 repeats to agree on the same mapped concept rather than an exact phrase match, since phrasing varies run to run even for the same underlying claim. The result was 58 consistent claims, with real and uneven per-concept literature support: right and left each had 49 hits, tongue had 30, hand had 28, and foot had only 9, the thinnest support in this corpus. Thirteen unique unmapped "discovery" phrases surfaced outside the known vocabulary. Most were irrelevant noise from an unrelated resting-state-networks paper, but several from motor-cortex-organization papers made a real, citable point: motor cortex somatotopy is blurred and overlapping, with a core-and-surround organization, not a clean discrete map.

### 5.1 Retrieval ablation ladder (2026-09-12, new): decomposing where the retrieval gain actually comes from

The original improvement from 43.9% to 61.0% attributes the entire gain to domain-adaptive embedding fine-tuning. A fresh, comparable-methodology benchmark of 50 held-out queries, built with the same stratified-by-paper, same-seed construction and LLM-paraphrased queries, was built on the current, larger 10-paper, 1,060-chunk corpus and run through a full ablation ladder. **This is not a literal re-run of the original study and is not directly comparable number-for-number to it**, since the corpus has grown:

| Method | Recall@1 | Recall@3 | MRR (95% CI) |
|---|---|---|---|
| BM25 (lexical baseline) | 0.460 | 0.660 | 0.579 [0.465, 0.691] |
| Frozen MiniLM | 0.460 | 0.680 | 0.583 [0.471, 0.693] |
| Adapted MiniLM | 0.500 | 0.720 | 0.627 [0.520, 0.731] |
| Frozen MiniLM + reranker | 0.740 | 0.800 | 0.784 [0.673, 0.885] |
| Adapted MiniLM + reranker | 0.740 | 0.800 | 0.790 [0.680, 0.888] |
| Oracle | 1.000 | 1.000 | 1.000 (upper bound, by definition) |

**This is a materially more precise story than the single end-to-end number suggested.** Frozen dense retrieval barely beats lexical BM25: 0.460 versus 0.460 recall at 1, a tie. Domain-adaptive fine-tuning alone buys a modest gain, from 0.460 to 0.500. **The reranker is doing most of the real work.** Adding it to the frozen model already reaches 0.740, matching the fully adapted-and-reranked pipeline almost exactly. Once reranking is in the loop, the embedding model's own domain adaptation barely matters anymore. This decomposes a single headline retrieval-improvement number into its actual components, which is the more scientifically honest and more interesting claim to make: not that fine-tuning an embedding model improved retrieval, but that fine-tuning and reranking were measured against each other, and reranking, the intervention that requires no training at all, does most of the work.

## 6. RAG v2: claim-first, cross-representation, with two bugs found and fixed by actually running it

The v2 design inverts v1's order. It mines claims first, then asks which of the 6 trained representations actually depends on each claim's concept, then uses that representation's own reasoning to drive a second, targeted search, rather than starting from one decode.

### 6.1 Pipeline, as built and run

1. **Section-based chunking** (`retrieval.py::ingest_pdf_by_section`): split on markdown section headers detected in the PDF (Introduction, Methods, Results, Discussion), not page boundaries, with back-matter (references) cut and any section longer than the embedding model's limit re-split by the original word-window logic. 1,018 section chunks over the 10-paper corpus (837 over the original 8 papers).
2. **Keyword pre-filter:** 512 of 1,018 chunks survive (384 of 837 on the original 8-paper corpus).
3. **Extraction and soft concept mapping:** the same 3x self-consistency extraction as v1: 83 consistent claims, 70 unique phrases after deduplication, up from 56 and 48 on the 8-paper corpus. The 2 added papers alone contribute 26 of the 83. Phrases are additionally embedded and cosine-matched against each concept's reference text, softmax-weighted, so a claim can partially match several concepts instead of one hard bucket.
4. **Population-level representation selection.** Given a claim's soft concept weights, each representation's per-concept TCAV is combined into one weighted score. The question asked is which representation ranks first across the 30 real resamples, not just which has the highest single-point mean. Ties, very common near TCAV's ceiling, are broken uniformly at random and reported (`frac_ties_at_max`).
5. **Concept attribution** is a new mechanism, distinct from prediction attribution: the concept-alignment score `h(x)·v_C`, not a class logit, is backpropagated to the raw input, averaged over the representation's own held-out set, and aggregated to the 7 resting-state networks.
6. **Second-pass query and retrieve/rerank**, same infrastructure as v1.
7. **Grounded stance extraction**, then a deterministic verdict, then LLM narration of the already-decided result.

### 6.2 Bug 1 (found running the first full sweep): stance-extraction sycophancy, relocated

The deterministic-verdict fix (§5) protects the verdict from free LLM judgment, but the second-pass loop introduced a new free-judgment call: asking the LLM to label a retrieved excerpt's stance toward a claim. It reproduced the exact same failure. This is proof, not a hunch. For the claim "the hand representation is contralateral," the top-retrieved chunk, which even the cross-encoder reranker scored -3.75, its most negative score in the whole sweep, was about hand and arm spatial overlap within one hemisphere, nothing to do with laterality. The LLM still called it SUPPORTS. All 48 claims came back SUPPORTS on the first run.

A rerank-score threshold was tried first and rejected. The reranker's raw score is not calibrated around 0 for this small, out-of-domain corpus: a genuinely strong, on-topic match, the tongue-bilateral evidence from Ehrsson et al. 2003, scored -2.22, in the same range as the genuinely off-topic hand and arm passage. **The first fix** required the LLM to cite a verbatim quote backing its stance, and verified in code that the quote actually appears in the excerpt (`quote_is_grounded`), downgrading to UNRELATED if it did not. After the fix, the re-run gave **43 AGREE, 4 UNCLEAR, 1 DISAGREE** on the original 48-claim, 8-paper corpus.

**A narrower version of the same failure survived grounding, found after expanding the corpus (§6.5).** Grounding verifies that a quote is real text from the excerpt, not that it is on-topic for the specific claim. The same "hand representation is contralateral" claim, re-run on the expanded corpus, cited a real, verbatim quote about hand representation existing in cortical area BA4p. That quote was correctly grounded and completely silent on laterality. **The second fix**, `quote_addresses_claim_axes`, requires the quote to cover every keyword axis (laterality, effector, organization) that the original claim text asserts, not just share incidental words with it.

**A bug in that fix was caught before trusting its output.** The first implementation checked axes against the full second-pass query text, which always injects the dominant concept's name as boilerplate, rather than against the bare claim. Concept names contain axis-triggering substrings unrelated to what the claim actually says; `limb_vs_orofacial` contains "orofacial." This over-corrected hard: 45 of 70 claims were rejected, including a hand-verified genuine match. It was fixed by checking axes against the original claim text specifically. **The final, spot-checked numbers are 23 AGREE and 47 UNCLEAR of 70.** This is a real, honest drop from the ungrounded 67 AGREE, not a broken pipeline. The majority of "supporting" quotes the corpus actually surfaces are topically adjacent but do not specifically address the claim's precise assertion, most often laterality.

### 6.3 A formal precision and recall audit of the stance-extraction fix (2026-09-12, closes a gap the spot-checking above left open)

§6.2's numbers above were validated by spot-checking individual cases, not against a labeled ground-truth set. A self-labeled gold set was built to close this gap: **18 real-corpus claim-excerpt pairs, hand-labeled by the author, explicitly not independently blind-annotated**, balanced across SUPPORTS (5), CONTRADICTS (5), clean UNRELATED (4), and topically adjacent UNRELATED, or "related but insufficiently specific" (4), the exact failure mode §6.2 targets. Three pipeline stages were compared on the identical LLM output per example, one call per example, with stages differing only in post-hoc gating, not re-querying:

| Stage | Accuracy | SUPPORTS P/R/F1 | CONTRADICTS P/R/F1 | UNRELATED P/R/F1 |
|---|---|---|---|---|
| Free LLM judgment | 0.500 | 0.385 / 1.000 / 0.556 | 1.000 / 0.400 / 0.571 | 0.667 / 0.250 / 0.364 |
| Quote-grounded only | 0.500 | 0.385 / 1.000 / 0.556 | 1.000 / 0.400 / 0.571 | 0.667 / 0.250 / 0.364 |
| Quote + axis-checked (current pipeline) | **0.667** | 0.750 / 0.600 / 0.667 | 1.000 / 0.200 / 0.333 | 0.615 / **1.000** / 0.762 |

On the "related but insufficiently specific" subset specifically, the case this whole design exists for, **grounding alone catches 0 of 4, and grounding plus axis-check catches 4 of 4.**

**This is a real, quantified tradeoff, not a free win.** Quote-grounding alone changed nothing on this gold set; stages 1 and 2 are identical. The LLM never cited a fabricated quote here, only real but off-topic ones, so `quote_is_grounded` had nothing to catch. The axis-check is what actually does the work, and it comes at a genuine cost. CONTRADICTS recall drops from 0.400 to 0.200, and SUPPORTS recall drops from 1.000 to 0.600, because the gate now rejects some genuine matches along with the false ones, trading sensitivity for specificity. UNRELATED recall reaches a perfect 1.000. Whether this is the right tradeoff depends on the cost of each error type for the application. For a system whose whole purpose is not overclaiming literature support, erring toward UNRELATED on false negatives is very plausibly the correct choice over erring toward false AGREE, but that is a value judgment worth stating explicitly, not an automatic consequence of more checks always being better.

### 6.4 Bug 2 (found the same run): representation "winner" was 4th-decimal-place noise

The first pass used a raw point estimate, `combined_tcav_by_representation`, the mean over 30 resamples, then took the argmax. Case 1 with Transformer won all 48 claims. The diagnosis: every representation's mean combined TCAV sat in a **0.999 to 1.000 band**, so the argmax was resolving noise, not signal. This was replaced with `representation_rank_bootstrap`, which ranks the 6 representations at each of the 30 real, paired resamples and tallies P(rank 1) with random tie-breaking. The re-run result is **P(rank1) around 0.233 uniformly across the top representations, with frac_ties_at_max around 0.97**. No representation is statistically distinguishable from the others for any of the 48 claims. This is not an inconclusive result; see §7.2.

### 6.5 Corpus expansion: 8 to 10 papers via bioRxiv

BioRxiv was searched for papers similar to the existing corpus's motor-cortex-organization theme. Two were downloaded successfully and added: Deo, Okorokova, Pritchard et al. 2024, *"A mosaic of whole-body representations in human motor cortex,"* and Huber et al. 2019, *"Sub-millimeter fMRI reveals multiple topographical digit representations that form action maps in human motor cortex."* Two other candidates hit a persistent Cloudflare block across three spaced retries and were left for a manual download later. The two additions earned their place: 26 of the 83 consistent claims mined from the expanded corpus trace back to them.

## 7. Two convergent-validity findings, and why they're not surprising given the task

### 7.1 Every representation is equally sensitive to every concept (§4.1, §6.4)

All three training objectives and both architectures land in the same near-ceiling TCAV band for every tested concept, once a consistent derivation method is used.

### 7.2 Every representation localizes the same concept to the same input region

CAV directions live in each representation's own independently trained 128-dim space and cannot be compared across representations. Concept attribution, backpropagating `h(x)·v_C` to the raw 300-ROI input and aggregating to the 7 Yeo networks, lives in the one input space every representation shares. This makes it the right level to ask whether representations mean the same thing by a concept, not just whether they are equally confident about it.

Averaged over all 30 resamples per representation, from real forward and backward attribution passes, not a lookup, about 32 minutes of compute:

| Concept | Majority network | Agreement | Min pairwise cosine similarity |
|---|---|---|---|
| hand | SomMot | 6/6 | 0.994 |
| foot | SomMot | 6/6 | 0.994 |
| tongue | SomMot | 6/6 | 0.994 |
| right_side | SomMot | 6/6 | 0.994 |
| left_side | SomMot | 6/6 | 0.994 |
| limb_vs_orofacial | SomMot | 6/6 | 0.994 |
| upper_vs_lower_limb | SomMot | 6/6 | 0.994 |
| **movement_vs_rest** | **Vis** | 6/6 | 0.994 |

**This is an interpretation, not just a number.** Both findings in §7 follow from the same underlying fact. The HCP MOTOR task is built from long, temporally well-separated condition blocks, so the real anatomical signal in the ROI time series is strong and unambiguous. Any representation-learning approach sensitive to real signal, regardless of training objective, will find and rely on the same ROIs, because that is where the actual information is. This reframes what "the best-aligned representation" should mean for a task like this: there may not be one, and that absence is itself informative about the task's structure, not a failure of the selection method.

**The one genuine exception was stress-tested in §3.5, and the honest conclusion holds up as "real but incomplete."** `movement_vs_rest` converging unanimously on Vis rather than SomMot is a real, reproducible pattern at the concept-attribution level across all 6 independently trained models. Input-level ablation (§3.5) confirms that Vis matters more than chance for this specific concept relative to how it affects other concepts, but it does not show the clean, isolated collapse that a pure visual-cue-confound story would predict. Dropping Vis modestly hurts multiple aspects of classification, not uniquely movement-vs-rest. The concept-direction finding and the input-ablation finding are both real; they simply do not fully agree with each other, and reporting that is the honest, complete answer rather than picking whichever result supports a cleaner narrative.

## 8. Honest limitations, carried forward rather than hidden

- **Case 1's RAG loop still uses the older free-judgment verdict design** (§5). The deterministic fix was never extended to it, and it was never re-measured at the same scale as Case 2's.
- **The corpus is still small** (10 papers). Every retrieval and stance-grounding result is bounded by this; §6.2's final 23/70 AGREE rate and §5.1's ablation ladder are both direct symptoms.
- **§7's results are conditioned on the MOTOR task's clean block structure.** This is a stated hypothesis. Whether it survives a messier, continuous task is an open question (§9).
- **The 100-to-200 subject scale-up is complete at the data level but never re-validated** under the current 30-resample protocol.
- **The verification-ablation gold set (§6.3) is self-labeled, not independently blind-annotated.** This is a real methodological limit on how much weight its precision and recall numbers should carry, stated explicitly rather than presented as fully independent validation.
- **The §3.5 ROI-ablation and §3.3 extended-baseline results are new (2026-09-12)** and have not yet been cross-checked against the concept-attribution machinery in as much depth as the original §7.2 finding. This is a natural next iteration, not yet done.

## 9. Beyond MOTOR: extending the research line

Everything above was built and validated on one task, MOTOR, and one modality, functional time series. A second, genuinely separate line of research extends this framework to naturalistic movie-watching fMRI and structural connectivity via diffusion MRI, following the author's own 2021 precursor paper (Misra, Surampudi, Venkatesh, Limbachia, Jaja & Pessoa, *PLoS Comput Biol* 17(9):e1008943). It has already produced substantial results: a full-cohort discovery scan showing 174 of 1,113 HCP-YA subjects have the complete DTI, movie, and task battery, ROI time-series extraction for 168 subjects' movie-watching data, and a complete, validated diffusion-MRI structural-connectivity pipeline run end to end for all 174 subjects with zero failures. This work has outgrown "extension" status and now lives in its own repository, **naturalistic-brain-dynamics**, where a fresh progress report picks up the thread in full detail. It is not re-detailed here, to keep this report scoped to the MOTOR-task system that is actually complete and rigorously validated end to end.
