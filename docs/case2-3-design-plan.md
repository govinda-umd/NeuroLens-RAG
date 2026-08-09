# Cases 1-3: A Representation Learning Throughline — Design Plan

> Design specification for Case 2 and Case 3, revised after correcting the framing of the whole project. **The organizing theme across all three cases is representation learning** — decoding (Case 1), contrastive multimodal alignment (Case 2), and probabilistic/generative dynamics (Case 3) — the same throughline as Eva Dyer's group's work (POYO, POSSM; see [literature-notes-tokenization.md](literature-notes-tokenization.md)). This supersedes the previous version of this document, which mis-scoped Case 2 as plain next-timestep forecasting.

## 0. Corrected framing

**Case 1** was originally conceived as a stepping stone toward representation learning — build a working encoder, verify it end-to-end, learn the pipeline. It ended up also being a strong standalone result (92.5% test macro-F1 multi-task decoding) — a genuine bonus, not the point. Reframed correctly: Case 1's Transformer produces a pooled 128-dim representation trained jointly for classification and HRF regression; that representation *is* the actual artifact of interest, and the decoding accuracy is one way (not the only way) of evidencing that it's good. This reframing doesn't require new code — it changes how the existing work gets described (see §5, resume framing) and motivates the concept-activation-vector work in §4.

**Case 2** is not forecasting. It's **contrastive multimodal representation learning**: two encoders — one for ROI time series, one for the condition/label — trained to align in a shared embedding space via a contrastive objective, CLIP-style (Radford et al. 2021). "Predicting the next timepoint" reappears here, but properly motivated, as an optional generative capability *conditioned on* the learned joint representation (§2.4) — not as the primary objective the way the previous version of this doc had it.

**Case 3** is unchanged in spirit from before: a Bayesian PGM + dynamical-systems model (SLDS/rSLDS) — deferred until after Case 2, per direction.

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

## 3. Case 3 — Bayesian dynamical systems (rSLDS), deferred until after Case 2

Unchanged from the previous plan, confirmed to come after Case 2: fit a recurrent switching linear dynamical system via **`lindermanlab/ssm`** on the raw ROI time series, and test whether its unsupervised discrete regimes correspond to the known movement conditions — a PGM-native, structurally interpretable alternative to Case 2's learned embedding space. Once both Case 2 and Case 3 exist, their representations can be compared directly (does contrastive alignment and unsupervised dynamical regime-switching converge on similar task structure, discovered two independent ways?).

## 4. Concept Activation Vector (CAV/TCAV) testing for Case 1 — **done**

Implemented in `src/neurolens/concepts.py` and [`09_concept_vectors.ipynb`](../notebooks/09_concept_vectors.ipynb). Scope: **label-derived concepts** (not literature-derived yet — that remains the harder open problem from [interpretability-methods-notes.md §4.1](interpretability-methods-notes.md#41-a-neurolens-rag-specific-variant-literature-derived-concept-hypotheses)). Five concepts tested on Case 1's Transformer, probed at the pooled 128-dim representation shared by both heads: `hand`, `foot`, `tongue` (each vs. everything else), and `right_side`/`left_side` (lateralized movements against each other).

**Result**: effector concepts scored exactly as expected (TCAV 1.0 for their own classes, 0.0 elsewhere, >0.99 probe accuracy). Laterality concepts surfaced a real, documented limitation — a CAV fit only on lateralized classes extrapolated a specific (not random) direction onto `baseline`/`tongue`, which were never part of its training set; see [interpretability-methods-notes.md §4.1](interpretability-methods-notes.md#41-a-neurolens-rag-specific-variant-literature-derived-concept-hypotheses) for the full write-up. This validates the TCAV mechanism against ground truth before the literature-derived version is attempted.

For each: gather positive/negative example windows from the training set by true label, extract pooled representations from the trained model, fit a linear probe (the CAV direction), then compute TCAV sensitivity scores against each class's logit for held-out test windows. This is real, verifiable science on this dataset (we know the true labels), and validates the CAV *mechanism* before ever attempting the much harder literature-derived version.

## 5. Data / paper corpus correction

The current 6-paper corpus over-indexes on infrastructure papers (Yeo 2011 parcellation, Van Essen 2013 HCP overview) that ground the *pipeline's methods* but aren't directly comparable *results*. What's actually useful for the RAG "compare our result to the literature" framing is **papers that decoded/classified the HCP MOTOR task themselves** — their reported accuracies and findings are the ones worth setting our 92.5% macro-F1 against. See a candidate list gathered separately.

## 6. Resume/portfolio framing (corrected)

Case 1, described accurately, is a **multi-task representation learning** result, not merely "a decoding model" — that's the vocabulary a PhD-level ML/AI resume should use, and it wasn't used the first time. Case 2, once built, becomes a genuinely strong complementary bullet: contrastive multimodal representation learning is squarely "modern AI lingo" (the same family as CLIP) and is a different, more advanced claim than a discriminative classifier. Case 3 supports the Bayesian-PGM-plus-dynamical-systems research narrative directly.

## 7. Compute feasibility (M3, 16GB)

Unchanged from before — Case 2's two small encoders plus a contrastive loss are comparable in size/cost to Case 1's existing models; `ssm` (Case 3) is CPU-bound and lightweight. No new hardware concerns.

## 8. Case 2 v2 results — **done**

Trained and evaluated in [`10_contrastive_representation.ipynb`](../notebooks/10_contrastive_representation.ipynb). Full numbers: [`results/case2_contrastive_results.json`](../results/case2_contrastive_results.json).

**Primary result** (window=32, Transformer, matching Case 1's config for direct comparison): test macro-F1 = **0.907**, vs. Case 1's direct-classifier **0.925** — the contrastively-aligned representation trails the freely-learned classifier by a real but modest ~1.8 points. This is exactly the "aligned vs. task-optimized representation" gap predicted in §2.3: constraining the decision boundary to be a semantically-initialized text-embedding direction, rather than a fully free linear layer, costs a small amount of raw accuracy. Text-to-brain retrieval precision@20 was a perfect 1.00 for all 6 classes.

**Window size (16/24/32/48/64) × architecture (GRU/Transformer) sweep** — the standout finding: **the architecture gap is much larger and more consistent here than in Case 1.** Transformer beat GRU at every single window size (0.888–0.907 vs. 0.815–0.851, a ~5–9 point gap) — versus only a ~1–2 point Transformer-over-GRU gap on Case 1's direct classification task. So re-running the GRU-vs-Transformer comparison for a different objective *was* worth doing — the contrastive objective rewards the Transformer's architecture more than plain supervised classification did, plausibly because aligning to a fixed, semantically structured target space benefits more from self-attention's flexibility than from a single recurrent summary vector. Window size itself showed no strong monotonic trend for either architecture (peaks around 24–32 for Transformer, 48 for GRU); GRU degraded most at the longest window (64), consistent with recurrent architectures generally handling longer sequences less gracefully than self-attention.

**Text-embedding-model comparison** (at the best window/architecture config) — a genuine, worth-remembering negative result: the default `sentence-transformers/all-MiniLM-L6-v2` (384-dim) was the *best* of three (0.907), beating both a smaller model (`paraphrase-MiniLM-L3-v2`, 0.891) and a larger, generally higher-quality-on-benchmarks model (`all-mpnet-base-v2`, 768-dim, 0.888). Don't over-read this as "bigger text models are worse" — with only 6 fixed, simple condition descriptions repeated across the whole dataset, there's very little for a "better" general-purpose sentence embedding model to actually improve; general text-similarity benchmark quality isn't obviously the thing that matters when the text side is this small and this simple.

## 9. Proposed build sequencing (updated)

1. ~~**CAV/TCAV for Case 1**~~ (§4) — **done**.
2. **Paper corpus correction** (§5) — real HCP MOTOR-decoding comparison papers *and* real motor-cognition/neuroscience hypothesis papers identified (Ehrsson et al. 2003, Meier et al. 2008, Wang et al. 2020, a GNN decoding paper); pending manual download (automated fetch blocked by PMC/publisher access gates for all four).
3. ~~**Case 2 v1**~~: skipped directly to v2.
4. ~~**Case 2 v2**~~ — **done** (§8).
5. **Case 2 v3 (stretch)**: conditional generation of future ROI activity from the learned joint representation. Not started.
6. **RAG↔CAV loop** (§4.1 of interpretability-methods-notes.md): buildable once the motor-cognition papers are indexed — retrieve a real neuroscience claim, extract its concept phrase, fit a CAV, test the model's sensitivity to it, closing the loop between retrieval and interpretability for real (not just label-derived) concepts.
7. **Case 3**: rSLDS via `lindermanlab/ssm`, after Case 2.

## References

- [case1-summary-report.md](case1-summary-report.md)
- [ml-design-report.md](ml-design-report.md)
- [interpretability-methods-notes.md](interpretability-methods-notes.md) §4.1
- [literature-notes-tokenization.md](literature-notes-tokenization.md) — POYO/POSSM background, the Eva Dyer throughline this design follows
- Radford et al. 2021, "Learning Transferable Visual Models From Natural Language Supervision" (CLIP) — the contrastive objective Case 2 follows
- [lindermanlab/ssm](https://github.com/lindermanlab/ssm) — rSLDS implementation for Case 3
