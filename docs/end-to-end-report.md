# NeuroLens-RAG: End-to-End Report

> Consolidated, current-as-of-2026-08-27 account of the full system: data → three representation-learning paradigms → interpretability → literature verification (v1 and v2). Where a number here differs from an older doc, this report is correct — several of the headline claims in earlier write-ups were superseded by later, more rigorous experiments (flagged explicitly below, not silently).

## 1. The question this project answers

Accuracy alone can't tell you whether a brain-decoding model learned something neurobiologically real or an incidental shortcut correlated with the label. NeuroLens-RAG is a framework for *testing* that distinction: train the same decoding task three structurally different ways, check whether each resulting representation depends on concepts a human would recognize (which body part, which side, movement vs. rest), and cross-check those concepts against independent neuroscience literature — with the checking mechanism itself held to the same standard of evidence as the thing it's checking.

## 2. Data

HCP Young Adult, MOTOR task, 100-subject pool (a 200-subject scale-up completed at the data level but never re-validated under the current protocol — paused, not abandoned). Each subject contributes 300-channel ROI time series (Schaefer-300 parcellation of the same BOLD signal, motion-regressed, detrended, z-scored per run), windowed causally into 32-TR (~23s) overlapping segments. Per window, two supervisory signals, both read at the window's last timepoint:

- `y`: a 6-way categorical label (baseline, left/right hand, left/right foot, tongue).
- `y_hrf`: a 5-channel continuous vector (one per non-baseline condition), each channel that condition's event timeline convolved with the canonical hemodynamic response function.

Splits are always subject-level (never window-level — overlapping windows from one subject are near-duplicates and would leak across a window-level split).

## 3. Three representation-learning paradigms

Same two backbones (GRU, Transformer; parameter-matched to 1.15× after a fix — GRU now defaults to 2 layers, closing what was originally a 1.83× gap) trained under three different objectives, so any representational difference is attributable to the objective, not an architecture confound. Every backbone exposes the same `forward_features(x) -> [B, 128]` interface, which is why adding a third paradigm — and later, testing all three with one interpretability mechanism — was cheap rather than requiring three separate pipelines.

- **Case 1 (supervised):** multi-task decoder, joint 6-way classification + HRF regression off one shared trunk. Loss = CE + 0.1·MSE.
- **Case 2 (supervised-contrastive):** brain encoder aligned to 6 frozen-MiniLM condition-description prototypes in a shared 64-d space via a **symmetric multi-positive contrastive loss**, `L = ½(L_b2t + L_t2b)`. `L_b2t` is standard cross-entropy, one brain window against all 6 prototypes. `L_t2b` is the direction a closed 6-item vocabulary would normally rule out (no single brain window uniquely pairs with a prototype) — resolved by treating *every* brain window sharing a prototype's true label as a positive for that prototype, averaged (Khosla et al. 2020-style multi-positive, not literal CLIP's one-to-one pairing). This is the loss that actually trained the checkpoints behind every Case 2 number in this report; the original, simpler single-direction (brain→text only) asymmetric loss was the initial design and is still in the codebase (`contrastive.py::train_contrastive`) but is not what the 30-resample bootstrap or anything downstream of it uses.
- **Case 3 (self-supervised):** brain encoder aligned to its *own window's* HRF vector via a symmetric InfoNCE loss (HRF is continuous and per-window, unlike Case 2's fixed prototypes, so both directions have real negatives). No label ever enters this loss. `BrainWithPostHocClassifier` fits a linear probe on frozen features after training — the only way to get a macro-F1 number and a target-class logit for CAV/TCAV out of a model that never saw a label.

### 3.1 Architecture comparison — the ranking flips by paradigm, not a fixed winner

30 paired repeated-split bootstraps, all three cases sharing the exact same 30 subject partitions (Case 2/3 reuse Case 1's splits verbatim, never redrawn — this is what makes the comparison paired rather than three separately-noisy studies):

| Case | GRU mean F1 (95% CI) | Transformer mean F1 (95% CI) | Winner (paired Wilcoxon) |
|---|---|---|---|
| 1 (supervised) | 0.906 [0.876, 0.931] | 0.922 [0.902, 0.945] | Transformer, p<0.0001 |
| 2 (supervised-contrastive) | 0.900 [0.866, 0.927] | 0.918 [0.898, 0.942] | Transformer, p<0.0001 |
| 3 (self-supervised) | 0.928 [0.903, 0.946] | 0.922 [0.903, 0.944] | **GRU**, p=0.0004 |

Transformer wins whenever a label drives training directly; GRU edges ahead in the one paradigm that never sees a label. This supersedes an earlier, smaller-scale claim that "Transformer always beats GRU" — that held at single-split scale but doesn't survive the population-level, paired comparison.

### 3.2 Is a learned sequence representation even necessary? A baseline check (2026-08-27)

Before extending this system to new datasets, a more basic question: is the classes' separability coming from *learned temporal structure* at all, or is it already sitting in the raw window, accessible to any generic function approximator? Two baselines (`baseline_mlp.py`), same forward-features interface as GRU/Transformer so they plug into the identical training harness and share Case 1's exact 30 resamples for a directly paired comparison:

- **FlattenMLP**: the whole 32×300 window flattened to one 9,600-d vector → a single hidden layer → classifier. Keeps every raw value; "time" is just "which flat index."
- **MeanPoolMLP**: the window averaged over time to one 300-d vector → a single hidden layer → classifier. Keeps only the spatial (per-ROI) pattern, discards temporal dynamics entirely.

| Model | Mean F1 (95% CI) | Params | vs. GRU (paired Wilcoxon) | vs. Transformer (paired Wilcoxon) |
|---|---|---|---|---|
| GRU | 0.906 [0.876, 0.931] | 166,539 | — | — |
| Transformer | 0.922 [0.902, 0.945] | 304,907 | — | — |
| FlattenMLP | 0.907 [0.882, 0.933] | 1,230,347 | p=0.73 (not significant) | p<0.0001 (Transformer wins) |
| MeanPoolMLP | 0.654 [0.616, 0.694] | 39,947 | p<0.0001 (GRU wins) | p<0.0001 (Transformer wins) |

**The honest answer, and it's more nuanced than "yes" or "no":** a plain MLP with *zero* learned temporal structure is statistically indistinguishable from GRU (p=0.73) once it's given the full, un-destroyed window — GRU's recurrence isn't demonstrably buying anything over handing all the raw values to a generic function approximator, at least on this task. But the Transformer significantly outperforms that same flat MLP (p<0.0001) *despite having 4× fewer parameters than it* — real evidence that self-attention is doing something a naive aggregation structurally can't, not just adding capacity. And destroying temporal order entirely (mean-pooling) costs ~25 points and is significant in the other direction — so temporal information within the window clearly matters, GRU's specific way of encoding it just isn't the thing making the difference.

**What this changes about how to read the whole project**: Case 1's GRU-vs-Transformer comparison (§3.1) isn't "does sequence modeling help" — a flat MLP already answers that partially (temporal information matters, mean-pooling proves it). It's closer to "does *attention specifically* find structure that recurrence and raw aggregation both miss" — a sharper, more specific claim than the one usually being made. Worth checking on Case 2/3 too before treating this as motor-classification-general; not yet done.

## 4. Interpretability: does the representation depend on the concept, not just correlate with it

CAV/TCAV (Kim et al. 2018), the mechanism: (1) define a concept via labeled positive/negative examples; (2) fit a linear probe on the model's pooled representation — the probe's normalized weight vector is the Concept Activation Vector; (3) take held-out examples of the target class, compute the gradient of that class's logit w.r.t. the representation, and dot it with the CAV direction; (4) TCAV score = the fraction of held-out examples where that directional derivative is positive. This is a genuine local sensitivity measure within the model's own function — stronger than correlation, not the same epistemic strength as a real intervention.

**Standardized derivation mechanism (2026-08-26):** every case now tests a concept the same way — fit a classification head on frozen pooled features from labeled examples, then use that head's differentiable logit for the directional derivative. For Case 1 this is the model's own trained head; for Case 2 and Case 3 it's a **post-hoc-fitted** head (`fit_post_hoc_classifier`, originally written for Case 3, reused completely unmodified on Case 2 because both models share the same `.brain_backbone`/`.brain_projection` attribute naming). This replaced Case 2's original text-arithmetic CAV derivation (subtracting two text-prototype embeddings, pulled back through the projection's transpose) as the default — not because that mechanism was wrong, but because of a real diagnosed artifact, below.

### 4.1 The 8-concept × 3-case × 2-architecture × 30-resample sweep (1,440 evaluations)

8 concepts, each grounded in real motor anatomy: `hand`, `foot`, `tongue`, `right_side`, `left_side`, `movement_vs_rest`, `limb_vs_orofacial`, `upper_vs_lower_limb`.

**A diagnosed artifact, not a real representation-quality gap.** Case 2's original text-derived CAVs scored low on laterality (right/left TCAV ≈ 0.33) — looking like a real weakness specific to the contrastive representation. A controlled experiment settled it: fitting a linear probe on Case 2's frozen features via the *exact same method* as Case 1/3 got probe accuracy 0.997–0.999 and TCAV 0.92–1.00 — indistinguishable from the other two paradigms. Root cause: MiniLM's sentence-embedding geometry is dominated by the concrete noun (hand/foot/tongue), so subtracting text prototypes to isolate a weak, secondary axis like laterality fights the embedding space's actual geometry. **All three paradigms converge to near-ceiling interpretability once tested with a consistent method** — the interesting result was methodological (derivation method matters independently of representation quality), not a paradigm-superiority finding.

## 5. RAG v1: literature-grounded verification, per-decode

**Retrieval:** 8-paper neuroscience corpus, chunked (originally page-scoped overlapping 220-word windows), embedded with `sentence-transformers/all-MiniLM-L6-v2`, dense cosine retrieval narrowed by a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (bi-encoder for cheap whole-corpus recall, cross-encoder for precision on the narrowed set — standard retrieve-then-rerank, motivated by the real cost asymmetry between the two). The bi-encoder was later domain-adaptively fine-tuned on in-domain query-passage pairs: top-1 chunk-retrieval accuracy 43.9%→61.0%, top-3 75.6%→85.4%. The reranker was never fine-tuned.

**The loop:** decode → 4-method attribution (Saliency, Integrated Gradients, exact Shapley, LIME over the 7 Yeo resting-state networks) locates a consensus network → a natural-language query → retrieve+rerank → a local LLM (`mlx-community/Llama-3.2-3B-Instruct-4bit` via `mlx_lm`) extracts a stance and a concept phrase → the phrase maps to a known concept (keyword match) → CAV/TCAV re-probes that concept against the model's actual representation.

**The measured failure that shaped the whole design:** an early version let the LLM freely judge AGREE/DISAGREE between the literature and the CAV evidence. It defaulted to AGREE in 10 of 12 real cases regardless of the actual TCAV score — sycophancy, measured, not hypothetical. Fix: **the verdict is computed deterministically in code from (stance, TCAV score); the LLM's only remaining job is to narrate an already-decided conclusion.** This fix reached Case 2's loop; Case 1's loop still uses the older free-judgment design and was never re-measured at the same scale — a real, still-open gap.

**Corpus-first mining (2026-08-22, page-based):** independent of any decode, mine the whole corpus for claims. A keyword pre-filter deliberately broader than the closed concept vocabulary (`somatotop, homuncul, hemispher, lateral, effector, ...`) admitted 432 of 879 chunks; each survivor extracted 3× with concept-level self-consistency (≥2 of 3 repeats must agree). Result: 58 consistent claims; per-concept literature support was real and uneven (right/left = 49 hits each, tongue = 30, hand = 28, foot = 9 — thinnest support in this corpus).

## 6. RAG v2: claim-first, cross-representation, with two bugs found and fixed by actually running it

The v2 design (`docs/v2/rag-cav-verification-loop-design.md`) inverts v1's order: mine claims first, then ask which of the *6* trained representations actually depends on each claim's concept, then use that representation's own reasoning to drive a second, targeted search — rather than starting from one decode.

### 6.1 Pipeline, as built and run

1. **Section-based chunking** (`retrieval.py::ingest_pdf_by_section`): split on markdown section headers detected in the PDF (Introduction/Methods/Results/Discussion), not page boundaries, with back-matter (references) cut and any section longer than the embedding model's limit re-split by the original word-window logic. 1,018 section chunks over the current 10-paper corpus (837 over the original 8 papers).
2. **Keyword pre-filter:** 512 of 1,018 chunks survive (384 of 837 on the original 8-paper corpus).
3. **Extraction + soft concept mapping:** the same 3× self-consistency extraction as v1 (83 consistent claims, 70 unique phrases after dedup, up from 56/48 on the 8-paper corpus — the 2 added bioRxiv papers alone contribute 26 of the 83); phrases are additionally embedded and cosine-matched against each concept's reference text (softmax-weighted), so a claim can partially match several concepts instead of one hard bucket.
4. **Population-level representation selection.** Given a claim's soft concept weights, combine each representation's per-concept TCAV (already computed in step 4.1's sweep) into one weighted score, then ask *which representation ranks #1 across the 30 real resamples* — not just which has the highest single-point mean. Ties (very common near TCAV's ceiling) are broken uniformly at random and reported (`frac_ties_at_max`), mirroring `concepts.py::cross_class_rank_bootstrap_test`'s existing tie-handling convention, now applied across representations instead of classes.
5. **Concept-attribution** (new mechanism, not prediction-attribution): backprop the concept-alignment score `h(x)·v_C` — not a class logit — to the raw input, averaged over the representation's own held-out set, aggregated to the 7 RSNs.
6. **Second-pass query + retrieve/rerank**, same infrastructure as v1.
7. **Grounded stance extraction**, then a deterministic verdict, then LLM narration of the already-decided result.

### 6.2 Bug 1 (found running the first full sweep): stance-extraction sycophancy, relocated

The deterministic-verdict fix (§5) protects the *verdict* from free LLM judgment, but the second-pass loop introduced a *new* free-judgment call — asking the LLM to label a retrieved excerpt's stance toward a claim — and it reproduced the exact same failure. Proof, not a hunch: for the claim "the hand representation is contralateral," the top-retrieved chunk — which even the cross-encoder reranker scored **-3.75**, its most negative score in the whole sweep — was about hand/arm spatial overlap within one hemisphere, nothing to do with laterality. The LLM still called it SUPPORTS. All 48 claims came back SUPPORTS on the first run.

A rerank-score threshold was tried first and rejected: the reranker's raw score isn't calibrated around 0 for this small, out-of-domain corpus — a genuinely strong, on-topic match (the tongue-bilateral evidence from Ehrsson et al. 2003) scored -2.22, in the same range as the genuinely off-topic hand/arm passage. Thresholding on score would have discarded good evidence right alongside bad. **First fix:** require the LLM to cite a verbatim quote backing its stance, and verify in code that the quote actually appears in the excerpt (`quote_is_grounded`) — downgrading to UNRELATED if it doesn't. Re-run after the fix: **43 AGREE / 4 UNCLEAR / 1 DISAGREE** (on the original 48-claim, 8-paper corpus) — real discrimination, including cases where a *high*-rerank-score excerpt was still correctly caught as off-topic once the LLM had to point at real text.

**A narrower version of the same failure survived grounding, found after expanding the corpus (§6.4).** Grounding verifies a quote is *real text from the excerpt* — not that it's *on-topic for the specific claim*. The same "hand representation is contralateral" claim, re-run on the expanded corpus, cited a real, verbatim quote about hand representation existing in cortical area BA4p — correctly grounded, and completely silent on laterality. **Second fix:** `quote_addresses_claim_axes` requires the quote to cover every keyword-axis (laterality / effector / organization) the *original claim text* asserts, not just share incidental words with it — full coverage, not intersection, since a claim asserting both an effector and laterality would otherwise pass against a quote addressing only the effector.

**A bug in that fix, caught before trusting its output.** The first implementation checked axes against the full second-pass *query* text (which always injects the dominant concept's name as boilerplate, e.g. "...points to the SomMot network. Evidence for or against **limb_vs_orofacial**...") rather than the bare claim — and concept names contain axis-triggering substrings unrelated to what the claim actually says (`limb_vs_orofacial` contains "orofacial", satisfying the effector-axis check regardless of the claim's real content). This over-corrected hard: 45 of 70 claims got rejected, including a hand-verified genuine match ("a bilateral set of frontoparietal areas was active", correctly SUPPORTS, wrongly rejected). Fixed by checking axes against the original claim text specifically, threaded through as a separate `claim_text` argument. Final, spot-checked numbers: **23 AGREE / 47 UNCLEAR** of 70 — a real, honest drop from the ungrounded 67 AGREE, not a broken pipeline: the majority of "supporting" quotes the corpus actually surfaces are topically adjacent (right paper, right general subject) but don't specifically address the claim's precise assertion, most often laterality. That's a real, sobering statement about retrieval depth on an 8–10 paper corpus, not a bug to explain away.

### 6.3 Bug 2 (found the same run): representation "winner" was 4th-decimal-place noise

The first pass used a raw point-estimate (`combined_tcav_by_representation`, mean over 30 resamples, argmax). Case 1/Transformer won all 48 claims. Diagnosis: every representation's mean combined TCAV sat in a **0.999–1.000 band** — the argmax was resolving noise, not signal. Replaced with `representation_rank_bootstrap`: rank the 6 representations at each of the 30 *real, paired* resamples, tally P(rank #1) with random tie-breaking. Re-run result: **P(rank1) ≈ 0.233 uniformly across the top representations, frac_ties_at_max ≈ 0.97** — i.e., no representation is statistically distinguishable from the others for any of the 48 claims. This is not an inconclusive result; see §7.2.

### 6.4 Corpus expansion (2026-08-27): 8 → 10 papers via bioRxiv

Searched bioRxiv for papers similar to the existing corpus's motor-cortex-organization theme. 4 candidates identified; 2 downloaded successfully and added — Deo, Okorokova, Pritchard et al. 2024, *"A mosaic of whole-body representations in human motor cortex"*, and Huber et al. 2019, *"Sub-millimeter fMRI reveals multiple topographical digit representations that form action maps in human motor cortex"*. The other 2 candidates hit a persistent Cloudflare block (error 1015) across three spaced retries with different headers — not a transient rate limit, left for a manual download later. The 2 additions earned their place: 26 of the 83 consistent claims mined from the expanded corpus trace back to them.
## 7. Two convergent-validity findings, and why they're not surprising given the task

### 7.1 Every representation is equally sensitive to every concept (§4.1, §6.3)

All three training objectives and both architectures land in the same near-ceiling TCAV band for every tested concept.

### 7.2 Every representation localizes the same concept to the same input region (new, 2026-08-26/27)

CAV *directions* live in each representation's own independently-trained 128-dim space and can't be compared across representations. Concept-*attribution* — `h(x)·v_C` backpropagated to the raw 300-ROI input, aggregated to the 7 Yeo networks — lives in the one input space every representation shares, so it's the right level to ask whether representations mean the same thing by a concept, not just whether they're equally confident about it.

Averaged over all 30 resamples per representation (real forward+backward attribution passes, not a lookup — ~32 minutes of compute):

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

Unanimous on every concept, similarity never below 0.994.

**Interpretation, not just the number.** Both findings in §7 follow from the same underlying fact, not two independent confirmations: the HCP MOTOR task is built from long, temporally well-separated condition blocks, so the real anatomical signal in the ROI time series is strong and unambiguous. Any representation-learning approach sensitive to real signal — regardless of training objective — will find and rely on the same ROIs, because that's where the actual information is. High TCAV and high attribution-agreement are the same fact seen twice, not two separate surprises. This reframes what "the best-aligned representation" should mean for a task like this: there may not be one, and that absence is itself informative about the task's structure, not a failure of the selection method.

**The one genuine exception, and it's not noise:** `movement_vs_rest` converging unanimously on **Vis** (visual cortex) rather than the "obvious" SomMot answer is a real, reproducible pattern across all 6 independently-trained models — plausibly a visual-cue confound in the block design (fixation/cue differences between rest and active blocks) rather than a genuine motor signal. Flagged as worth a closer look, not explained away.

## 8. Honest limitations, carried forward rather than hidden

- **Case 1's RAG loop still uses the older free-judgment verdict design** (§5) — the deterministic fix was never extended to it, and it was never re-measured at the same scale as Case 2's.
- **The corpus is still small** (10 papers, up from 8 — §6.4). Every retrieval and stance-grounding result is bounded by this; a query-refinement experiment in v1 hit a ceiling specifically attributable to corpus size, and §6.2's final 23/70 AGREE rate is a direct symptom of it (most retrieved evidence is topically adjacent, not axis-specific).
- **§7's results are conditioned on the MOTOR task's clean block structure** — this is a stated hypothesis, not yet stress-tested. `data/raw/hcp_movie_watching/` is reserved for a planned second HCP dataset (continuous, naturalistic) specifically to test whether the convergence in §7 survives a messier task.
- **The 100→200 subject scale-up is complete at the data level but never re-validated** under the current 30-resample protocol.
- **The §3.2 baseline check is Case 1 only.** Whether Case 2/3's contrastive/self-supervised representations similarly fail to beat a flat-MLP baseline (or whether the objective itself is what makes GRU/Transformer's structure matter there) hasn't been tested.
- **The Case 1 vs. Case 3 attribution comparison in §7.2 uses resample 0 for the four §6 pipeline steps that need a concrete model (not the population-level TCAV parts), while §7.2 itself is averaged over all 30** — the two analyses use different amounts of the available population by design (§6's per-claim pipeline needs one concrete representation to run retrieval/stance/verdict against; §7.2 is a standalone population-level check), not an oversight, but worth stating plainly rather than letting the two "30 resamples" claims blur together.
