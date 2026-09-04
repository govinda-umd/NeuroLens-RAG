# NeuroLens-RAG: End-to-End Report

> Consolidated, current-as-of-2026-08-27 account of the full system: data → three representation-learning paradigms → interpretability → literature verification (v1 and v2). Where a number here differs from an older doc, this report is correct — several of the headline claims in earlier write-ups were superseded by later, more rigorous experiments (flagged explicitly below, not silently). §§1-8 describe that system as it stood on MOTOR alone; §9 (added 2026-08-28) picks up the trajectory from there — the extension work in progress and the ideas it's building toward, so the project's direction is legible to someone who wasn't in the room for it.

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

## 9. Beyond MOTOR: extending the research line (2026-08-28, moved 2026-09-04)

Everything above was built and validated on one task (MOTOR) and one modality (functional time series). §7.2 and §8 both flag the obvious next question directly: does any of this survive a messier task, and is there a genuinely different modality worth bringing in? This section is a historical record of that extension work as it happened here — it has since outgrown "extension" status and moved to its own repository, **[naturalistic-brain-dynamics](../../naturalistic-brain-dynamics)**, once it became a genuinely separate line of research (structural connectivity + naturalistic movie-watching dynamics) rather than an add-on to this project's MOTOR-task + RAG-verification system. The content below is left as-is for the historical record; new work on this line happens in that repo, not here.

### 9.1 Why: extending the 2021 precursor paper's research line

NeuroLens-RAG's Case 1/2/3 + CAV/TCAV + RAG verification framework has so far only ever touched MOTOR — a task built from long, clean, discrete condition blocks. The project owner's own prior work (Misra, Surampudi, Venkatesh, Limbachia, Jaja & Pessoa, 2021, *PLoS Comput Biol* 17(9):e1008943 — GRU-based decoding on HCP naturalistic movie-watching fMRI) is the natural line to extend this system into: a continuous, mixed signal instead of clean blocks, plus a genuinely new modality (structural connectivity via diffusion MRI) the 2021 paper never touched, plus two learning paradigms beyond classification — generative forecasting of future ROI activity, and edge-level attribution on an anatomical graph rather than just ROI-level attribution on a time series.

### 9.2 Full HCP-YA discovery scan: mapping what's actually available before committing

Rather than assume subject availability, `scripts/run_full_hcp_discovery_scan.py` scanned all 1,113 HCP-YA subjects directly against S3 (2 calls each, threaded, 2026-08-27; `data/hcp_full_discovery_scan.{json,csv}`):

| | Count | % of 1,113 |
|---|---|---|
| Have DTI | 1,065 | 96% |
| Have 7T movie-watching | 184 | 17% |
| DTI + movie + all 7 standard tasks + rest | **174** | 16% |

7T eligibility, not DTI, is the real bottleneck — once a subject cleared the 7T protocol, the rest of the battery came essentially for free. Only 11 of these 174 subjects overlap with the current 90-subject MOTOR pool, which means this is functionally a **different cohort**, not an extension of the existing one. Open decision, not yet made: build the movie/SC work on this new 174-subject cohort as its own pool (loses direct subject-level comparability with the existing MOTOR numbers, gains a real within-subject three-modality design), or re-run MOTOR on this same cohort too so every result eventually shares one subject pool.

### 9.3 Movie-watching: bringing the system to a continuous, mixed signal

`docs/movie/movie-watching-dataset-plan.md` lays out bringing Case 1/2/3 + CAV/TCAV + RAG verification to HCP 7T movie-watching data, the same data the 2021 precursor paper used. One finding from that paper is close to a pre-registered confirmation of this report's own §7.2 hypothesis: a parameter-matched feed-forward network there loses to the GRU by **~45 points** (44.86% vs. 89.46%) on movie-watching, versus this report's §3.2 finding that a flat MLP *ties* Case 1's GRU on MOTOR (p=0.73). Same kind of comparison, opposite result — real evidence that temporal structure matters far more for continuous, naturalistic stimuli than for MOTOR's clean blocks. Real open design questions, not yet resolved: Case 3's alignment target on data with no discrete conditions to regress against, whether to keep NeuroLens-RAG's fixed 32-TR windowing convention or the precursor paper's continuous per-timepoint decoding, and a movie-content concept taxonomy for CAV/TCAV (scene content, not motor effectors).

**ROI-timeseries extraction, completed (2026-09-01 → 2026-09-02, ~22.6 hours).** `scripts/movie_roi_extraction.py` mirrors the MOTOR pipeline's exact masker settings (Schaefer-300, same standardization/detrend, `Movement_Regressors` confound regression) for direct cross-modality comparability, producing `X` only — no labels yet, per the explicit instruction to prepare the data before deciding what to build on it. `scripts/run_movie_extraction_batch.py` ran this over all 4 movie runs for the same 174-subject cohort the DTI/SC batch used (§9.4), not the full 184-subject movie-eligible pool, so movie and structural data exist for the same subjects. **685/696 runs succeeded, 11 failed** — genuine 404s, not crashes: 6 subjects turned out to be missing 1–2 of their 4 movie runs on S3, meaning the original discovery scan's `has_movie` check (which only tested for *any* movie folder, not all 4 specific runs) missed that they'd only partially completed the 7T protocol. **168/174 subjects have the complete 4-run set.** Every successful run validated finite with a consistent timepoint count across subjects per run (921/918/915/901 TRs for MOVIE1–4). Total footprint: 731MB, against ~0.95TB of raw functional volumes downloaded and deleted per-run along the way — the same disk-discipline pattern as §9.4's SC pipeline.

### 9.4 Structural connectivity: a new modality, validated by actually running the pipeline

`docs/structural/dti-sc-pipeline-plan.md` — turning HCP diffusion MRI into per-subject structural connectomes on the same Schaefer-300 parcellation the fMRI pipeline already uses, so a future graph-neural-network extension (parked in `case2-3-design-plan.md`) aligns node-for-node with the existing ROI time series with no cross-atlas correspondence step needed.

**Tooling and normalization, resolved rather than assumed.** MRtrix3, FSL, and ANTs command-line tools (the field-standard pairing for this data) are not installable in this environment — resolved with DIPY (pip-installable) for CSD-based tractography and `antspyx` for applying HCP's nonlinear warp fields. A literature search specifically on streamline-count normalization found no field consensus (raw count, SIFT2/LiFE-weighted, length-normalized, volume-normalized, and log-transformed schemes are all in active use, with no agreed default) — so rather than bake in one contested choice at preprocessing time, the pipeline stores raw sufficient statistics (streamline count, mean streamline length, ROI volumes) and defers the normalization choice to analysis time (`src/neurolens/sc_normalization.py`). Chosen default: $S_{ij} = \log_{10}(1 + N_{ij})$ on the raw counts.

**Single-subject smoke test (subject 100610, 2026-08-28), validated end to end.** All 300 Schaefer labels survive the MNI→native-diffusion-space warp (86% overlap with the diffusion brain mask); CSD + whole-brain probabilistic tractography produces 1.23M streamlines in ~36 minutes.

**LiFE, investigated with real rigor and then dropped, not silently degraded.** LiFE (DIPY's streamline-weighting method, the intended SIFT2 analog) was tried at four different scales — the full 1.2M-streamline tractogram, a 200K global-random subsample, a 50K global-random subsample, and a 45,206-streamline *stratified* subsample specifically designed to guarantee every one of the subject's 52,968 real ROI-pair connections at least one representative streamline. All four were killed by the OS. The last attempt was run alone, with no competing processes, and confirmed via explicit exit-code capture to be **SIGKILL (137)** — ruling out streamline count, subsampling strategy, and process contention alike as the cause. Most likely driver: HCP's 288-direction gradient table (far denser than LiFE's published validation scale) overwhelms DIPY's per-voxel signal-prediction design matrix independent of how many streamlines are fit. LiFE was dropped from the pipeline's deliverables entirely rather than shipped as a zero-filled array that would look like a real null result without being one — raw count and mean length don't depend on it and are unaffected.

**Batch runner, resumable and crash-isolated.** `scripts/run_dti_sc_batch.py` runs each subject's DIPY pipeline as an isolated subprocess (a crash during any one subject's tractography can't take down the whole multi-day orchestrator), skips subjects already completed, deletes each subject's raw diffusion volume and T1w file immediately after its arrays are built, and does not retain the full tractogram in batch mode (~1.2GB/subject × 174 would otherwise add ~200GB on top of the ~226GB of diffusion-volume downloads already budgeted).

**Batch complete (2026-09-01): 174/174 subjects, 0 failures, 0 crashes, 0 degenerate outputs.** Ran ~3.9 days (2026-08-28 20:16 → 2026-09-01 12:40) at a steady ~1.95 subjects/hour — the subprocess-isolation fix held for the full run with no OOM kills or silent deaths of the kind that repeatedly hit the LiFE investigation (§9.4 above). Every subject's three arrays validated present; connectome statistics are consistent with the single-subject smoke test and show no degenerate (all-zero) cases: mean 50,459 nonzero ROI-pair connections per subject (range 33,740–61,374), mean max streamline count 2,878 (range 1,177–4,731). Total footprint for the full cohort: **1.3GB** (vs. the ~226GB of raw diffusion volumes downloaded and deleted per-subject along the way) — the sufficient-statistics design (§9.4) doing exactly what it was meant to.

### 9.5 The three research extensions this infrastructure is being built for

None of these are designed yet — named here as the destination this data-acquisition work is aimed at, the same way Case 2/3's design predated their MOTOR implementation:

1. **Contrastive concept-mining on movie-watching's messier signal.** MOTOR's clean blocks mean every representation already converges on the same obvious concepts (§7). Movie-watching's continuous, mixed signal is the right setting to look for genuinely novel concepts in a Case-2-style contrastive setup, not just re-confirm what MOTOR already showed.
2. **Generative forecasting of future ROI time series.** Already scoped in `case2-3-design-plan.md` §2.4 and partially explored for Case 2 on MOTOR via a frozen linear probe (`results/case2_forecasting_results.json`) — flagged in §8 as unstarted at real scale. Movie-watching's long, continuous timeseries is a structurally better fit for real sequence forecasting than MOTOR's short causal windows.
3. **Edge-level concept attribution on the structural connectome.** "Which anatomical connections were responsible for a given concept" is the graph-native version of an idea already logged (`docs/interview-prep-neurolens-rag.md` §6.6 names GNNExplainer/PGExplainer specifically for this), waiting on the SC-GNN front-end this pipeline's node correspondence (§9.4) is designed to support.

### 9.6 What's still open

- The 174-subject-cohort-vs-shared-pool decision (§9.2) is unresolved.
- A GNN front-end architecture, a movie-watching concept taxonomy, the forecasting objective's exact form, and the edge-attribution mechanism each need their own design pass before any of §9.5 is buildable.
- A motor/movie/structural repo restructuring is planned (`docs/repo-restructuring-plan.md`) but the high-risk migration of existing motor content is deliberately deferred, not scheduled.
- **The Case 1 vs. Case 3 attribution comparison in §7.2 uses resample 0 for the four §6 pipeline steps that need a concrete model (not the population-level TCAV parts), while §7.2 itself is averaged over all 30** — the two analyses use different amounts of the available population by design (§6's per-claim pipeline needs one concrete representation to run retrieval/stance/verdict against; §7.2 is a standalone population-level check), not an oversight, but worth stating plainly rather than letting the two "30 resamples" claims blur together.
