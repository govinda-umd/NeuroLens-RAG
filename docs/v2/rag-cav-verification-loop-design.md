# RAG-CAV Verification Loop v2 — Claim-First Design

> Design specification for merging the two currently-separate RAG-CAV tracks into one loop that starts from a mined, self-consistent literature claim instead of a single decoded window. First proposed 2026-08-25, written up as §12 of [`docs/interview-prep-neurolens-rag.md`](../interview-prep-neurolens-rag.md); this doc pulls that proposal out into its own spec, separate from the interview-rehearsal document, since it's a real build target rather than interview color. Nothing in this document is built yet except where explicitly marked **(built)**.

## 0. Where this sits relative to v1

The existing system has two RAG-CAV tracks, built and measured independently:

1. **Query-first, per-decode loop** (`explain_decoded_window_with_cav_loop` / `explain_decoded_window_with_query_refinement` in `src/neurolens/pipeline.py`): decode → attribution → query → retrieve → LLM extracts a stance+claim from the retrieved excerpt → claim becomes a CAV/TCAV probe → deterministic verdict. Verifies literature *about one specific prediction*.
2. **Corpus-first mining sweep** (`results/corpus_claim_extraction_sweep.json`, built as a standalone script, not yet a reusable pipeline function): keyword pre-filter → 3x self-consistent LLM extraction over every surviving chunk, independent of any decode. Mines the corpus for claims *without reference to any model*.

These answer different questions and don't currently talk to each other. This design connects them: mine claims first (track 2), then verify each claim against whichever trained representation actually depends on it (a generalization of track 1's CAV step), then use that representation's own attribution pattern to drive a second, targeted literature search — closing the loop from "what does the literature say" back to "does a real trained model's own reasoning corroborate it," in the direction that's currently missing.

```
today (two disconnected tracks):

  [one decode] -> attribution -> query -> retrieve -> LLM claim -> CAV/TCAV -> verdict     (track 1)
  [whole corpus] -> keyword filter -> LLM claim (3x consistency) -> concept mapping         (track 2, stops here)

v2 (one loop, claim-first):

  [whole corpus] -> keyword filter -> LLM claim (3x consistency)
        -> CAV/TCAV against all 6 representations (3 paradigms x 2 architectures), one uniform
           post-hoc-classifier-head mechanism for every case, no case-specific derivation trick
        -> pick best-aligned representation
        -> concept-attribution on that representation's own held-out set (not prediction-attribution,
           no single decode involved)
        -> second-pass query, framed as "what this representation actually relies on for this concept"
        -> second, targeted retrieval pass over the (already-filtered) corpus
        -> deterministic synthesis: literature claim vs. model's own measured behavior
```

## 1. Corpus, chunking, indexing — reused unchanged **(built)**

`ingest_pdf_directory` → `chunk_words` (`src/neurolens/retrieval.py`): each PDF is converted page-by-page via `pymupdf4llm`, then split into overlapping 220-word windows per page (50-word overlap, `DEFAULT_CHUNK_SIZE`/`DEFAULT_OVERLAP`) — finer-grained than one chunk per page.

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (`EMBEDDING_MODEL_NAME`), domain-adaptively fine-tuned via contrastive fine-tuning on in-domain query-passage pairs (top-1 chunk-retrieval accuracy 43.9%→61.0%, top-3 75.6%→85.4% on an 880-chunk gold-labeled benchmark).

**Precision point, keep distinct**: this is a separate MiniLM instance from the frozen MiniLM used in Case 2/3 for brain–text CAV alignment (`contrastive.py`'s `CONDITION_DESCRIPTIONS` + `encode_condition_prototypes`). Same base architecture family, different weights, different job. Don't conflate "the MiniLM" as if there's only one in the project — there are two, doing unrelated things.

**Terminology, worth being precise about**: MiniLM is **not** an LLM in the generative sense — it's an encoder-only distilled transformer with no text-generation capability at all, used here purely as a sentence-embedding model (a bi-encoder: text in, fixed-length vector out, nothing else). The one generative LLM anywhere in this project is `mlx-community/Llama-3.2-3B-Instruct-4bit` (§3). MiniLM and Llama play entirely different roles and neither can substitute for the other.

## 2. Keyword pre-filter before any LLM call — reused unchanged **(built)**

Broad keyword list (`somatotop, homuncul, topograph, hemispher, ipsilateral, contralateral, lateraliz, lateral, asymmetr, bilateral, unilateral, limb, digit, finger, toe, hand, foot, feet, tongue, orofacial, articulat, effector, gradient, selectiv`), deliberately broader than the narrow 5-concept keyword list used for phrase-to-concept mapping. Measured against the real 879-chunk corpus: broad list admits 432 chunks (49.1%), narrow list admits 301 (34.2%) — 136 chunks caught only by the broader filter. Reduces LLM calls before the expensive step, no change needed for v2.

## 3. Claim extraction, concept mapping, CAV/TCAV verification — extraction is built, mapping is proposed to change

`build_concept_extraction_prompt` run against `mlx-community/Llama-3.2-3B-Instruct-4bit` (`DEFAULT_LOCAL_LLM`, via `mlx_lm`), 3x per surviving chunk. A claim is kept only if it recurs in ≥2 of 3 repeats, checked at the concept level rather than exact-phrase match (phrasing varies run to run even when the underlying claim doesn't). This is the "claims need to be relevant" requirement, operationalized as self-consistency rather than trusting a single LLM call.

**Phrase-to-concept mapping, current state, verified against the code**: `map_phrase_to_known_concept` (`concepts.py`) is deliberately simple keyword substring matching against `LITERATURE_CONCEPT_KEYWORDS` — not a learned classifier, no embedding involved. Real limitation: a claim phrase whose wording doesn't happen to contain one of the hard-coded keywords maps to nothing, even if it clearly means one of the known concepts.

**Proposed replacement, not yet built**: embed the claim phrase and each concept's reference text (its keyword set, or a short canonical description) with the retrieval MiniLM (§1), and use cosine similarity + softmax to get a *soft* weighting over concepts instead of a hard keyword match — e.g. a claim maps 0.6 to `hand`, 0.3 to `upper_vs_lower_limb`, 0.1 elsewhere, rather than a single all-or-nothing bucket or nothing at all. This directly reuses the "combine scores" resolution already reached in the interview-prep doc's §6.6 (open-claim soft-decomposition discussion): TCAV is only ever defined relative to one class at a time, so the well-defined way to use a soft weighting is to run the existing per-class TCAV computation once per concept the claim partially maps to, then linearly combine the resulting scalars by the similarity weights — not to blend CAV *directions* before testing, which was already shown to be underspecified there.

**CAV/TCAV derivation, standardized across all three cases — no open-vocabulary route for Case 2.** Every case now tests a mapped concept the same way: fit a linear probe / classification head on the frozen backbone's pooled features from labeled examples, then use that head's differentiable logit for CAV/TCAV's directional derivative. For Case 1 this is the model's own trained head; for Case 2 and Case 3 it's a **post-hoc-fitted** head on frozen contrastive features. This is not a new idea to build — it's already implemented and validated: `case3.py::fit_post_hoc_classifier` fits an `sklearn.LogisticRegression` on frozen backbone features, then copies its weights into a real `nn.Linear` (wrapped in `BrainWithPostHocClassifier`) so CAV/TCAV's autograd-based directional derivative can run through it exactly like Case 1's native head. It was reused **completely unmodified** on Case 2's `ContrastiveModel` for the fitted-probe CAV sweep (`results/case2_fitted_probe_cav_sweep.json`, commit `d606413`) because `ContrastiveModel` happens to share the same `.brain_backbone` attribute naming as `BrainHRFModel` — result: probe accuracy 0.997–0.999, TCAV 0.92–1.00, indistinguishable from Case 1/3. Case 2's separate `concepts_case2.py::open_vocabulary_concept_direction` mechanism (adjoint pullback of a text-prototype difference through the projection's transpose, no brain examples needed) is real, working code — it is **not** deleted or wrong — but it's superseded as the *standard* mechanism for this design specifically because claims are already mapped onto known concepts before testing (previous paragraph), at which point labeled brain examples exist for every concept being tested and the no-examples-needed property the open-vocabulary route exists for stops being the limiting factor. Standardizing on the post-hoc-classifier-head route for every case is what makes §4 and §6 below uniform instead of three separate case-specific mechanisms.

**Output of this step, per claim**: a concept (or a soft weighting over several concepts) plus a stance (supports/contradicts/unrelated) plus which chunk(s) it came from.

## 4. Pick the best-aligned representation, run concept-attribution, refine the query — the actual new piece

**Two different kinds of attribution, not one — keep this distinction sharp.** (a) *Prediction-attribution*, already built (`infer_rsn_attribution`, 4-method consensus, §6.1 of the interview-prep doc): which input regions drove the model toward the class it predicted for one decoded window. (b) *Concept-attribution*, logged as an idea but not yet built (interview-prep doc §6.6, "Concept-vector input attribution"): given a fixed CAV direction $v_C$, the concept-alignment score $h(x)\cdot v_C$ is itself a differentiable scalar function of the raw input $x$ — backprop it through the backbone and the result is an input-level saliency map *for the concept*, using a different backward target than the class-logit gradient already used in (a). This design needs **(b)**, not (a): starting from a corpus-mined claim/concept rather than a decode, there is no predicted class to attribute to yet — what we want to know is which resting-state networks a given representation actually leans on *for that concept*, which is exactly what enables checking "does an RSN that should exist for this concept, given the literature, actually show up in the model's own attribution" and refining the follow-up query around it.

**The query being refined here is the second-pass query, `build_refined_query`'s role — not the first query.** `build_query_text` (§6.1-derived, prediction-attribution → text) builds the *first* query for one decode in the existing per-decode loop. `build_refined_query` already takes CAV loop results, picks the highest-TCAV concept, and builds a concept-steered follow-up query for a second retrieval round (§7 of the interview-prep doc's "CAV-aware query refinement" improvement). This design's new query is the same *role* as `build_refined_query`'s output — a second, targeted query — just built from concept-attribution's RSN pattern plus the originating literature claim, instead of from a bare concept name.

| Piece | Status | Where |
|---|---|---|
| Prediction-attribution: 4-method consensus on one decoded window | **built** | `infer_rsn_attribution`, `pipeline.py` |
| First-pass query from prediction-attribution | **built** | `build_query_text`, `pipeline.py` |
| Second-pass query from the CAV loop's highest-TCAV concept | **built** | `build_refined_query`, `pipeline.py` |
| Run a second-pass query as a real retrieval round | **built** | `explain_decoded_window_with_query_refinement`, `pipeline.py` |
| Post-hoc classification head giving every case a differentiable logit | **built**, needs to become the default for Case 2 (§3) | `case3.py::fit_post_hoc_classifier`, already validated on Case 2 unmodified |
| Concept-attribution: backprop $h(x)\cdot v_C$ to $x$ instead of a class logit | **new** — new backward target only, no new architecture, since the differentiable head from the row above already exists for every case | none yet |
| Test a claim's TCAV score against all 6 case×architecture representations and pick the best-aligned one | **new** — mechanically simple, needs the 30-resample checkpoints from `results/case{1,2,3}_bootstrap_30resamples.json` loaded and probed per representation | none yet |
| Second-pass query built from concept-attribution + the originating claim | **new**, extends `build_refined_query`'s pattern | none yet |

**On which examples to run concept-attribution — there is no open design question here.** Every case/architecture/resample already has its own held-out test split (the same one its TCAV score for this concept was already computed from). Run concept-attribution over exactly that held-out set, on the winning representation from the alignment check, and average — the same averaging TCAV itself already does over held-out examples (§6.2 of the interview-prep doc, step 6). No new sampling decision to make, no canonical-window question to resolve — it only looked unresolved because the write-up considered inventing a new example set instead of reusing the one the pipeline already carries for that exact model.

## 5. Second, targeted retrieval pass over the filtered corpus — reused unchanged **(built)**

Standard retrieve-then-rerank pattern: MiniLM bi-encoder cosine-similarity search over the already-keyword-filtered chunks, then `cross-encoder/ms-marco-MiniLM-L-6-v2` (`RERANKER_MODEL_NAME`) reranks the narrowed candidate set by scoring (query, chunk) pairs jointly — a fresh forward pass per pair, too expensive over the whole corpus, so applied only after the bi-encoder narrows the field.

**Third distinct model, worth keeping straight**: this reranker is also MiniLM-family but a separately-trained cross-encoder checkpoint — not the same weights as either the retrieval bi-encoder (§1) or Case 2's frozen prototype encoder. Three MiniLM-family models total in this system, each doing a different job.

## 6. Deterministic synthesis — carries the one real risk in this design

`build_final_synthesis_prompt` / `build_final_synthesis_prompt_case2` already exist. LoRA fine-tuning was already tried once, specifically on the verdict-judgment failure (an earlier free-judgment version defaulted to AGREE in 10 of 12 real cases regardless of the actual TCAV score): it fixed output-format compliance but **did not fix the underlying reasoning**, reproduced across 2 independent runs.

**Does this design still need LoRA? Probably not, and that's a feature, not a gap.** LoRA in v1 was patching a symptom — a free-judgment LLM that couldn't be trusted to weigh evidence. This design removes the free judgment by construction (below), which removes the problem LoRA was fine-tuning away, rather than fixing it more thoroughly. If a fine-tuned Llama is still wanted here, its job would be narrower and lower-stakes than v1's attempt: making the *narration* read better, not making a judgment more trustworthy — a fine-tuning target worth being honest about the scope of, if pursued, rather than assuming it inherits v1's original motivation.

**This is unrelated to fine-tuning MiniLM.** The domain-adaptive fine-tuning already done (§1: contrastive fine-tuning on query-passage pairs, +11–17 pts top-1 retrieval) is a completely separate effort, on a completely separate model, for a completely separate purpose — retrieval accuracy, not judgment or synthesis. MiniLM has no generation capability at all (§1's terminology note), so it cannot be fine-tuned "to generate domain-sensitive concepts" in any sense — the closest real analog would be a *further*, distinct contrastive fine-tune of MiniLM specifically for §3's proposed soft concept-matching (claim phrase ↔ concept reference text), which is a third, not-yet-started fine-tuning target, different in purpose from both the existing retrieval fine-tune and any possible Llama LoRA.

**This step is exactly where the measured sycophancy bug lived**: an earlier free-judgment version of the verdict step defaulted to AGREE in 10 of 12 real cases regardless of the actual TCAV score. If step 6 lets Llama, fine-tuned or not, freely synthesize across (claim, TCAV score, which representation won §4's alignment check, second-pass evidence), the same failure mode is a live risk at a larger scale, not a hypothetical one.

**Required design constraint, not optional**: compute any verdict-like conclusion deterministically from the numbers already in hand — exactly the rule already forced on the simpler loop (`expected_verdict_from_stance_and_tcav`, `pipeline.py`). Restrict the LLM, fine-tuned or not, to narrating an already-decided conclusion. Never let it freely weigh the combined evidence itself.

**This constraint must be applied uniformly regardless of which representation won §4's alignment check.** A claim's best-fit representation could turn out to be Case 1's — and Case 1's loop currently still uses the older free-judgment design (§7 of the interview-prep doc, still an open, unfixed gap as of 2026-08-22). Building step 6 as one shared, deterministic synthesis path used for every case — rather than reusing Case 1's and Case 2's existing, divergent synthesis code as-is — would close that standing gap as a side effect. Building it as a thin wrapper that dispatches to whichever case's existing (and still-divergent) synthesis function would not.

## 7. Concrete build list, in an order that keeps each piece independently testable

1. Standardize Case 2's CAV/TCAV derivation on `fit_post_hoc_classifier` (already built, already validated on Case 2 unmodified, §3) as the default, rather than a case-specific choice — mechanically nothing new to build, just a convention change in whichever code currently picks Case 2's derivation route.
2. A MiniLM-embedding-based soft concept-mapping function (§3): claim phrase + each concept's reference text → cosine similarity → softmax weights, replacing/supplementing `map_phrase_to_known_concept`'s hard keyword match.
3. A function that, given a concept/claim (or a soft weighting over several), loads the 30-resample checkpoints for all 3 cases × 2 architectures and returns a combined TCAV score per representation — run the existing per-class TCAV computation once per concept the claim maps to, then linearly combine by the similarity weights (§3's already-resolved "combine scores, not directions" rule). No new statistical method needed; `cross_class_rank_bootstrap_test` already exists in `concepts.py` for the significance side.
4. A "best-aligned representation" selection rule on top of (3) — simplest version: argmax combined TCAV; more defensible version: population-level using the existing rank-bootstrap significance test rather than a raw point estimate.
5. A concept-attribution function (§4): backprop $h(x)\cdot v_C$ (not a class logit) to the input, run over the winning representation's existing held-out set, averaged — a new backward target through infrastructure (the differentiable post-hoc head from item 1) that already exists.
6. A second-pass query template that states what a representation's concept-attribution + the originating literature claim jointly suggest — extending `build_refined_query`'s pattern (§4), not replacing it.
7. One shared, deterministic synthesis function used regardless of which case originated the claim — the piece that would retire the Case 1 free-judgment gap if built this way from the start (§6).

## 8. What this does and doesn't fix

**Fixes**: removes the query-first loop's dependence on starting from one specific decode; makes literature-checking claim-driven and cross-paradigm from the first step, using infrastructure (the paired 30-resample checkpoints) that didn't exist when the original query-first loop was designed.

**Doesn't fix on its own**: the corpus is still small (8 papers, 879 chunks) — that ceiling applies to v2 exactly as it applied to v1, no loop redesign changes it. And the Case 1 free-judgment gap only closes if step 6 is deliberately built as one shared deterministic path (§6) rather than dispatching to the two cases' existing, still-divergent synthesis implementations.
