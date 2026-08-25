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
        -> CAV/TCAV against all 6 representations (3 paradigms x 2 architectures)
        -> pick best-aligned representation
        -> attribution on that representation (averaged over held-out class windows, no single decode)
        -> attribution-derived query, framed as "what this representation actually relies on"
        -> second, targeted retrieval pass over the (already-filtered) corpus
        -> deterministic synthesis: literature claim vs. model's own measured behavior
```

## 1. Corpus, chunking, indexing — reused unchanged **(built)**

`ingest_pdf_directory` → `chunk_words` (`src/neurolens/retrieval.py`): each PDF is converted page-by-page via `pymupdf4llm`, then split into overlapping 220-word windows per page (50-word overlap, `DEFAULT_CHUNK_SIZE`/`DEFAULT_OVERLAP`) — finer-grained than one chunk per page.

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (`EMBEDDING_MODEL_NAME`), domain-adaptively fine-tuned via contrastive fine-tuning on in-domain query-passage pairs (top-1 chunk-retrieval accuracy 43.9%→61.0%, top-3 75.6%→85.4% on an 880-chunk gold-labeled benchmark).

**Precision point, keep distinct**: this is a separate MiniLM instance from the frozen MiniLM used in Case 2/3 for brain–text CAV alignment (`contrastive.py`'s `CONDITION_DESCRIPTIONS` + `encode_condition_prototypes`). Same base architecture family, different weights, different job. Don't conflate "the MiniLM" as if there's only one in the project — there are two, doing unrelated things.

## 2. Keyword pre-filter before any LLM call — reused unchanged **(built)**

Broad keyword list (`somatotop, homuncul, topograph, hemispher, ipsilateral, contralateral, lateraliz, lateral, asymmetr, bilateral, unilateral, limb, digit, finger, toe, hand, foot, feet, tongue, orofacial, articulat, effector, gradient, selectiv`), deliberately broader than the narrow 5-concept keyword list used for phrase-to-concept mapping. Measured against the real 879-chunk corpus: broad list admits 432 chunks (49.1%), narrow list admits 301 (34.2%) — 136 chunks caught only by the broader filter. Reduces LLM calls before the expensive step, no change needed for v2.

## 3. Claim extraction, concept mapping, CAV/TCAV verification — reused unchanged **(built)**

`build_concept_extraction_prompt` run against `mlx-community/Llama-3.2-3B-Instruct-4bit` (`DEFAULT_LOCAL_LLM`, via `mlx_lm`), 3x per surviving chunk. A claim is kept only if it recurs in ≥2 of 3 repeats, checked at the concept level rather than exact-phrase match (phrasing varies run to run even when the underlying claim doesn't). This is the "claims need to be relevant" requirement, operationalized as self-consistency rather than trusting a single LLM call.

Mapped phrases become CAV/TCAV probes via keyword match (closed vocabulary, `concepts.py`) or Case 2's open-vocabulary embedding route (`concepts_case2.py::open_vocabulary_concept_direction`) for phrases outside the fixed 8-concept dictionary.

**Output of this step, per claim**: a concept (or open-vocabulary direction) plus a stance (supports/contradicts/unrelated) plus which chunk(s) it came from.

## 4. Pick the best-aligned representation, run attribution, generate a query — the actual new piece

This is where v2 diverges from what exists. Half of the mechanism is already built, aimed at a different starting point:

| Piece | Status | Where |
|---|---|---|
| 4-method attribution consensus (Saliency/IG/Shapley/LIME) on one decoded window | **built** | `infer_rsn_attribution`, `pipeline.py` |
| Turn attribution consensus into a natural-language query | **built** | `build_query_text`, `pipeline.py` |
| Pick the highest-TCAV concept from a CAV loop, build a concept-steered follow-up query | **built** | `build_refined_query`, `pipeline.py` |
| Run that as a second retrieval round | **built** | `explain_decoded_window_with_query_refinement`, `pipeline.py` |
| Test a claim's TCAV score against *all 6* case×architecture representations, not just one | **new** | none yet — needs the 30-resample checkpoints from `results/case{1,2,3}_bootstrap_30resamples.json` loaded and probed per representation |
| Pick the representation that actually depends on the claim's concept (highest TCAV, or a population-level version using all 30 resamples per representation) | **new** | none yet |
| Run attribution *without* a single decode | **new**, open design question below | none yet |

**Open design question, not resolved**: `infer_rsn_attribution` takes a concrete input window `x`. A corpus-mined claim has no associated decode to attribute. Two candidate resolutions:

- **(a) Canonical window**: the single highest-confidence held-out window of the class the claim concerns.
- **(b) Averaged attribution**: run attribution over many held-out windows of that class and average the per-network scores.

**Recommendation: (b).** A single canonical window reintroduces exactly the single-example fragility that CAV/TCAV's held-out-set averaging (§6.2 of the interview-prep doc, step 6: TCAV score = fraction of held-out examples with positive directional derivative) was built specifically to avoid. Using one example for the query-generating attribution step while using many examples for the CAV/TCAV verification step would be an inconsistency worth avoiding on principle, not just for elegance.

## 5. Second, targeted retrieval pass over the filtered corpus — reused unchanged **(built)**

Standard retrieve-then-rerank pattern: MiniLM bi-encoder cosine-similarity search over the already-keyword-filtered chunks, then `cross-encoder/ms-marco-MiniLM-L-6-v2` (`RERANKER_MODEL_NAME`) reranks the narrowed candidate set by scoring (query, chunk) pairs jointly — a fresh forward pass per pair, too expensive over the whole corpus, so applied only after the bi-encoder narrows the field.

**Third distinct model, worth keeping straight**: this reranker is also MiniLM-family but a separately-trained cross-encoder checkpoint — not the same weights as either the retrieval bi-encoder (§1) or Case 2's frozen prototype encoder. Three MiniLM-family models total in this system, each doing a different job.

## 6. Deterministic synthesis — carries the one real risk in this design

`build_final_synthesis_prompt` / `build_final_synthesis_prompt_case2` already exist. LoRA fine-tuning was already tried once, specifically on the verdict-judgment failure: it fixed output-format compliance but **did not fix the underlying reasoning**, reproduced across 2 independent runs.

**This step is exactly where the measured sycophancy bug lived**: an earlier free-judgment version of the verdict step defaulted to AGREE in 10 of 12 real cases regardless of the actual TCAV score. If step 6 fine-tunes Llama to freely synthesize across (claim, TCAV score, which representation won §4's alignment check, second-pass evidence), the same failure mode is a live risk at a larger scale, not a hypothetical one.

**Required design constraint, not optional**: compute any verdict-like conclusion deterministically from the numbers already in hand — exactly the rule already forced on the simpler loop (`expected_verdict_from_stance_and_tcav`, `pipeline.py`). Restrict the LLM, fine-tuned or not, to narrating an already-decided conclusion. Never let it freely weigh the combined evidence itself.

**This constraint must be applied uniformly regardless of which representation won §4's alignment check.** A claim's best-fit representation could turn out to be Case 1's — and Case 1's loop currently still uses the older free-judgment design (§7 of the interview-prep doc, still an open, unfixed gap as of 2026-08-22). Building step 6 as one shared, deterministic synthesis path used for every case — rather than reusing Case 1's and Case 2's existing, divergent synthesis code as-is — would close that standing gap as a side effect. Building it as a thin wrapper that dispatches to whichever case's existing (and still-divergent) synthesis function would not.

## 7. Concrete build list, in an order that keeps each piece independently testable

1. A function that, given a concept/claim, loads the 30-resample checkpoints for all 3 cases × 2 architectures and returns TCAV scores per representation (extends existing per-case CAV code; no new statistical method needed, `cross_class_rank_bootstrap_test` already exists in `concepts.py` for the significance side).
2. A "best-aligned representation" selection rule on top of (1) — simplest version: argmax TCAV; more defensible version: population-level using the existing rank-bootstrap significance test rather than a raw point estimate.
3. An averaged-attribution function: `infer_rsn_attribution` run over N held-out windows of the claim's class, on the winning representation from (2), scores averaged before consensus-network selection.
4. A query template that states what a representation's attribution + the originating literature claim jointly suggest — extending `build_query_text`'s pattern, not replacing it.
5. One shared, deterministic synthesis function used regardless of which case originated the claim — the piece that would retire the Case 1 free-judgment gap if built this way from the start.

## 8. What this does and doesn't fix

**Fixes**: removes the query-first loop's dependence on starting from one specific decode; makes literature-checking claim-driven and cross-paradigm from the first step, using infrastructure (the paired 30-resample checkpoints) that didn't exist when the original query-first loop was designed.

**Doesn't fix on its own**: the corpus is still small (8 papers, 879 chunks) — that ceiling applies to v2 exactly as it applied to v1, no loop redesign changes it. And the Case 1 free-judgment gap only closes if step 6 is deliberately built as one shared deterministic path (§6) rather than dispatching to the two cases' existing, still-divergent synthesis implementations.
