"""One file: brain signal in, LLM text out.

Ties together the pieces built separately so far into a single callable:

    ROI window -> Transformer decode -> RSN attribution -> template query
    -> retrieval -> structured LLM prompt -> generated text

`generate_fn` is pluggable. No local LLM is installed yet (see
docs/decoded-state-to-text-report.md and the pending hardware discussion),
so `stub_generate` stands in for now — it exercises every other step for
real and makes the missing piece explicit rather than silent. Swap in a
real local-LLM call (Ollama / mlx-lm / llama.cpp) once one is chosen,
without changing anything upstream of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn

import re

from .concepts import explain_literature_concept, map_phrase_to_known_concept
from .concepts_case2 import explain_literature_concept_case2
from .interpretability import NETWORK_NAMES, compare_methods
from .retrieval import TextChunk, retrieve_and_rerank, retrieve_chunks

METHODS = ["saliency", "integrated_gradients", "shapley", "lime"]
GRADIENT_METHODS = ["saliency", "integrated_gradients"]
PERTURBATION_METHODS = ["shapley", "lime"]


def decode_window(
    model: nn.Module, x: torch.Tensor, device: torch.device
) -> dict:
    """x: [1, L, n_rois]. Returns predicted class, confidence, and HRF prediction."""
    model.eval()
    with torch.no_grad():
        logits, hrf_pred = model(x.to(device))
        probs = torch.softmax(logits, dim=1)
    pred_class = int(probs.argmax(dim=1).item())
    return {
        "pred_class": pred_class,
        "confidence": float(probs[0, pred_class].item()),
        "probs": probs[0].cpu().numpy().tolist(),
        "hrf_pred": None if hrf_pred is None else hrf_pred[0].detach().cpu().numpy().tolist(),
    }


def infer_rsn_attribution(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    network_indices: dict[str, np.ndarray],
    device: torch.device,
    lime_num_samples: int = 300,
) -> dict:
    """Runs all four interpretability methods; flags gradient-vs-perturbation
    disagreement rather than silently picking one (see
    docs/interpretability-methods-notes.md §2 — the two families disagree
    ~35% of the time on the top network)."""
    result = compare_methods(model, x, target_class, network_indices, device, lime_num_samples)
    top_by_method = {
        method: NETWORK_NAMES[int(np.argmax(result[method]["normalized"]))] for method in METHODS
    }
    gradient_top = {top_by_method[m] for m in GRADIENT_METHODS}
    perturbation_top = {top_by_method[m] for m in PERTURBATION_METHODS}
    families_agree = gradient_top == perturbation_top

    return {
        "per_method": {m: result[m]["normalized"].tolist() for m in METHODS},
        "top_network_by_method": top_by_method,
        "families_agree": families_agree,
        "consensus_network": (
            top_by_method["shapley"] if not families_agree else next(iter(gradient_top))
        ),
    }


def build_query_text(
    condition: str,
    confidence: float,
    rsn_attribution: dict,
    subject_id: str,
    task: str,
    run: str,
) -> str:
    """Level 0+1 template, per docs/decoded-state-to-text-report.md."""
    caveat = ""
    if not rsn_attribution["families_agree"]:
        caveat = (
            f" (gradient-based methods instead point to "
            f"{rsn_attribution['top_network_by_method']['saliency']})"
        )
    return (
        f"Decoded condition: {condition} (confidence {confidence:.0%}). "
        f"Primary contributing resting-state network: {rsn_attribution['consensus_network']}"
        f"{caveat}. {task} task, subject {subject_id}, run {run}."
    )


def build_llm_prompt(query_text: str, retrieved: list[dict]) -> str:
    """Structured stance-labeling prompt: forces grounding to a cited excerpt
    ID per claim, and an explicit 'not discussed' fallback instead of
    fabricating a connection when nothing is closely relevant (see
    docs/decoded-state-to-text-report.md Level 2)."""
    excerpt_block = "\n\n".join(
        f"[Excerpt {i + 1}] (source: {r['source_file']}, page {r['page']}, "
        f"similarity={r['score']:.2f})\n{r['text']}"
        for i, r in enumerate(retrieved)
    )
    return f"""You are helping a neuroscience researcher interpret a decoded brain-activity result using retrieved literature excerpts.

DECODED RESULT:
{query_text}

RETRIEVED EXCERPTS:
{excerpt_block}

INSTRUCTIONS:
1. For each excerpt, state in one line whether it SUPPORTS, CONTRADICTS, or is UNRELATED to the decoded result, citing the excerpt number.
2. Then write a 2-3 sentence synthesis comparing the decoded result to the literature: does it agree with prior findings, conflict with them, or is it not clearly discussed in the retrieved excerpts?
3. Do not cite anything not present in the excerpts above. If no excerpt is closely related, say so explicitly rather than inventing a connection.
"""


def stub_generate(prompt: str) -> str:
    """Placeholder generate_fn for testing the pipeline without loading an LLM
    (e.g. in unit tests, or when no local LLM is installed)."""
    n_excerpts = prompt.count("[Excerpt")
    return (
        "[STUB — no generate_fn provided] "
        f"Prompt built successfully with {n_excerpts} retrieved excerpts. "
        "Pass a real generate_fn (e.g. make_mlx_generate_fn()) to get an actual synthesis."
    )


DEFAULT_LOCAL_LLM = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def make_mlx_generate_fn(model_name: str = DEFAULT_LOCAL_LLM, max_tokens: int = 400):
    """Loads a local quantized LLM once via mlx-lm (Apple Silicon) and returns
    a generate_fn(prompt) -> str closure for use with explain_decoded_window.
    Loading the model is the expensive part (~seconds); reuse the returned
    closure across many calls rather than re-creating it per window."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(model_name)

    def generate_fn(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return mlx_generate(model, tokenizer, prompt=chat_prompt, max_tokens=max_tokens, verbose=False)

    return generate_fn


def explain_decoded_window(
    *,
    model: nn.Module,
    x: torch.Tensor,
    subject_id: str,
    task: str,
    run: str,
    class_to_condition: dict[str, str],
    network_indices: dict[str, np.ndarray],
    device: torch.device,
    embedding_model: SentenceTransformer,
    corpus_chunks: list[TextChunk],
    corpus_embeddings: np.ndarray,
    generate_fn=stub_generate,
    reranker=None,
    candidate_k: int = 20,
    top_k: int = 8,
) -> dict:
    """The one-file pipeline: ROI window -> decode -> RSN attribution ->
    query -> retrieval (optionally reranked) -> LLM text. Returns every
    intermediate artifact for auditability, not just the final text.

    Pass a `reranker` (from `retrieval.load_reranker()`) to retrieve
    `candidate_k` dense candidates and rerank down to `top_k` with a
    cross-encoder — no training data needed, see
    docs/case1-summary-report.md §10. Omit it to use plain dense retrieval.
    """
    decoded = decode_window(model, x, device)
    condition = class_to_condition[str(decoded["pred_class"])]

    rsn_attribution = infer_rsn_attribution(model, x, decoded["pred_class"], network_indices, device)

    query_text = build_query_text(condition, decoded["confidence"], rsn_attribution, subject_id, task, run)

    if reranker is not None:
        retrieved_df = retrieve_and_rerank(
            query_text,
            model=embedding_model,
            embeddings=corpus_embeddings,
            chunks=corpus_chunks,
            reranker=reranker,
            candidate_k=candidate_k,
            top_k=top_k,
        )
    else:
        retrieved_df = retrieve_chunks(
            query_text,
            model=embedding_model,
            embeddings=corpus_embeddings,
            chunks=corpus_chunks,
            top_k=top_k,
        )
    retrieved = retrieved_df.to_dict(orient="records")

    prompt = build_llm_prompt(query_text, retrieved)
    generated_text = generate_fn(prompt)

    return {
        "subject_id": subject_id,
        "task": task,
        "run": run,
        "decoded": decoded,
        "condition": condition,
        "rsn_attribution": rsn_attribution,
        "query_text": query_text,
        "retrieved": retrieved,
        "prompt": prompt,
        "generated_text": generated_text,
    }


# --- The RAG <-> CAV loop (docs/interpretability-methods-notes.md §4.1) ---
# decode -> RSN attribution -> retrieve literature -> LLM extracts a
# concept phrase from a supporting excerpt -> map it to a testable concept
# -> fit a CAV and test the model's actual sensitivity -> feed the result
# back into a final synthesis that integrates decode + literature + concept
# test, rather than treating retrieval as a bolt-on citation lookup.


def build_concept_extraction_prompt(excerpt_text: str) -> str:
    return f"""Read this neuroscience literature excerpt. If it makes a specific, testable claim about which body part, side of the body, or type of movement is neurally represented (for example: "hand movement is contralateral", "tongue representation is bilateral"), state that claim as a short phrase (5-10 words). If it makes no such specific claim, respond with exactly: NONE

EXCERPT:
{excerpt_text}

CONCEPT PHRASE (or NONE):"""


def extract_concept_phrases(retrieved: list[dict], generate_fn) -> list[dict]:
    """Runs the concept-extraction prompt over every retrieved excerpt;
    keeps only genuine short phrases, discards NONE / degenerate responses."""
    phrases = []
    for r in retrieved:
        response = generate_fn(build_concept_extraction_prompt(r["text"])).strip()
        if response.upper().startswith("NONE") or len(response) == 0 or len(response) > 150:
            continue
        phrases.append({"source_file": r["source_file"], "page": r["page"], "phrase": response})
    return phrases


def run_cav_loop(
    model: nn.Module,
    cav_train_loader,
    cav_test_loader,
    device: torch.device,
    class_names: list[str],
    target_class: int,
    phrases: list[dict],
) -> list[dict]:
    """For each extracted phrase, map it to a known testable concept and
    run the CAV/TCAV test against the decoded class. Deduplicates by
    concept so the same CAV isn't refit twice in one call."""
    results = []
    seen_concepts: set[str] = set()
    for p in phrases:
        matched = map_phrase_to_known_concept(p["phrase"])
        if matched is None:
            results.append({**p, "matched_concepts": None, "results": None})
            continue
        matched_list = matched if isinstance(matched, list) else [matched]
        new_concepts = [c for c in matched_list if c not in seen_concepts]
        if not new_concepts:
            continue
        seen_concepts.update(new_concepts)
        explanation = explain_literature_concept(
            model, cav_train_loader, cav_test_loader, device, class_names, p["phrase"], target_class
        )
        results.append({**p, **explanation})
    return results


def build_final_synthesis_prompt(query_text: str, stance_text: str, cav_results: list[dict]) -> str:
    tested = [r for r in cav_results if r.get("matched_concepts")]
    if tested:
        cav_block = "\n".join(
            f"- Concept phrase '{r['phrase']}' (from {r['source_file']}) maps to: "
            + ", ".join(
                f"{concept} (TCAV sensitivity for the decoded class = {vals['tcav_score_for_decoded_class']:.2f})"
                for concept, vals in r["results"].items()
            )
            for r in tested
        )
    else:
        cav_block = "No literature-derived concept phrases could be mapped to a testable concept."

    return f"""You previously analyzed retrieved literature excerpts for this decoded brain-activity result:

{query_text}

Your excerpt-by-excerpt stance analysis:
{stance_text}

You then independently tested whether the model's decision is actually sensitive to the concepts the literature invokes, using Concept Activation Vectors (a linear-probe technique applied directly to the model's internal representation, not just reading the literature):
{cav_block}

Write a short (3-4 sentence) researcher-facing synthesis integrating all three lines of evidence: the decoded result itself, what the retrieved literature claims, and whether your own concept-sensitivity test agrees with the literature's claim. Be explicit about agreement or disagreement between the literature and the concept test — that comparison is the most scientifically interesting part, not a restatement of either alone."""


def explain_decoded_window_with_cav_loop(
    *,
    model: nn.Module,
    x: torch.Tensor,
    subject_id: str,
    task: str,
    run: str,
    class_to_condition: dict[str, str],
    network_indices: dict[str, np.ndarray],
    device: torch.device,
    embedding_model: SentenceTransformer,
    corpus_chunks: list[TextChunk],
    corpus_embeddings: np.ndarray,
    cav_train_loader,
    cav_test_loader,
    generate_fn=stub_generate,
    reranker=None,
    candidate_k: int = 20,
    top_k: int = 8,
) -> dict:
    """The complete loop: everything `explain_decoded_window` does, plus
    literature-derived concept extraction, CAV testing, and a final
    synthesis that integrates decode + attribution + literature + concept
    test into one researcher-facing paragraph."""
    base = explain_decoded_window(
        model=model, x=x, subject_id=subject_id, task=task, run=run,
        class_to_condition=class_to_condition, network_indices=network_indices, device=device,
        embedding_model=embedding_model, corpus_chunks=corpus_chunks, corpus_embeddings=corpus_embeddings,
        generate_fn=generate_fn, reranker=reranker, candidate_k=candidate_k, top_k=top_k,
    )

    class_names = [class_to_condition[str(c)] for c in range(len(class_to_condition))]
    phrases = extract_concept_phrases(base["retrieved"], generate_fn)
    cav_results = run_cav_loop(
        model, cav_train_loader, cav_test_loader, device, class_names, base["decoded"]["pred_class"], phrases
    )
    final_prompt = build_final_synthesis_prompt(base["query_text"], base["generated_text"], cav_results)
    final_synthesis = generate_fn(final_prompt)

    base["extracted_concept_phrases"] = phrases
    base["cav_loop_results"] = cav_results
    base["final_synthesis_prompt"] = final_prompt
    base["final_synthesis"] = final_synthesis
    return base


# --- RAG-LLM improvement 1: CAV-aware query refinement (no training) ---
# The cheapest of the three improvement ideas discussed. `run_cav_loop`
# already tells us, for every literature-derived concept phrase the LLM
# noticed, how sensitive the model's own decision actually is to it
# (TCAV). The first-pass retrieval query (`build_query_text`) has no way
# to know that in advance, so it only ever searches on the decoded label
# and attributed RSN. This adds a second retrieval round whose query is
# steered toward whichever concept the model turned out to be MOST
# sensitive to - i.e., retrieval gets to react to what the model actually
# uses, not just what it output.


def build_refined_query(base_query_text: str, cav_loop_results: list[dict]) -> dict | None:
    """Picks the highest-TCAV literature-derived concept and builds a
    concept-focused follow-up query. Returns None if nothing was testable
    (so callers can fall back to the first-pass result untouched)."""
    candidates = []
    for r in cav_loop_results:
        if not r.get("matched_concepts"):
            continue
        for concept, vals in r["results"].items():
            score = vals.get("tcav_score_for_decoded_class")
            if score is not None:
                candidates.append((score, concept))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    best_score, best_concept = candidates[0]
    concept_phrase = best_concept.replace("_", " ")
    refined_query = (
        f"{base_query_text} Specifically: neural representation and lateralization of "
        f"{concept_phrase} movement."
    )
    return {"refined_query": refined_query, "steered_by_concept": best_concept, "tcav_score": best_score}


def explain_decoded_window_with_query_refinement(
    *,
    model: nn.Module,
    x: torch.Tensor,
    subject_id: str,
    task: str,
    run: str,
    class_to_condition: dict[str, str],
    network_indices: dict[str, np.ndarray],
    device: torch.device,
    embedding_model: SentenceTransformer,
    corpus_chunks: list[TextChunk],
    corpus_embeddings: np.ndarray,
    cav_train_loader,
    cav_test_loader,
    generate_fn=stub_generate,
    reranker=None,
    candidate_k: int = 20,
    top_k: int = 8,
) -> dict:
    """Runs the existing CAV-RAG loop to find the highest-TCAV concept, then
    issues a SECOND, concept-steered retrieval round and re-synthesizes
    from those excerpts instead of the generic first-pass ones. No model
    training involved - the only new ingredient is using a signal the
    pipeline already computes (TCAV) to decide what to search for next."""
    base = explain_decoded_window_with_cav_loop(
        model=model, x=x, subject_id=subject_id, task=task, run=run,
        class_to_condition=class_to_condition, network_indices=network_indices, device=device,
        embedding_model=embedding_model, corpus_chunks=corpus_chunks, corpus_embeddings=corpus_embeddings,
        cav_train_loader=cav_train_loader, cav_test_loader=cav_test_loader,
        generate_fn=generate_fn, reranker=reranker, candidate_k=candidate_k, top_k=top_k,
    )

    refinement = build_refined_query(base["query_text"], base["cav_loop_results"])
    if refinement is None:
        base["refinement"] = None
        return base

    if reranker is not None:
        refined_df = retrieve_and_rerank(
            refinement["refined_query"], model=embedding_model, embeddings=corpus_embeddings,
            chunks=corpus_chunks, reranker=reranker, candidate_k=candidate_k, top_k=top_k,
        )
    else:
        refined_df = retrieve_chunks(
            refinement["refined_query"], model=embedding_model, embeddings=corpus_embeddings,
            chunks=corpus_chunks, top_k=top_k,
        )
    refined_retrieved = refined_df.to_dict(orient="records")
    refined_prompt = build_llm_prompt(refinement["refined_query"], refined_retrieved)
    refined_stance_text = generate_fn(refined_prompt)
    refined_final_prompt = build_final_synthesis_prompt(
        refinement["refined_query"], refined_stance_text, base["cav_loop_results"]
    )
    refined_final_synthesis = generate_fn(refined_final_prompt)

    base["refinement"] = {
        **refinement,
        "refined_retrieved": refined_retrieved,
        "refined_stance_text": refined_stance_text,
        "refined_final_synthesis": refined_final_synthesis,
        "top_excerpt_before": base["retrieved"][0] if base["retrieved"] else None,
        "top_excerpt_after": refined_retrieved[0] if refined_retrieved else None,
    }
    return base


# --- Case 2: the CAV-RAG loop for the contrastive brain-text model ---
# Case 2 has no per-network attribution built yet (compare_methods() targets
# a classifier logit + optional HRF head, a different signature than the
# contrastive model's similarity logits) — query text here is coarser than
# Case 1's (condition + confidence only, no RSN mention). Everything else
# mirrors Case 1's loop, except the concept direction comes from text
# alone (concepts_case2.py) instead of a labeled-brain-example probe.


def decode_window_case2(contrastive_model: nn.Module, x: torch.Tensor, device: torch.device) -> dict:
    """x: [1, L, n_rois]. Decodes via prototype classification (argmax
    cosine similarity to the 6 text prototypes), not a classifier head."""
    contrastive_model.eval()
    with torch.no_grad():
        z_brain = contrastive_model.encode_brain(x.to(device))
        z_text = contrastive_model.text_encoder()
        temperature = contrastive_model.log_temperature.exp().clamp(max=100)
        logits = z_brain @ z_text.T * temperature
        probs = torch.softmax(logits, dim=1)
    pred_class = int(probs.argmax(dim=1).item())
    return {
        "pred_class": pred_class,
        "confidence": float(probs[0, pred_class].item()),
        "probs": probs[0].cpu().numpy().tolist(),
    }


def build_query_text_case2(condition: str, confidence: float, subject_id: str, task: str, run: str) -> str:
    return (
        f"Decoded condition: {condition} (confidence {confidence:.0%}), via brain-text "
        f"joint-embedding prototype classification. {task} task, subject {subject_id}, run {run}."
    )


def run_cav_loop_case2(
    contrastive_model: nn.Module,
    cav_test_loader,
    device: torch.device,
    class_names: list[str],
    target_class: int,
    phrases: list[dict],
) -> list[dict]:
    """Case2 analogue of `run_cav_loop` — no cav_train_loader, since the
    concept direction is derived from text prototypes, not fit on brain
    examples."""
    results = []
    seen_concepts: set[str] = set()
    for p in phrases:
        matched = map_phrase_to_known_concept(p["phrase"])
        if matched is None:
            results.append({**p, "matched_concepts": None, "results": None})
            continue
        matched_list = matched if isinstance(matched, list) else [matched]
        new_concepts = [c for c in matched_list if c not in seen_concepts]
        if not new_concepts:
            continue
        seen_concepts.update(new_concepts)
        explanation = explain_literature_concept_case2(
            contrastive_model, cav_test_loader, device, class_names, p["phrase"], target_class
        )
        results.append({**p, **explanation})
    return results


def build_final_synthesis_prompt_case2(query_text: str, stance_text: str, cav_results: list[dict]) -> str:
    """Same synthesis prompt as Case 1's, plus a required structured verdict
    tag — this is what makes automated faithfulness scoring possible
    (`score_synthesis_faithfulness` below parses it and checks it against
    the TCAV number independently, without re-reading the LLM's prose)."""
    tested = [r for r in cav_results if r.get("matched_concepts")]
    if tested:
        cav_block = "\n".join(
            f"- Concept phrase '{r['phrase']}' (from {r['source_file']}) maps to: "
            + ", ".join(
                f"{concept} (TCAV sensitivity for the decoded class = {vals['tcav_score_for_decoded_class']:.2f})"
                for concept, vals in r["results"].items()
            )
            for r in tested
        )
    else:
        cav_block = "No literature-derived concept phrases could be mapped to a testable concept."

    return f"""You previously analyzed retrieved literature excerpts for this decoded brain-activity result:

{query_text}

Your excerpt-by-excerpt stance analysis:
{stance_text}

You then independently tested whether the model's decision is actually sensitive to the concepts the literature invokes, using Concept Activation Vectors derived directly from the model's own brain-text embedding space:
{cav_block}

Write a short (3-4 sentence) researcher-facing synthesis integrating all three lines of evidence: the decoded result itself, what the retrieved literature claims, and whether your own concept-sensitivity test agrees with the literature's claim. Be explicit about agreement or disagreement between the literature and the concept test.

End your response with exactly one final line, formatted precisely as:
VERDICT: AGREE
or
VERDICT: DISAGREE
or
VERDICT: UNCLEAR
where AGREE means the concept test and the literature claim point the same direction, DISAGREE means they conflict, and UNCLEAR means the evidence is too thin to say either way."""


def parse_verdict_tag(text: str) -> str | None:
    match = re.search(r"VERDICT:\s*(AGREE|DISAGREE|UNCLEAR)", text.upper())
    return match.group(1) if match else None


def expected_verdict_from_tcav(tcav_score: float | None, high: float = 0.7, low: float = 0.3) -> str:
    """Deterministic, LLM-independent ground truth for faithfulness scoring:
    a fixed threshold rule applied to the actual TCAV number, so the LLM's
    self-reported verdict can be checked against something it didn't
    generate itself."""
    if tcav_score is None:
        return "UNCLEAR"
    if tcav_score >= high:
        return "AGREE"
    if tcav_score <= low:
        return "DISAGREE"
    return "UNCLEAR"


def score_synthesis_faithfulness(cav_results: list[dict], final_synthesis: str) -> dict:
    """Compares the LLM's self-reported VERDICT tag against a verdict
    computed independently from the actual TCAV score (via a fixed
    threshold rule). When more than one concept was tested (common - the
    same decode often surfaces several literature phrases), it's genuinely
    ambiguous which one a single one-sentence verdict is "about", so this
    reports both a strict bound (must match the FIRST-extracted concept -
    likely an underestimate of faithfulness, since the LLM may reasonably
    be responding to a different tested concept) and a lenient bound (must
    match AT LEAST ONE tested concept - likely an overestimate, since it
    credits coincidental agreement). True faithfulness sits between them;
    reporting the range rather than picking one is the honest choice.
    Returns None fields if no concept was testable at all."""
    tested = [r for r in cav_results if r.get("matched_concepts")]
    if not tested:
        return {
            "llm_verdict": None, "faithful_strict": None, "faithful_lenient": None,
            "all_tested_concepts": None,
        }

    all_tested = [
        (concept, vals["tcav_score_for_decoded_class"], expected_verdict_from_tcav(vals["tcav_score_for_decoded_class"]))
        for r in tested
        for concept, vals in r["results"].items()
    ]
    llm_verdict = parse_verdict_tag(final_synthesis)
    first_expected = all_tested[0][2]
    faithful_strict = None if llm_verdict is None else (llm_verdict == first_expected)
    faithful_lenient = None if llm_verdict is None else any(llm_verdict == e for _, _, e in all_tested)
    return {
        "llm_verdict": llm_verdict,
        "faithful_strict": faithful_strict,
        "faithful_lenient": faithful_lenient,
        "all_tested_concepts": [{"concept": c, "tcav_score": t, "expected_verdict": e} for c, t, e in all_tested],
    }


def explain_decoded_window_case2(
    *,
    contrastive_model: nn.Module,
    x: torch.Tensor,
    subject_id: str,
    task: str,
    run: str,
    class_to_condition: dict[str, str],
    device: torch.device,
    embedding_model: SentenceTransformer,
    corpus_chunks: list[TextChunk],
    corpus_embeddings: np.ndarray,
    generate_fn=stub_generate,
    reranker=None,
    candidate_k: int = 20,
    top_k: int = 8,
) -> dict:
    decoded = decode_window_case2(contrastive_model, x, device)
    condition = class_to_condition[str(decoded["pred_class"])]
    query_text = build_query_text_case2(condition, decoded["confidence"], subject_id, task, run)

    if reranker is not None:
        retrieved_df = retrieve_and_rerank(
            query_text, model=embedding_model, embeddings=corpus_embeddings, chunks=corpus_chunks,
            reranker=reranker, candidate_k=candidate_k, top_k=top_k,
        )
    else:
        retrieved_df = retrieve_chunks(
            query_text, model=embedding_model, embeddings=corpus_embeddings, chunks=corpus_chunks, top_k=top_k,
        )
    retrieved = retrieved_df.to_dict(orient="records")

    prompt = build_llm_prompt(query_text, retrieved)
    generated_text = generate_fn(prompt)

    return {
        "subject_id": subject_id, "task": task, "run": run, "decoded": decoded, "condition": condition,
        "query_text": query_text, "retrieved": retrieved, "prompt": prompt, "generated_text": generated_text,
    }


def explain_decoded_window_with_cav_loop_case2(
    *,
    contrastive_model: nn.Module,
    x: torch.Tensor,
    subject_id: str,
    task: str,
    run: str,
    class_to_condition: dict[str, str],
    device: torch.device,
    embedding_model: SentenceTransformer,
    corpus_chunks: list[TextChunk],
    corpus_embeddings: np.ndarray,
    cav_test_loader,
    generate_fn=stub_generate,
    reranker=None,
    candidate_k: int = 20,
    top_k: int = 8,
) -> dict:
    base = explain_decoded_window_case2(
        contrastive_model=contrastive_model, x=x, subject_id=subject_id, task=task, run=run,
        class_to_condition=class_to_condition, device=device, embedding_model=embedding_model,
        corpus_chunks=corpus_chunks, corpus_embeddings=corpus_embeddings, generate_fn=generate_fn,
        reranker=reranker, candidate_k=candidate_k, top_k=top_k,
    )

    class_names = [class_to_condition[str(c)] for c in range(len(class_to_condition))]
    phrases = extract_concept_phrases(base["retrieved"], generate_fn)
    cav_results = run_cav_loop_case2(
        contrastive_model, cav_test_loader, device, class_names, base["decoded"]["pred_class"], phrases
    )
    final_prompt = build_final_synthesis_prompt_case2(base["query_text"], base["generated_text"], cav_results)
    final_synthesis = generate_fn(final_prompt)
    faithfulness = score_synthesis_faithfulness(cav_results, final_synthesis)

    base["extracted_concept_phrases"] = phrases
    base["cav_loop_results"] = cav_results
    base["final_synthesis_prompt"] = final_prompt
    base["final_synthesis"] = final_synthesis
    base["faithfulness"] = faithfulness
    return base
