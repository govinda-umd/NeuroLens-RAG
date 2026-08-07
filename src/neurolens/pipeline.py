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
