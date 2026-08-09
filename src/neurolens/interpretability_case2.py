"""RSN attribution for Case 2, generalized to arbitrary free text.

Case 1's attribution (`interpretability.py`) explains one of 6 fixed
classifier logits. Case 2 has no classifier, but its brain-text similarity
score plays exactly the same role a class logit does, with one added
capability Case 1 structurally cannot offer: the "class" doesn't have to be
one of the 6 known conditions. Any sentence — a literature-derived phrase
extracted by the RAG-CAV loop, say — can be embedded through the same
frozen MiniLM + trained projection Case 2 already uses (see
`embed_text_query`), and attribution can then explain which resting-state
network makes THIS specific window's brain embedding align with THAT
specific sentence. This is a local (per-window, per-network) complement to
CAV/TCAV's global (direction-based, aggregated-over-windows) test of the
same literature-derived concept.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency
from lime.lime_tabular import LimeTabularExplainer
from sentence_transformers import SentenceTransformer
from torch import nn

from .interpretability import NETWORK_NAMES, aggregate_attribution_to_networks, normalize


def embed_text_query(
    text: str, embedding_model: SentenceTransformer, contrastive_model: nn.Module
) -> torch.Tensor:
    """Embeds ARBITRARY text through the same frozen-MiniLM + trained-projection
    path Case 2 used for its 6 known conditions, landing it in the shared
    brain-text space even though it was never one of the training sentences.
    Returns a unit vector, shape [embed_dim]."""
    raw = embedding_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    device = next(contrastive_model.text_encoder.projection.parameters()).device
    raw_t = torch.tensor(raw, dtype=torch.float32, device=device)
    with torch.no_grad():
        projected = contrastive_model.text_encoder.projection(raw_t)
        z_text_query = nn.functional.normalize(projected, dim=0)
    return z_text_query.cpu()


class Case2TextQueryWrapper(nn.Module):
    """Wraps a ContrastiveModel + one fixed query-text embedding into a
    single-output similarity model, so gradient-based Captum methods (which
    expect model(x) -> a tensor to index into) work unchanged."""

    def __init__(self, contrastive_model: nn.Module, z_text_query: torch.Tensor):
        super().__init__()
        self.contrastive_model = contrastive_model
        self.register_buffer("z_text_query", z_text_query)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_brain = self.contrastive_model.encode_brain(x)
        return (z_brain @ self.z_text_query).unsqueeze(-1)  # [B, 1]


def saliency_attribution_text(wrapper: Case2TextQueryWrapper, x: torch.Tensor) -> np.ndarray:
    method = Saliency(lambda inp: wrapper(inp))
    attr = method.attribute(x, target=0)
    return attr.squeeze(0).detach().cpu().numpy()


def integrated_gradients_attribution_text(
    wrapper: Case2TextQueryWrapper, x: torch.Tensor, n_steps: int = 50
) -> np.ndarray:
    method = IntegratedGradients(lambda inp: wrapper(inp))
    baseline = torch.zeros_like(x)
    attr = method.attribute(x, baselines=baseline, target=0, n_steps=n_steps)
    return attr.squeeze(0).detach().cpu().numpy()


def _make_masked_similarity_fn(
    contrastive_model: nn.Module,
    z_text_query: torch.Tensor,
    x: torch.Tensor,
    network_indices: dict[str, np.ndarray],
    device: torch.device,
):
    """Same network-ablation mechanic as Case 1's `_make_masked_predict_fn`,
    but returns the raw similarity score (a regression target) instead of a
    softmax class probability, since there's no class here — just "how well
    does this window match this text."""
    x = x.to(device)
    z_text_query = z_text_query.to(device)

    def predict_fn(mask_batch: np.ndarray) -> np.ndarray:
        mask_batch = np.atleast_2d(mask_batch)
        n = mask_batch.shape[0]
        x_batch = x.repeat(n, 1, 1).clone()
        for row, mask in enumerate(mask_batch):
            for net_idx, net_name in enumerate(NETWORK_NAMES):
                if mask[net_idx] == 0:
                    x_batch[row, :, network_indices[net_name]] = 0.0
        with torch.no_grad():
            z_brain = contrastive_model.encode_brain(x_batch)
            sim = (z_brain @ z_text_query).cpu().numpy()
        return sim  # [n]

    return predict_fn


def exact_shapley_networks_text(
    contrastive_model: nn.Module,
    z_text_query: torch.Tensor,
    x: torch.Tensor,
    network_indices: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Exact Shapley values (128 coalitions) over the similarity score
    instead of a class probability — mechanically identical to Case 1's
    `exact_shapley_networks`, just a different value function."""
    predict_fn = _make_masked_similarity_fn(contrastive_model, z_text_query, x, network_indices, device)
    n = len(NETWORK_NAMES)
    all_masks = np.array(list(itertools.product([0, 1], repeat=n)))
    values = predict_fn(all_masks)
    value_lookup = {tuple(mask): v for mask, v in zip(all_masks, values)}

    phi = np.zeros(n)
    players = list(range(n))
    for i in players:
        others = [p for p in players if p != i]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                s = len(subset)
                weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                mask_s = np.zeros(n, dtype=int)
                for j in subset:
                    mask_s[j] = 1
                mask_si = mask_s.copy()
                mask_si[i] = 1
                phi[i] += weight * (value_lookup[tuple(mask_si)] - value_lookup[tuple(mask_s)])
    return phi


def lime_networks_text(
    contrastive_model: nn.Module,
    z_text_query: torch.Tensor,
    x: torch.Tensor,
    network_indices: dict[str, np.ndarray],
    device: torch.device,
    num_samples: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """LIME in regression mode (the target is a continuous similarity score,
    not a class probability)."""
    predict_fn = _make_masked_similarity_fn(contrastive_model, z_text_query, x, network_indices, device)
    n = len(NETWORK_NAMES)
    rng = np.random.RandomState(seed)
    background = rng.randint(0, 2, size=(200, n))

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=NETWORK_NAMES,
        categorical_features=list(range(n)),
        discretize_continuous=False,
        mode="regression",
        random_state=seed,
    )
    instance = np.ones(n)
    explanation = explainer.explain_instance(instance, predict_fn, num_features=n, num_samples=num_samples)
    weight_by_index = dict(explanation.local_exp[1])
    return np.array([weight_by_index.get(i, 0.0) for i in range(n)])


def compare_methods_text(
    contrastive_model: nn.Module,
    embedding_model: SentenceTransformer,
    text_query: str,
    x: torch.Tensor,
    network_indices: dict[str, np.ndarray],
    device: torch.device,
    lime_num_samples: int = 300,
) -> dict[str, dict[str, np.ndarray]]:
    """End-to-end: embed `text_query` (any free text, not just the 6 known
    conditions) into the shared space, then run all four attribution
    methods explaining "which RSN made this window's brain embedding align
    with this sentence." Same return shape as Case 1's `compare_methods`."""
    z_text_query = embed_text_query(text_query, embedding_model, contrastive_model)
    wrapper = Case2TextQueryWrapper(contrastive_model, z_text_query).to(device)

    x_dev = x.to(device)
    x_dev.requires_grad_(True)

    saliency_raw = aggregate_attribution_to_networks(saliency_attribution_text(wrapper, x_dev), network_indices)
    ig_raw = aggregate_attribution_to_networks(
        integrated_gradients_attribution_text(wrapper, x_dev), network_indices
    )
    shapley_raw = exact_shapley_networks_text(contrastive_model, z_text_query, x, network_indices, device)
    lime_raw = lime_networks_text(
        contrastive_model, z_text_query, x, network_indices, device, num_samples=lime_num_samples
    )

    results = {
        "saliency": saliency_raw,
        "integrated_gradients": ig_raw,
        "shapley": shapley_raw,
        "lime": lime_raw,
    }
    return {name: {"raw": vec, "normalized": normalize(vec)} for name, vec in results.items()}
