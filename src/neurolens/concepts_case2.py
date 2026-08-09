"""Concept Activation Vectors for Case 2's contrastive brain-text model.

Case 1's CAV (`concepts.py`) fits a logistic-regression probe on *labeled
brain examples* to find a concept direction, because the classifier there
only ever sees brain-side representations — there is no other space to
derive a direction from.

Case 2's brain and text encoders share one embedding space, so a concept
direction can instead be built directly from *text* — the difference
between two condition prototypes' embeddings, e.g.
`embed("left hand movement") - embed("right hand movement")` for
`left_side`. No labeled brain examples are needed to define the concept.

The one piece of care this requires: that text-side direction lives in the
64-dim shared embedding space (post `brain_projection`), while the
directional-derivative test needs to operate in the brain backbone's
128-dim hidden space `h` (pre-projection) — the same space Case 1's TCAV
uses, and the space in which the model's actual nonlinear behavior
(projection -> normalize -> bilinear similarity) lives. `brain_projection`
is linear (`h -> z`), so its transpose pulls a z-space direction back into
h-space (the adjoint of a linear map) without needing any additional
fitting. See docs/project-summary.md for the full design rationale.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Same 5 concepts as Case 1's CONCEPT_DEFINITIONS, defined identically by
# class-index sets so results are directly comparable across cases.
CASE2_CONCEPT_DEFINITIONS: dict[str, tuple[set[int], set[int]]] = {
    "hand": ({1, 2}, {0, 3, 4, 5}),
    "foot": ({3, 4}, {0, 1, 2, 5}),
    "tongue": ({5}, {0, 1, 2, 3, 4}),
    "right_side": ({2, 4}, {1, 3}),
    "left_side": ({1, 3}, {2, 4}),
}


def extract_pooled_features_case2(
    contrastive_model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Runs contrastive_model.brain_backbone.forward_features over every
    window. Returns (features [N, backbone_dim], true_labels [N]) — the
    pre-projection representation, matching where Case 1's TCAV operates."""
    contrastive_model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            features = contrastive_model.brain_backbone.forward_features(x)
            all_features.append(features.cpu().numpy())
            all_labels.append(batch["y"].numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


def text_concept_direction(
    contrastive_model: nn.Module,
    positive_classes: set[int],
    negative_classes: set[int],
) -> np.ndarray:
    """The Case2-native concept direction: mean text-embedding of the
    positive classes minus mean of the negative classes, in the 64-dim
    shared space, pulled back into the brain backbone's 128-dim h-space via
    `brain_projection.weight.T` (the adjoint of the linear projection
    h -> z). Returns a unit vector in h-space, comparable in role (not
    value) to Case 1's `train_cav()["direction"]`."""
    with torch.no_grad():
        z_text = contrastive_model.text_encoder()  # [num_classes, embed_dim], already unit-norm
        positive_idx = sorted(positive_classes)
        negative_idx = sorted(negative_classes)
        v_z = z_text[positive_idx].mean(dim=0) - z_text[negative_idx].mean(dim=0)
        v_z = v_z / v_z.norm()

        weight = contrastive_model.brain_projection.weight  # [embed_dim, backbone_dim]
        v_h = weight.T @ v_z  # pullback: backbone_dim
        v_h = v_h / v_h.norm()
    return v_h.cpu().numpy()


def case2_tcav_score(
    contrastive_model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    cav_direction: np.ndarray,
    device: torch.device,
) -> dict:
    """Directional derivative of the decoded class's similarity logit with
    respect to h, along `cav_direction`, evaluated through the model's real
    forward math (brain_projection -> normalize -> bilinear similarity with
    the fixed text prototypes) — mechanically identical in spirit to Case
    1's `tcav_score`, just through a different head."""
    class_mask = labels == target_class
    if class_mask.sum() == 0:
        return {"tcav_score": None, "n_examples": 0}

    h = torch.tensor(features[class_mask], dtype=torch.float32, device=device, requires_grad=True)
    z_pre = contrastive_model.brain_projection(h)
    z_brain = nn.functional.normalize(z_pre, dim=-1)
    z_text = contrastive_model.text_encoder()
    temp = contrastive_model.log_temperature.exp().clamp(max=100)
    logits = z_brain @ z_text.T * temp
    logits[:, target_class].sum().backward()
    grads = h.grad.detach().cpu().numpy()

    sensitivities = grads @ cav_direction
    return {
        "tcav_score": float((sensitivities > 0).mean()),
        "n_examples": int(class_mask.sum()),
    }


def run_case2_concept_analysis(
    contrastive_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    concepts: dict[str, tuple[set[int], set[int]]] = CASE2_CONCEPT_DEFINITIONS,
) -> dict:
    """End-to-end: derive each concept's direction from text alone, score
    TCAV sensitivity for every class on test brain features. No train_loader
    needed — unlike Case 1, the concept directions don't come from brain
    examples at all."""
    test_features, test_labels = extract_pooled_features_case2(contrastive_model, test_loader, device)

    results = {}
    for concept_name, (positive, negative) in concepts.items():
        direction = text_concept_direction(contrastive_model, positive, negative)
        scores = {}
        for class_idx, class_name in enumerate(class_names):
            result = case2_tcav_score(contrastive_model, test_features, test_labels, class_idx, direction, device)
            scores[class_name] = result["tcav_score"]
        results[concept_name] = {"scores": scores}
    return results


# --- Literature-derived concepts, Case 2 flavor ---
# Reuses Case 1's keyword matcher (concept vocabulary is identical) but
# skips the logistic-regression fit entirely.

from .concepts import map_phrase_to_known_concept  # noqa: E402


def explain_literature_concept_case2(
    contrastive_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    phrase: str,
    target_class: int,
) -> dict:
    matched = map_phrase_to_known_concept(phrase)
    if matched is None:
        return {"phrase": phrase, "matched_concepts": None, "results": None}

    matched_concepts = matched if isinstance(matched, list) else [matched]
    test_features, test_labels = extract_pooled_features_case2(contrastive_model, test_loader, device)

    results = {}
    for concept_name in matched_concepts:
        positive, negative = CASE2_CONCEPT_DEFINITIONS[concept_name]
        direction = text_concept_direction(contrastive_model, positive, negative)
        tcav = case2_tcav_score(contrastive_model, test_features, test_labels, target_class, direction, device)
        results[concept_name] = {
            "tcav_score_for_decoded_class": tcav["tcav_score"],
            "n_test_examples_of_decoded_class": tcav["n_examples"],
        }
    return {"phrase": phrase, "matched_concepts": matched_concepts, "results": results}
