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
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader

from .interpretability_case2 import embed_text_query

# Same 5 concepts as Case 1's CONCEPT_DEFINITIONS, defined identically by
# class-index sets so results are directly comparable across cases.
CASE2_CONCEPT_DEFINITIONS: dict[str, tuple[set[int], set[int]]] = {
    "hand": ({1, 2}, {0, 3, 4, 5}),
    "foot": ({3, 4}, {0, 1, 2, 5}),
    "tongue": ({5}, {0, 1, 2, 3, 4}),
    "right_side": ({2, 4}, {1, 3}),
    "left_side": ({1, 3}, {2, 4}),
}

# Imported (not re-hand-written) so the 3 additional 2026-08-22 concepts
# can't silently drift out of sync between the two modules.
from .concepts import EXTENDED_CONCEPT_DEFINITIONS as CASE2_EXTENDED_CONCEPT_DEFINITIONS  # noqa: E402


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


# --- Open-vocabulary CAV: test ANY phrase, not just the 5 predefined concepts ---
#
# `text_concept_direction` above requires the concept to be expressible as a
# difference between subsets of the 6 known condition classes - which is why
# `explain_literature_concept_case2` can only test phrases that keyword-match
# one of CASE2_CONCEPT_DEFINITIONS's 5 entries, silently dropping everything
# else. Nothing about the underlying mechanism actually requires that: any
# free-text phrase can be embedded through the same frozen-MiniLM +
# trained-projection path (`embed_text_query`, already built for
# text-conditioned attribution) and used as a concept direction directly.
# This is the generalization - genuinely open-vocabulary concept testing,
# not just open-vocabulary extraction. Case 1 cannot follow: its CAV
# direction requires labeled brain examples, which don't exist for a concept
# that was never one of the 6 training labels. Open-vocabulary interpretability
# is a capability the shared embedding space enables, not a generic add-on.


def open_vocabulary_concept_direction(
    contrastive_model: nn.Module,
    embedding_model: SentenceTransformer,
    phrase: str,
    contrast_phrase: str | None = None,
) -> np.ndarray:
    """Concept direction for ANY phrase, no CASE2_CONCEPT_DEFINITIONS
    membership required. If `contrast_phrase` is given, the direction is
    phrase - contrast_phrase (most precise when a natural opposite exists,
    e.g. "left hand" vs "right hand"). Otherwise the direction is phrase
    minus the mean of the 6 known condition prototypes - an implicit
    "generic condition" baseline, still well-defined for a single phrase
    with no natural opposite (e.g. a claim like "bilateral representation"
    extracted from literature that isn't naturally paired with anything).
    Returns a unit vector in h-space, same role as `text_concept_direction`'s
    output."""
    z_phrase = embed_text_query(phrase, embedding_model, contrastive_model)
    if contrast_phrase is not None:
        z_other = embed_text_query(contrast_phrase, embedding_model, contrastive_model)
    else:
        with torch.no_grad():
            z_other = contrastive_model.text_encoder().mean(dim=0).cpu()
            z_other = z_other / z_other.norm()

    v_z = z_phrase - z_other
    v_z = v_z / v_z.norm()

    with torch.no_grad():
        weight = contrastive_model.brain_projection.weight.cpu()  # [embed_dim, backbone_dim]
        v_h = weight.T @ v_z
        v_h = v_h / v_h.norm()
    return v_h.numpy()


def explain_open_vocabulary_concept_case2(
    contrastive_model: nn.Module,
    embedding_model: SentenceTransformer,
    test_loader: DataLoader,
    device: torch.device,
    phrase: str,
    target_class: int,
    contrast_phrase: str | None = None,
) -> dict:
    """Open-vocabulary analogue of `explain_literature_concept_case2` - always
    testable, since it never checks phrase membership against a fixed
    dictionary."""
    direction = open_vocabulary_concept_direction(contrastive_model, embedding_model, phrase, contrast_phrase)
    test_features, test_labels = extract_pooled_features_case2(contrastive_model, test_loader, device)
    tcav = case2_tcav_score(contrastive_model, test_features, test_labels, target_class, direction, device)
    return {
        "phrase": phrase,
        "contrast_phrase": contrast_phrase,
        "tcav_score_for_decoded_class": tcav["tcav_score"],
        "n_test_examples_of_decoded_class": tcav["n_examples"],
    }


# --- Random-direction null (Kim et al. 2018's own recommended significance
# check): a TCAV score alone doesn't say whether the concept direction found
# something real or whether ANY direction in h-space would score similarly
# for this class. Compare the real concept's TCAV score against a null
# distribution built from random unit directions on the SAME held-out
# examples, and report where the real score falls relative to that null.


def random_direction_null_tcav_scores(
    contrastive_model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    device: torch.device,
    n_random: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """TCAV scores for `n_random` random unit directions in h-space, on the
    same held-out examples of `target_class` a real concept would be tested
    against. A meaningless direction should score near 0.5 (a random
    direction is as likely to align with the gradient as not); the spread of
    this distribution is what turns a single real TCAV score into a
    statement with a p-value instead of a bare number."""
    rng = np.random.default_rng(seed)
    backbone_dim = features.shape[1]
    scores = []
    for _ in range(n_random):
        v = rng.normal(size=backbone_dim).astype(np.float32)
        v = v / np.linalg.norm(v)
        result = case2_tcav_score(contrastive_model, features, labels, target_class, v, device)
        if result["tcav_score"] is not None:
            scores.append(result["tcav_score"])
    return np.array(scores)


def empirical_p_value(real_score: float, null_scores: np.ndarray) -> float:
    """Two-sided empirical p-value: fraction of the null distribution at
    least as extreme (as far from 0.5) as the real score."""
    real_extremity = abs(real_score - 0.5)
    null_extremity = np.abs(null_scores - 0.5)
    return float((null_extremity >= real_extremity).mean())
