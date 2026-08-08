"""Concept Activation Vector (CAV) / TCAV testing for Case 1's learned representation.

Implements Kim et al. 2018's TCAV method: define a concept via labeled
positive/negative examples, fit a linear probe in the model's internal
representation to get a Concept Activation Vector, then measure how
sensitive each class's logit is to moving along that direction.

v1 scope: label-derived concepts (built from the true movement labels
already on disk), not literature-derived ones — see
docs/interpretability-methods-notes.md §4.1 for why literature-derived
concepts are a harder, still-open problem (turning a retrieved concept
*phrase* into a labeled example set). Label-derived concepts let us
validate the TCAV mechanism itself against ground truth before attempting
that harder version.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

# name -> (positive_classes, negative_classes), by class index (0=baseline,
# 1=left_hand, 2=right_hand, 3=left_foot, 4=right_foot, 5=tongue)
CONCEPT_DEFINITIONS: dict[str, tuple[set[int], set[int]]] = {
    "hand": ({1, 2}, {0, 3, 4, 5}),
    "foot": ({3, 4}, {0, 1, 2, 5}),
    "tongue": ({5}, {0, 1, 2, 3, 4}),
    "right_side": ({2, 4}, {1, 3}),
    "left_side": ({1, 3}, {2, 4}),
}


def extract_pooled_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Runs model.forward_features over every window in `loader`.
    Returns (features [N, hidden_dim], true_labels [N])."""
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            features = model.forward_features(x)
            all_features.append(features.cpu().numpy())
            all_labels.append(batch["y"].numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


def train_cav(
    features: np.ndarray,
    labels: np.ndarray,
    positive_classes: set[int],
    negative_classes: set[int],
) -> dict:
    """Fits a linear probe separating positive vs. negative concept examples.
    Returns the CAV direction (normalized) and the probe's held-in accuracy
    (a sanity check — low accuracy means the concept isn't linearly
    separable in this representation, and the resulting direction shouldn't
    be trusted)."""
    is_positive = np.isin(labels, list(positive_classes))
    is_negative = np.isin(labels, list(negative_classes))
    mask = is_positive | is_negative
    X = features[mask]
    y = is_positive[mask].astype(int)

    probe = LogisticRegression(max_iter=1000)
    probe.fit(X, y)
    accuracy = probe.score(X, y)

    direction = probe.coef_[0]
    direction = direction / np.linalg.norm(direction)
    return {"direction": direction, "probe_accuracy": accuracy, "n_examples": int(mask.sum())}


def tcav_score(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    cav_direction: np.ndarray,
    device: torch.device,
) -> dict:
    """TCAV sensitivity of `target_class`'s logit to the CAV direction,
    computed over test examples whose true label is `target_class`
    (Kim et al. 2018's directional-derivative formulation)."""
    class_mask = labels == target_class
    if class_mask.sum() == 0:
        return {"tcav_score": None, "n_examples": 0}

    h = torch.tensor(features[class_mask], dtype=torch.float32, device=device, requires_grad=True)
    logits = model.classifier(h)
    logits[:, target_class].sum().backward()
    grads = h.grad.detach().cpu().numpy()

    sensitivities = grads @ cav_direction
    return {
        "tcav_score": float((sensitivities > 0).mean()),
        "n_examples": int(class_mask.sum()),
    }


def run_concept_analysis(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    concepts: dict[str, tuple[set[int], set[int]]] = CONCEPT_DEFINITIONS,
) -> dict:
    """End-to-end: fit each concept's CAV on train features, score TCAV
    sensitivity for every class on test features. Returns a nested dict:
    {concept: {"probe_accuracy": ..., "scores": {class_name: tcav_score}}}."""
    train_features, train_labels = extract_pooled_features(model, train_loader, device)
    test_features, test_labels = extract_pooled_features(model, test_loader, device)

    results = {}
    for concept_name, (positive, negative) in concepts.items():
        cav = train_cav(train_features, train_labels, positive, negative)
        scores = {}
        for class_idx, class_name in enumerate(class_names):
            result = tcav_score(model, test_features, test_labels, class_idx, cav["direction"], device)
            scores[class_name] = result["tcav_score"]
        results[concept_name] = {
            "probe_accuracy": cav["probe_accuracy"],
            "n_train_examples": cav["n_examples"],
            "scores": scores,
        }
    return results
