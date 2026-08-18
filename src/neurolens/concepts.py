"""Concept Activation Vector (CAV) / TCAV testing for Case 1's learned representation.

Implements Kim et al. 2018's TCAV method: define a concept via labeled
positive/negative examples, fit a linear probe in the model's internal
representation to get a Concept Activation Vector, then measure how
sensitive each class's logit is to moving along that direction.

Two tiers:
- Label-derived concepts (`CONCEPT_DEFINITIONS`): built from the true
  movement labels already on disk. Used to validate the TCAV mechanism
  itself against ground truth (see docs/interpretability-methods-notes.md
  §4 and §2.1).
- Literature-derived concepts (`map_phrase_to_known_concept`): closes the
  loop described in docs/interpretability-methods-notes.md §4.1 — an LLM
  extracts a candidate concept phrase from a retrieved, SUPPORTS-labeled
  paper excerpt (see `pipeline.py`), and this module maps that phrase onto
  one of the *already-defined* label-derived concepts via keyword
  matching. This is the pragmatic answer to the previously-open question
  ("how to turn a concept phrase into a labeled example set without heavy
  manual curation"): rather than solving that problem in full generality,
  literature concepts are constrained to the ones we can already build
  labeled examples for from existing task labels. A real limitation
  (not every literature claim maps onto our 5 concepts), but a real,
  working v1 rather than an unsolved todo.
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


# --- Literature-derived concepts: map an LLM-extracted phrase onto a testable concept ---

# Deliberately simple keyword substring matching, not a learned classifier -
# transparent and auditable, which matters since this is the step that
# decides what literature is allowed to claim we can test.
LITERATURE_CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "hand": ["hand", "finger", "digit", "manual"],
    "foot": ["foot", "feet", "toe"],
    "tongue": ["tongue", "orofacial", "lingual"],
    "right_side": ["contralateral", "lateral", "lateraliz", "unilateral", "asymmetr"],
    "left_side": ["contralateral", "lateral", "lateraliz", "unilateral", "asymmetr"],
}


def map_phrase_to_known_concept(phrase: str) -> str | None:
    """Maps a free-text concept phrase (extracted by an LLM from a
    retrieved excerpt) onto one of CONCEPT_DEFINITIONS's keys via keyword
    matching. Returns None if no known concept matches - an honest "we
    can't test this claim yet" rather than a forced, meaningless mapping.

    Laterality phrases match both `right_side` and `left_side` (the
    keywords don't distinguish direction) - `explain_literature_concept`
    tests both and reports both, since the CAV machinery can't yet tell
    "the paper means left" from "the paper means right" without a more
    careful extraction step.
    """
    phrase_lower = phrase.lower()
    matched = [
        concept for concept, keywords in LITERATURE_CONCEPT_KEYWORDS.items()
        if any(kw in phrase_lower for kw in keywords)
    ]
    return matched[0] if len(matched) == 1 else (matched if matched else None)


def explain_literature_concept(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    phrase: str,
    target_class: int,
) -> dict:
    """The RAG->CAV loop's core step: given a literature-extracted phrase
    and the class the pipeline decoded, map the phrase to known concept(s),
    fit each one's CAV, and report TCAV sensitivity for the decoded class
    specifically - "does the model's decision for this class actually
    depend on the concept the retrieved literature invokes?"
    """
    matched = map_phrase_to_known_concept(phrase)
    if matched is None:
        return {"phrase": phrase, "matched_concepts": None, "results": None}

    matched_concepts = matched if isinstance(matched, list) else [matched]
    train_features, train_labels = extract_pooled_features(model, train_loader, device)
    test_features, test_labels = extract_pooled_features(model, test_loader, device)

    results = {}
    for concept_name in matched_concepts:
        positive, negative = CONCEPT_DEFINITIONS[concept_name]
        cav = train_cav(train_features, train_labels, positive, negative)
        tcav = tcav_score(model, test_features, test_labels, target_class, cav["direction"], device)
        results[concept_name] = {
            "probe_accuracy": cav["probe_accuracy"],
            "tcav_score_for_decoded_class": tcav["tcav_score"],
            "n_test_examples_of_decoded_class": tcav["n_examples"],
        }
    return {"phrase": phrase, "matched_concepts": matched_concepts, "results": results}


# --- Statistical significance for TCAV scores ---
#
# See docs/population-level-evaluation-plan.md §6 for the full derivation
# and the dead end that preceded this: a null of TCAV scores from directions
# drawn uniformly at random in the representation's hidden space does NOT
# work here (near-extreme scores for almost any fixed direction, since
# within-class gradients are highly consistent - a property of the
# representation's separability, not of the direction's meaning). What
# works instead is asking a different question: does this *specific,
# already-fitted* direction single out its intended class more than the
# other 5, and does that hold up under resampling? This was validated ad
# hoc for Case 2's open-vocabulary CAV (P=1.000 across 1000 resamples);
# promoted here into a reusable function so Case 1 and Case 3 don't have
# to reimplement it.


def extract_pooled_features_with_subjects(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like `extract_pooled_features`, but also returns each window's
    subject_id so callers can bootstrap resample at the subject level
    (never window level - see docs/population-level-evaluation-plan.md
    §3.3 for why window-level resampling would be invalid: windows from
    the same subject/run are not independent draws)."""
    model.eval()
    all_features, all_labels, all_subjects = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            features = model.forward_features(x)
            all_features.append(features.cpu().numpy())
            all_labels.append(batch["y"].numpy())
            all_subjects.extend(batch["subject_id"])
    return np.concatenate(all_features), np.concatenate(all_labels), np.array(all_subjects)


def cross_class_rank_bootstrap_test(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cav_direction: np.ndarray,
    target_class: int,
    class_names: list[str],
    device: torch.device,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Holds `cav_direction` fixed. On each of `n_resamples` subject-level
    bootstrap resamples of `features`/`labels`, computes the TCAV score
    against every class's own held-out examples and checks whether
    `target_class` ranks #1 of all classes. Returns the target class's
    TCAV score distribution (mean + 95% CI) plus P(target ranks #1) - the
    fraction of resamples where the direction most specifically implicates
    its intended class, not just scores highly for it in isolation.

    TCAV's underlying score is a binarized fraction-positive-gradient, so
    when a representation is highly separable it commonly saturates at
    exactly 1.0 for *several* classes at once under a given direction -
    not just the intended target. Ties must be broken randomly (`rng.choice`
    among all classes tied at the max), never by `max()`'s implicit
    lowest-index-wins rule: that would silently bias P(rank1) toward
    low-index classes regardless of which class the direction actually
    implicates, an artifact rather than a real null finding. `frac_ties_at_max`
    reports how often the target was tied with >=1 other class at the top
    score, so a caller can tell a genuine null apart from a saturated-score
    regime where this rank test has little power to discriminate."""
    rng = np.random.RandomState(seed)
    unique_subjects = np.unique(subject_ids)
    num_classes = len(class_names)
    subject_to_idx = {s: np.where(subject_ids == s)[0] for s in unique_subjects}

    target_scores = []
    rank1_count = 0
    tie_count = 0
    for _ in range(n_resamples):
        resampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        idx = np.concatenate([subject_to_idx[s] for s in resampled_subjects])
        feats_r, labels_r = features[idx], labels[idx]

        class_scores = {}
        for c in range(num_classes):
            result = tcav_score(model, feats_r, labels_r, c, cav_direction, device)
            class_scores[c] = result["tcav_score"] if result["tcav_score"] is not None else -1.0
        target_scores.append(class_scores[target_class])

        top_score = max(class_scores.values())
        tied_classes = [c for c, s in class_scores.items() if s == top_score]
        if len(tied_classes) > 1 and target_class in tied_classes:
            tie_count += 1
        winner = tied_classes[0] if len(tied_classes) == 1 else tied_classes[rng.randint(len(tied_classes))]
        if winner == target_class:
            rank1_count += 1

    target_scores = np.array(target_scores)
    lo, hi = np.percentile(target_scores, [2.5, 97.5])
    return {
        "target_class": class_names[target_class],
        "mean_tcav": float(target_scores.mean()),
        "ci_95": [float(lo), float(hi)],
        "p_rank1": rank1_count / n_resamples,
        "frac_ties_at_max": tie_count / n_resamples,
        "n_resamples": n_resamples,
    }
