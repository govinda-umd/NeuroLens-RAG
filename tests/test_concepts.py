"""CAV/TCAV mechanism (concepts.py) on synthetic, trivially-separable data.

A real trained model isn't needed to test the mechanism itself: train_cav
just needs features + labels, and tcav_score just needs any nn.Module with
a differentiable `.classifier` head, per its documented contract (see
concepts.py -- Case 2/3 reuse this exact function on a post-hoc-fitted
classifier for precisely this reason).
"""

import numpy as np
import torch
from torch import nn

from neurolens.concepts import train_cav, tcav_score, EXTENDED_CONCEPT_DEFINITIONS, CONCEPT_DEFINITIONS


class TinyLinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)


def _synthetic_two_group_features(n_per_class: int = 50, dim: int = 8, seed: int = 0):
    rng = np.random.RandomState(seed)
    # Positive-class features live near +1 on dim 0; negative near -1.
    # Every other dimension is pure noise -- the CAV direction should
    # recover dim 0 as (close to) the separating axis.
    pos = rng.randn(n_per_class, dim) * 0.1
    pos[:, 0] += 1.0
    neg = rng.randn(n_per_class, dim) * 0.1
    neg[:, 0] -= 1.0
    features = np.concatenate([pos, neg]).astype(np.float32)
    labels = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    return features, labels


def test_train_cav_recovers_near_perfect_separator():
    features, labels = _synthetic_two_group_features()
    result = train_cav(features, labels, positive_classes={0}, negative_classes={1})
    assert result["probe_accuracy"] > 0.99
    assert result["n_examples"] == len(labels)
    direction = result["direction"]
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-5  # returned direction is a unit vector
    # The separating axis is dim 0 by construction -- it should dominate.
    assert np.argmax(np.abs(direction)) == 0


def test_tcav_score_is_high_for_aligned_direction():
    features, labels = _synthetic_two_group_features()
    cav = train_cav(features, labels, positive_classes={0}, negative_classes={1})

    model = TinyLinearClassifier(in_dim=features.shape[1], num_classes=2)
    with torch.no_grad():
        # Weight class 0's logit to increase along the true separating axis
        # (dim 0), so its gradient w.r.t. h points the same way the CAV does.
        model.classifier.weight.zero_()
        model.classifier.weight[0, 0] = 1.0

    result = tcav_score(model, features, labels, target_class=0, cav_direction=cav["direction"], device=torch.device("cpu"))
    assert result["n_examples"] > 0
    assert result["tcav_score"] > 0.9  # gradient and CAV direction should agree almost everywhere


def test_extended_concept_definitions_is_a_strict_superset():
    assert set(CONCEPT_DEFINITIONS.keys()).issubset(EXTENDED_CONCEPT_DEFINITIONS.keys())
    for name, definition in CONCEPT_DEFINITIONS.items():
        assert EXTENDED_CONCEPT_DEFINITIONS[name] == definition


def test_concept_definitions_partition_disjoint_classes():
    # Every concept's positive and negative class sets must be disjoint --
    # a class can't simultaneously be evidence for and against a concept.
    for name, (positive, negative) in EXTENDED_CONCEPT_DEFINITIONS.items():
        assert positive.isdisjoint(negative), f"{name} has overlapping positive/negative classes"
