"""Compare four interpretability methods for inferring which resting-state
network (RSN) drove a single decoded prediction.

Two families, pooled to the same 7-Yeo-network granularity so their outputs
are directly comparable:

- **Gradient-based** (operate on the continuous 300-ROI input directly):
  Saliency and Integrated Gradients, via Captum. Attribution is computed
  per (timestep, ROI), then aggregated by summing |attribution| across the
  32-timestep window and across the ROIs belonging to each network.
- **Perturbation/coalition-based** (operate on a 7-"player" abstraction —
  one player per network, ablate whole networks at once): exact Shapley
  values and LIME. Pooling to 7 networks (instead of 300 ROIs or 32x300
  timepoint-ROI cells) is what makes both tractable here: 2^7 = 128
  coalitions can be enumerated exactly for Shapley (no sampling
  approximation needed), and LIME's local surrogate only has to fit 7
  features instead of thousands.

All four ultimately return a comparable length-7 network attribution
vector (one score per network, in `NETWORK_NAMES` order).
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, Saliency
from lime.lime_tabular import LimeTabularExplainer
from torch import nn

NETWORK_NAMES = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

_ROI_LABEL_RE = re.compile(r"^7Networks_(LH|RH)_([A-Za-z]+)_")


def load_roi_to_network(roi_labels_path: Path) -> np.ndarray:
    """Returns an array of length n_rois with each ROI's network name."""
    labels = pd.read_csv(roi_labels_path, sep="\t")
    networks = []
    for label in labels["roi_label"]:
        match = _ROI_LABEL_RE.match(label)
        if not match:
            raise ValueError(f"Unrecognized ROI label format: {label}")
        networks.append(match.group(2))
    return np.array(networks)


def network_roi_indices(roi_to_network: np.ndarray) -> dict[str, np.ndarray]:
    return {net: np.where(roi_to_network == net)[0] for net in NETWORK_NAMES}


def aggregate_attribution_to_networks(
    attr: np.ndarray, network_indices: dict[str, np.ndarray]
) -> np.ndarray:
    """attr: [window_length, n_rois] elementwise attribution -> [7] per-network magnitude."""
    per_roi = np.abs(attr).sum(axis=0)  # [n_rois]
    return np.array([per_roi[idx].sum() for idx in [network_indices[n] for n in NETWORK_NAMES]])


def normalize(vec: np.ndarray) -> np.ndarray:
    total = np.abs(vec).sum()
    return np.abs(vec) / total if total > 0 else np.zeros_like(vec)


def saliency_attribution(model: nn.Module, x: torch.Tensor, target_class: int) -> np.ndarray:
    """x: [1, L, n_rois]. Returns raw signed attribution [L, n_rois]."""
    forward_fn = lambda inp: model(inp)[0]
    method = Saliency(forward_fn)
    attr = method.attribute(x, target=target_class)
    return attr.squeeze(0).detach().cpu().numpy()


def integrated_gradients_attribution(
    model: nn.Module, x: torch.Tensor, target_class: int, n_steps: int = 50
) -> np.ndarray:
    """x: [1, L, n_rois]. Baseline = all-zero (the z-scored 'no signal' point)."""
    forward_fn = lambda inp: model(inp)[0]
    method = IntegratedGradients(forward_fn)
    baseline = torch.zeros_like(x)
    attr = method.attribute(x, baselines=baseline, target=target_class, n_steps=n_steps)
    return attr.squeeze(0).detach().cpu().numpy()


def _make_masked_predict_fn(
    model: nn.Module,
    x: torch.Tensor,
    network_indices: dict[str, np.ndarray],
    device: torch.device,
):
    """Returns predict_fn(mask_batch[N, 7]) -> softmax probs [N, num_classes].

    mask[j] == 1 means network j's ROI channels are left intact; 0 means
    they're zeroed out (ablated) for the whole window.
    """
    x = x.to(device)

    def predict_fn(mask_batch: np.ndarray) -> np.ndarray:
        mask_batch = np.atleast_2d(mask_batch)
        n = mask_batch.shape[0]
        x_batch = x.repeat(n, 1, 1).clone()  # [n, L, n_rois]
        for row, mask in enumerate(mask_batch):
            for net_idx, net_name in enumerate(NETWORK_NAMES):
                if mask[net_idx] == 0:
                    x_batch[row, :, network_indices[net_name]] = 0.0
        with torch.no_grad():
            logits, _ = model(x_batch)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict_fn


def exact_shapley_networks(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    network_indices: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Exact Shapley values over the 7 networks (128 coalitions enumerated exactly —
    tractable because n=7, unlike ROI- or voxel-level Shapley which needs approximation).
    """
    predict_fn = _make_masked_predict_fn(model, x, network_indices, device)
    n = len(NETWORK_NAMES)
    all_masks = np.array(list(itertools.product([0, 1], repeat=n)))  # [128, 7]
    probs = predict_fn(all_masks)[:, target_class]  # [128]
    value_lookup = {tuple(mask): v for mask, v in zip(all_masks, probs)}

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


def lime_networks(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    network_indices: dict[str, np.ndarray],
    device: torch.device,
    num_samples: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """LIME (Ribeiro et al. 2016) over the same 7-network coalition space:
    fits a locally weighted linear surrogate around the all-networks-present
    input, explaining the predicted class' probability.
    """
    predict_fn = _make_masked_predict_fn(model, x, network_indices, device)
    n = len(NETWORK_NAMES)
    rng = np.random.RandomState(seed)
    background = rng.randint(0, 2, size=(200, n))

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=NETWORK_NAMES,
        categorical_features=list(range(n)),
        discretize_continuous=False,
        mode="classification",
        random_state=seed,
    )
    instance = np.ones(n)  # all networks present = the real, unablated window
    explanation = explainer.explain_instance(
        instance,
        predict_fn,
        labels=(target_class,),
        num_features=n,
        num_samples=num_samples,
    )
    # local_exp gives (feature_index, weight) pairs — as_list() instead returns
    # "NetworkName=1"-style strings for categorical features, which don't
    # round-trip back to NETWORK_NAMES cleanly.
    weight_by_index = dict(explanation.local_exp[target_class])
    return np.array([weight_by_index.get(i, 0.0) for i in range(n)])


def compare_methods(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    network_indices: dict[str, np.ndarray],
    device: torch.device,
    lime_num_samples: int = 300,
) -> dict[str, dict[str, np.ndarray]]:
    """Runs all four methods on one window; returns {method: {"raw": ..., "normalized": ...}}."""
    x_dev = x.to(device)
    x_dev.requires_grad_(True)

    saliency_raw = aggregate_attribution_to_networks(
        saliency_attribution(model, x_dev, target_class), network_indices
    )
    ig_raw = aggregate_attribution_to_networks(
        integrated_gradients_attribution(model, x_dev, target_class), network_indices
    )
    shapley_raw = exact_shapley_networks(model, x, target_class, network_indices, device)
    lime_raw = lime_networks(
        model, x, target_class, network_indices, device, num_samples=lime_num_samples
    )

    results = {
        "saliency": saliency_raw,
        "integrated_gradients": ig_raw,
        "shapley": shapley_raw,
        "lime": lime_raw,
    }
    return {name: {"raw": vec, "normalized": normalize(vec)} for name, vec in results.items()}
