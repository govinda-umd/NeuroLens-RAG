"""Baseline MLP for Case 1: tests whether a non-recurrent, non-attention
model on the raw window already separates the 6 motor classes, before
crediting any of Case 1/2/3's accuracy to their learned sequence
representations specifically. If a trivial aggregation of the same raw
input gets close to GRU/Transformer's numbers, the classes are already
close to linearly-ish separable in the raw feature space and the
sequence modeling isn't doing much beyond what a generic function
approximator over the raw window already captures. If it does much
worse, that's real evidence the temporal structure GRU/Transformer learn
is doing real work, not just adding capacity.

Two variants, differing only in how they throw away temporal structure
(neither has any learned notion of time order):
- FlattenMLP: whole window flattened to one 9600-dim vector -- keeps
  every raw timepoint x ROI value, but "time" is just "which flat index".
- MeanPoolMLP: window averaged over time to one 300-dim vector -- keeps
  only the spatial (per-ROI) activation pattern, discards temporal
  dynamics entirely.

Same forward_features/forward interface as GRUDecoder/TransformerDecoder
(model_builder.py), so `engine.py::train_experiment` and `concepts.py`'s
CAV/TCAV code run on either variant completely unmodified.
"""

from __future__ import annotations

import torch
from torch import nn


class FlattenMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 300,
        window_length: int = 32,
        hidden_size: int = 128,
        num_classes: int = 6,
        num_conditions: int = 5,
        include_hrf_head: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = input_size * window_length
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.hrf_head = nn.Linear(hidden_size, num_conditions) if include_hrf_head else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return self.net(flat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.forward_features(x)
        logits = self.classifier(h)
        hrf_pred = self.hrf_head(h) if self.hrf_head is not None else None
        return logits, hrf_pred


class MeanPoolMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 300,
        hidden_size: int = 128,
        num_classes: int = 6,
        num_conditions: int = 5,
        include_hrf_head: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.hrf_head = nn.Linear(hidden_size, num_conditions) if include_hrf_head else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)  # [B, L, 300] -> [B, 300], last-timepoint info discarded on purpose
        return self.net(pooled)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.forward_features(x)
        logits = self.classifier(h)
        hrf_pred = self.hrf_head(h) if self.hrf_head is not None else None
        return logits, hrf_pred
