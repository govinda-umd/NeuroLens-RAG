"""GRU and Transformer multi-task encoders for MOTOR-task brain decoding.

Both models share the same interface: forward(x) with x of shape
[batch, window_length, n_rois], returning (logits, hrf_pred). `hrf_pred` is
`None` when `include_hrf_head=False`, so classification-only experiments use
an architecture that genuinely has no regression head (rather than one with
an unused head), matching the experimental ladder in
docs/ml-design-report.md §7.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class GRUDecoder(nn.Module):
    """X [B, L, 300] -> GRU -> final hidden [B, hidden_size] -> heads."""

    def __init__(
        self,
        input_size: int = 300,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 6,
        num_conditions: int = 5,
        include_hrf_head: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.hrf_head = nn.Linear(hidden_size, num_conditions) if include_hrf_head else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """The pooled representation before either head — the actual
        multi-task-learned representation (see docs/case2-3-design-plan.md
        §1), used directly by concept-activation-vector probing."""
        _, h_n = self.gru(x)
        return h_n[-1]  # final layer's hidden state, [B, hidden_size]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.forward_features(x)
        logits = self.classifier(h)
        hrf_pred = self.hrf_head(h) if self.hrf_head is not None else None
        return logits, hrf_pred


class TransformerDecoder(nn.Module):
    """X [B, L, 300] -> linear projection -> Transformer encoder -> final token -> heads."""

    def __init__(
        self,
        input_size: int = 300,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 6,
        num_conditions: int = 5,
        include_hrf_head: bool = True,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.hrf_head = nn.Linear(d_model, num_conditions) if include_hrf_head else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """The pooled representation before either head — the actual
        multi-task-learned representation (see docs/case2-3-design-plan.md
        §1), used directly by concept-activation-vector probing."""
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)  # [B, L, d_model]
        return h[:, -1, :]  # final temporal token (v1 pooling choice; see open questions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        pooled = self.forward_features(x)
        logits = self.classifier(pooled)
        hrf_pred = self.hrf_head(pooled) if self.hrf_head is not None else None
        return logits, hrf_pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
