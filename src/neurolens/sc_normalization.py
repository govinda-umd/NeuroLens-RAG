"""Analysis-time normalization schemes for the raw SC sufficient statistics
produced by scripts/dti_sc_pipeline.py (docs/structural/dti-sc-pipeline-plan.md
Sec 5). Each function is a cheap post-hoc transform of the stored arrays --
none require re-tractography.
"""

from __future__ import annotations

import numpy as np


def log_streamline_count(sc_streamline_count: np.ndarray) -> np.ndarray:
    """S_ij = log10(1 + N_ij). Reduces the right-skew of raw streamline
    counts; chosen normalization for this project's SC connectomes."""
    return np.log10(1.0 + sc_streamline_count)


def volume_normalized(sc_streamline_count: np.ndarray, roi_volumes: np.ndarray) -> np.ndarray:
    """N_ij / (V_i + V_j), correcting for the tractography bias toward
    larger ROIs accumulating more streamline endpoints."""
    denom = roi_volumes[:, None] + roi_volumes[None, :]
    return np.divide(sc_streamline_count, denom, out=np.zeros_like(sc_streamline_count, dtype=np.float64), where=denom > 0)


def length_normalized(sc_streamline_count: np.ndarray, sc_mean_length: np.ndarray) -> np.ndarray:
    """N_ij / L_ij, correcting for the tractography bias toward shorter
    connections accumulating more streamlines."""
    return np.divide(
        sc_streamline_count, sc_mean_length,
        out=np.zeros_like(sc_streamline_count, dtype=np.float64),
        where=sc_mean_length > 0,
    )
