"""Classification and HRF-regression metrics for the brain-decoding experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    labels = list(range(num_classes))
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1_per_class.tolist(),
        "support": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def hrf_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred: [N, n_conditions]."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    correlations = []
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c], y_pred[:, c]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            correlations.append(float("nan"))
        else:
            correlations.append(float(np.corrcoef(yt, yp)[0, 1]))

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-8, 1.0 - ss_res / ss_tot, np.nan).tolist()

    return {
        "mse": mse,
        "mae": mae,
        "per_condition_correlation": correlations,
        "per_condition_r2": r2,
    }


def empty_hrf_metrics() -> dict:
    """Placeholder for classification-only models with no HRF head."""
    return {"mse": None, "mae": None, "per_condition_correlation": None, "per_condition_r2": None}
