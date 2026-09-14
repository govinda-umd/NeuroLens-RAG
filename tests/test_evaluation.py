"""classification_metrics against hand-computed values on a known example."""

import numpy as np

from neurolens.evaluation import classification_metrics


def test_perfect_predictions_score_1():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    metrics = classification_metrics(y_true, y_pred, num_classes=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_known_confusion_matrix_and_macro_f1():
    # Class 0: 2/2 correct. Class 1: 1/2 correct (1 predicted as 2).
    # Class 2: 2/2 correct. Hand-computed macro-F1 below.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])
    metrics = classification_metrics(y_true, y_pred, num_classes=3)

    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    # precision/recall per class: class0 P=1,R=1,F1=1; class1 P=1,R=0.5,F1=2/3;
    # class2 P=2/3,R=1,F1=0.8 -> macro F1 = (1 + 2/3 + 0.8)/3
    expected_macro_f1 = (1.0 + 2 / 3 + 0.8) / 3
    assert abs(metrics["macro_f1"] - expected_macro_f1) < 1e-9


def test_absent_class_does_not_crash_and_scores_zero():
    # Class 2 never appears in either array -- zero_division=0 should give
    # it 0 precision/recall/F1 rather than raising or returning NaN.
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = classification_metrics(y_true, y_pred, num_classes=3)
    assert metrics["per_class_f1"][2] == 0.0
    assert not np.isnan(metrics["macro_f1"])
