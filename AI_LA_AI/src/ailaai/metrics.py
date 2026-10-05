"""Classification metrics with explicit binary labels and zero-division rules."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, log_loss


def macro_f1(y_true: Any, y_pred: Any) -> float:
    """Compute binary Macro-F1 with labels fixed to [0, 1]."""
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays with matching shape.")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Binary labels must be 0 or 1.")
    return float(f1_score(truth, prediction, labels=[0, 1], average="macro", zero_division=0))


def binary_confusion(y_true: Any, y_pred: Any) -> np.ndarray:
    """Return a 2x2 confusion matrix with rows/columns ordered [0, 1]."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def probability_log_loss(y_true: Any, positive_probability: Any) -> float:
    """Compute binary log loss after validating P(class 1)."""
    truth = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(positive_probability, dtype=np.float64)
    if truth.ndim != 1 or not len(truth) or truth.shape != probability.shape or not np.isfinite(probability).all():
        raise ValueError("Labels/probabilities must be finite one-dimensional arrays of equal length.")
    if not np.isin(truth, [0, 1]).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Labels must be binary and probabilities must be in [0, 1].")
    return float(log_loss(truth, np.column_stack((1 - probability, probability)), labels=[0, 1]))


def classification_report(y_true: Any, positive_probability: Any, threshold: float = 0.5) -> dict[str, Any]:
    """Return a compact, JSON-friendly metric summary."""
    truth = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(positive_probability, dtype=np.float64)
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "macro_f1": macro_f1(truth, prediction),
        "log_loss": probability_log_loss(truth, probability),
        "confusion_matrix": binary_confusion(truth, prediction).tolist(),
        "accuracy": float((truth == prediction).mean()),
    }
