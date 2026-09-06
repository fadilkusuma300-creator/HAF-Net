from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_f1_threshold(y_true: Iterable[int], probability: Iterable[float]) -> tuple[float, float]:
    """Select one validation threshold that maximizes minority-class F1."""
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probability), dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_value = np.nanmax(f1_values)
    best_indices = np.flatnonzero(np.isclose(f1_values, best_value, rtol=0.0, atol=1e-12))
    # On exact ties, use the larger threshold to avoid unnecessary false positives.
    selected_index = int(best_indices[-1])
    return float(thresholds[selected_index]), float(best_value)


def classification_metrics(
    y_true: Iterable[int],
    probability: Iterable[float],
    threshold: float,
) -> dict[str, float]:
    """Compute threshold-dependent and threshold-free binary classification metrics."""
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probability), dtype=float)
    prediction = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)

    roc_auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    pr_auc = average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(matthews_corrcoef(y, prediction)) if len(np.unique(prediction)) > 1 else 0.0,
        "gmean": float(math.sqrt(max(sensitivity * specificity, 0.0))),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
