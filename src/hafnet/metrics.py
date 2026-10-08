"""Binary decision and ranking metrics from one shared prediction vector."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from .validation import binary_labels

def _paired(y_true, scores, *, require_both=False):
    y = binary_labels(y_true, require_both=require_both)
    p = np.asarray(scores, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("Scores must be finite and have the same shape as labels.")
    return y, p

def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Maximize validation minority-class F1; break exact ties at the highest threshold."""
    y, p = _paired(y_true, probabilities, require_both=True)
    precision, recall, thresholds = precision_recall_curve(y, p)
    numerator = 2.0 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best]), float(f1[best])

def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """AP uses sklearn average_precision_score, not trapezoidal PR area."""
    y, p = _paired(y_true, probabilities)
    if not np.isfinite(threshold):
        raise ValueError("The operating threshold must be finite.")
    prediction = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    both = np.unique(y).size == 2
    return {"accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "gmean": float(np.sqrt(sensitivity * specificity))}
