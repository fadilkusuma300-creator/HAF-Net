"""Input contracts shared by data, metrics, and statistical procedures."""
from __future__ import annotations
import numpy as np

def binary_labels(values, *, require_both: bool = False) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Labels must be a nonempty one-dimensional array.")
    try:
        numeric = y.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must be numerical binary values.") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError("Labels must contain only finite values 0 and 1.")
    result = numeric.astype(np.int64)
    if require_both and np.unique(result).size != 2:
        raise ValueError("Both target classes are required for this procedure.")
    return result

def finite_matrix(values, *, expected_features: int | None = None) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("Features must be a nonempty two-dimensional matrix.")
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite; implicit imputation is not permitted.")
    if expected_features is not None and x.shape[1] != expected_features:
        raise ValueError("The feature count differs from the fitted preprocessing schema.")
    return x

def integral_values(values, *, name: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError(f"{name} must contain finite integer values.")
    return a.astype(np.int64)
