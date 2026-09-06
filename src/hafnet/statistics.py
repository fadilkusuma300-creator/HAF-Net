from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class PairedInference:
    dataset: str
    metric: str
    n_pairs: int
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float | None = None


def paired_bootstrap_ci(
    proposed: Iterable[float],
    comparator: Iterable[float],
    repetitions: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float, float]:
    """Bootstrap the paired mean difference and return estimate plus percentile interval."""
    proposed_array = np.asarray(list(proposed), dtype=float)
    comparator_array = np.asarray(list(comparator), dtype=float)
    if proposed_array.shape != comparator_array.shape or proposed_array.ndim != 1:
        raise ValueError("Paired samples must be one-dimensional arrays with identical shape.")
    if len(proposed_array) < 2:
        raise ValueError("At least two paired observations are required.")
    differences = proposed_array - comparator_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(int(repetitions), len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    alpha = 1.0 - float(confidence)
    low, high = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(differences.mean()), float(low), float(high)


def paired_wilcoxon(proposed: Iterable[float], comparator: Iterable[float]) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value."""
    proposed_array = np.asarray(list(proposed), dtype=float)
    comparator_array = np.asarray(list(comparator), dtype=float)
    if proposed_array.shape != comparator_array.shape:
        raise ValueError("Paired samples must have identical shape.")
    differences = proposed_array - comparator_array
    if np.allclose(differences, 0.0):
        return 1.0
    result = wilcoxon(proposed_array, comparator_array, alternative="two-sided", zero_method="wilcox", method="auto")
    return float(result.pvalue)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down adjustment controlling family-wise error."""
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        candidate = min((m - rank) * p[index], 1.0)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted
