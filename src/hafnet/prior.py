from __future__ import annotations

import numpy as np
import torch

from .grouping import HierarchySpec


def statistical_group_prior(
    x_train_standardized: np.ndarray,
    y_train: np.ndarray,
    hierarchy: HierarchySpec,
) -> np.ndarray:
    """Estimate group-level class separation using the training partition only."""
    x = np.asarray(x_train_standardized, dtype=float)
    y = np.asarray(y_train, dtype=int)
    if not (np.any(y == 0) and np.any(y == 1)):
        raise ValueError("Both classes are required to estimate the statistical prior.")

    feature_separation = np.abs(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))
    group_scores: list[float] = []
    for group in hierarchy.groups:
        indices = [index for subgroup in group.subgroups for index in subgroup.indices]
        group_scores.append(float(np.mean(feature_separation[indices])))

    scores = np.asarray(group_scores, dtype=float)
    if scores.sum() <= 0:
        return np.full(len(group_scores), 1.0 / len(group_scores), dtype=np.float32)
    return (scores / scores.sum()).astype(np.float32)


def combined_prior(
    x_train_standardized: np.ndarray,
    y_train: np.ndarray,
    hierarchy: HierarchySpec,
    domain_prior: list[float] | tuple[float, ...],
    rho: float,
) -> torch.Tensor:
    """Combine a heuristic domain prior with the training-derived group prior."""
    if not 0 <= float(rho) <= 1:
        raise ValueError("rho must satisfy 0 <= rho <= 1.")
    domain_array = np.asarray(domain_prior, dtype=float)
    if domain_array.shape != (len(hierarchy.groups),):
        raise ValueError("Domain prior dimension does not match the number of level-1 groups.")
    if not np.all(np.isfinite(domain_array)) or np.any(domain_array <= 0) or domain_array.sum() <= 0:
        raise ValueError("Domain-prior weights must be finite positive values.")
    domain_array = domain_array / domain_array.sum()
    statistical = statistical_group_prior(x_train_standardized, y_train, hierarchy)
    prior = float(rho) * domain_array + (1.0 - float(rho)) * statistical
    prior = prior / prior.sum()
    return torch.tensor(prior, dtype=torch.float32)


def permute_prior(prior: torch.Tensor, permutation: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Change the correspondence between prior components and financial groups."""
    indices = list(int(index) for index in permutation)
    if sorted(indices) != list(range(prior.numel())):
        raise ValueError("Prior permutation must contain every group index exactly once.")
    return prior[indices]
