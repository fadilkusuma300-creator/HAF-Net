from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def resolve_checkpoint_seed(checkpoint: Path, requested_seed: int | None) -> int:
    match = re.search(r"_seed(\d+)(?:\D|$)", checkpoint.name)
    checkpoint_seed = int(match.group(1)) if match else None
    if requested_seed is not None and checkpoint_seed is not None and int(requested_seed) != checkpoint_seed:
        raise ValueError("--seed does not match the checkpoint filename.")
    seed = int(requested_seed) if requested_seed is not None else checkpoint_seed
    if seed is None:
        raise ValueError("Provide --seed when it cannot be read from the checkpoint filename.")
    return seed


def select_cases(
    attention: np.ndarray,
    y: np.ndarray,
    desired_groups: tuple[int, ...] = (0, 1, 2),
) -> list[tuple[int, int, bool]]:
    from .validation import binary_labels
    y = binary_labels(y)
    attention = np.asarray(attention, dtype=float)
    if attention.ndim != 2 or attention.shape[0] != len(y) or not np.isfinite(attention).all():
        raise ValueError("Attention must be a finite matrix with one row per observation.")
    if np.any(attention < 0) or not np.allclose(attention.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Group-attention rows must be probability distributions.")
    if len(set(desired_groups)) != len(desired_groups) or any(g < 0 or g >= attention.shape[1] for g in desired_groups):
        raise ValueError("Requested group indices must be unique and in range.")
    positive_indices = np.flatnonzero(y == 1)
    if len(positive_indices) < len(desired_groups):
        raise ValueError("The test partition has fewer positive observations than requested cases.")
    selected: list[tuple[int, int, bool]] = []
    dominant = np.argmax(attention, axis=1)
    for group_index in desired_groups:
        selected_indices = {row_index for row_index, _, _ in selected}
        candidates = np.asarray(
            [index for index in positive_indices if dominant[index] == group_index and index not in selected_indices],
            dtype=int,
        )
        if len(candidates):
            case = int(candidates[np.argmax(attention[candidates, group_index])])
            fallback = False
        else:
            remaining = np.asarray([index for index in positive_indices if int(index) not in selected_indices], dtype=int)
            case = int(remaining[np.argmax(attention[remaining, group_index])])
            fallback = True
        selected.append((case, group_index, fallback))
    return selected
