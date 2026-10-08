from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_seed_list(seeds: Iterable[int], label: str) -> list[int]:
    supplied = list(seeds)
    if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0 or seed >= 2**32 for seed in supplied):
        raise ValueError(f"{label} must be integers in [0, 2**32).")
    values = [int(seed) for seed in supplied]
    if not values:
        raise ValueError(f"{label} must contain at least one seed.")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicate seeds.")
    return values


def validate_disjoint_seeds(evaluation_seeds: Iterable[int], tuning_seeds: Iterable[int]) -> None:
    overlap = sorted(set(map(int, evaluation_seeds)).intersection(map(int, tuning_seeds)))
    if overlap:
        raise ValueError(f"Evaluation seeds must be distinct from tuning seeds; overlapping values: {overlap}.")


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device string to a PyTorch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def count_trainable_parameters(module: torch.nn.Module) -> int:
    """Count parameters whose gradients are enabled."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding and stable indentation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
