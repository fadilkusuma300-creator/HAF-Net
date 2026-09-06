from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainOnlyPreprocessor:
    """Winsorize and standardize numeric features using training statistics only."""

    lower_: np.ndarray | None = None
    upper_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    @staticmethod
    def _check_complete(x: np.ndarray, label: str) -> None:
        if not np.isfinite(x).all():
            count = int(np.size(x) - np.isfinite(x).sum())
            raise ValueError(
                f"{label} contains {count} missing or non-finite values. "
                "The benchmark protocol assumes complete numeric inputs."
            )

    def fit(self, x: np.ndarray) -> "TrainOnlyPreprocessor":
        x = np.asarray(x, dtype=np.float64)
        self._check_complete(x, "Training data")
        self.lower_ = np.quantile(x, 0.01, axis=0)
        self.upper_ = np.quantile(x, 0.99, axis=0)
        clipped = np.clip(x, self.lower_, self.upper_)
        self.mean_ = clipped.mean(axis=0)
        self.scale_ = clipped.std(axis=0, ddof=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in (self.lower_, self.upper_, self.mean_, self.scale_)):
            raise RuntimeError("Preprocessing statistics have not been fitted.")
        x = np.asarray(x, dtype=np.float64)
        self._check_complete(x, "Input data")
        clipped = np.clip(x, self.lower_, self.upper_)
        standardized = (clipped - self.mean_) / self.scale_
        return standardized.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
