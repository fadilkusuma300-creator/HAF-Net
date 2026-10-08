"""Training-only winsorization followed by training-only standardization."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .validation import finite_matrix

@dataclass
class TrainOnlyPreprocessor:
    lower_: np.ndarray | None = None
    upper_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "TrainOnlyPreprocessor":
        x = finite_matrix(values)
        self.lower_ = np.quantile(x, 0.01, axis=0)
        self.upper_ = np.quantile(x, 0.99, axis=0)
        clipped = np.clip(x, self.lower_, self.upper_)
        self.mean_ = clipped.mean(axis=0)
        self.scale_ = clipped.std(axis=0, ddof=0)
        self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if any(v is None for v in (self.lower_, self.upper_, self.mean_, self.scale_)):
            raise ValueError("Fit the preprocessor before transforming features.")
        x = finite_matrix(values, expected_features=len(self.mean_))
        return ((np.clip(x, self.lower_, self.upper_) - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def to_dict(self) -> dict:
        if self.mean_ is None:
            raise ValueError("The preprocessor is not fitted.")
        return {key: getattr(self, key).tolist() for key in ("lower_", "upper_", "mean_", "scale_")}
