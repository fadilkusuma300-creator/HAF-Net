from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalPair:
    """Indices for one distressed firm's anchor and positive pre-filing observations."""

    positive_local_index: int
    anchor_local_index: int
    anchor_year: int
    company_name: str


class TemporalPairBank:
    """Index American training-time temporal pairs and same-year healthy negatives.

    A positive row represents the fiscal year immediately before filing. With
    tau_anchor=2 and tau_positive=1, the anchor is therefore one fiscal year earlier
    than the positive row. All indices are built exclusively from the training partition.
    """

    def __init__(self, train_metadata: pd.DataFrame, y_train: np.ndarray, anchor_gap: int = 1) -> None:
        if anchor_gap < 1:
            raise ValueError("anchor_gap must be a positive integer.")
        required = {"company_name", "year"}
        if not required.issubset(train_metadata.columns):
            self.pairs: dict[int, TemporalPair] = {}
            self.negative_pool: dict[int, np.ndarray] = {}
            self.company_by_index = np.asarray([], dtype=object)
            return

        metadata = train_metadata.reset_index(drop=True).copy()
        targets = np.asarray(y_train, dtype=int)
        metadata["target"] = targets
        metadata["company_name"] = metadata["company_name"].astype(str)
        self.company_by_index = metadata["company_name"].to_numpy(dtype=object)
        lookup = {
            (str(row.company_name), int(row.year)): index
            for index, row in metadata.iterrows()
        }

        self.pairs = {}
        for positive_index in np.flatnonzero(targets == 1):
            row = metadata.iloc[positive_index]
            positive_year = int(row["year"])
            anchor_year = positive_year - int(anchor_gap)
            company = str(row["company_name"])
            anchor_index = lookup.get((company, anchor_year))
            if anchor_index is None:
                continue
            self.pairs[int(positive_index)] = TemporalPair(
                positive_local_index=int(positive_index),
                anchor_local_index=int(anchor_index),
                anchor_year=anchor_year,
                company_name=company,
            )

        self.negative_pool = {}
        for year, block in metadata.groupby("year"):
            healthy_indices = block.index[block["target"].to_numpy(dtype=int) == 0].to_numpy(dtype=int)
            if len(healthy_indices):
                self.negative_pool[int(year)] = healthy_indices

    def sample_for_batch(
        self,
        batch_local_indices: np.ndarray,
        negatives: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Sample anchor, positive, and same-year healthy negative indices for a batch."""
        positive_indices: list[int] = []
        anchor_indices: list[int] = []
        negative_indices: list[np.ndarray] = []

        for local_index in batch_local_indices.tolist():
            pair = self.pairs.get(int(local_index))
            if pair is None:
                continue
            pool = self.negative_pool.get(pair.anchor_year)
            if pool is None or len(pool) == 0:
                continue
            pool = pool[self.company_by_index[pool] != pair.company_name]
            if len(pool) == 0:
                continue
            selected = rng.choice(pool, size=negatives, replace=len(pool) < negatives)
            positive_indices.append(pair.positive_local_index)
            anchor_indices.append(pair.anchor_local_index)
            negative_indices.append(np.asarray(selected, dtype=int))

        if not positive_indices:
            return None
        return (
            np.asarray(anchor_indices, dtype=int),
            np.asarray(positive_indices, dtype=int),
            np.stack(negative_indices, axis=0),
        )
