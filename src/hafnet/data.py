from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .grouping import HierarchySpec, build_hierarchy
from .preprocess import TrainOnlyPreprocessor


@dataclass
class SplitData:
    """Prepared feature matrix, labels, row IDs, and non-predictive metadata."""

    x: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray
    metadata: pd.DataFrame


@dataclass
class PreparedDataset:
    """Dataset container used by HAF-Net and all comparison methods."""

    name: str
    feature_ids: tuple[str, ...]
    feature_labels: tuple[str, ...]
    hierarchy: HierarchySpec
    train: SplitData
    validation: SplitData
    test: SplitData
    preprocessor: TrainOnlyPreprocessor
    raw_train_frame: pd.DataFrame
    raw_validation_frame: pd.DataFrame
    raw_test_frame: pd.DataFrame


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    return cleaned


def load_taiwan(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """Load the UCI Taiwan bankruptcy table and assign stable T1-T95 feature IDs."""
    frame = _clean_columns(pd.read_csv(path))
    target_candidates = [
        column
        for column in frame.columns
        if column.lower().replace(" ", "") in {"bankrupt?", "bankrupt", "bankruptcy"}
    ]
    if not target_candidates:
        raise ValueError("Taiwan data must contain the 'Bankrupt?' target column.")
    target_column = target_candidates[0]
    features = frame.drop(columns=[target_column]).copy()
    if features.shape[1] != 95:
        raise ValueError(f"Expected 95 Taiwan predictors, found {features.shape[1]}.")

    feature_labels = tuple(str(column) for column in features.columns)
    features.columns = [f"T{index}" for index in range(1, 96)]
    features = features.apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(frame[target_column], errors="raise").astype(int).to_numpy()
    if set(np.unique(y)).difference({0, 1}):
        raise ValueError("Taiwan target must be binary.")

    features["row_id"] = np.arange(len(features), dtype=int)
    return features, y, feature_labels


def _binary_status(series: pd.Series) -> np.ndarray:
    """Map the public American status labels to 0/1 without altering row timing."""
    text = series.astype(str).str.strip().str.lower()
    positive = {"failed", "failure", "bankrupt", "bankruptcy", "1", "true"}
    negative = {"alive", "healthy", "non-bankrupt", "nonbankrupt", "0", "false"}
    unknown = set(text.unique()).difference(positive | negative)
    if unknown:
        raise ValueError(f"Unrecognized American status labels: {sorted(unknown)[:8]}")
    return text.isin(positive).astype(int).to_numpy()


def load_american(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """Load the public American firm-year table using X1-X18 as predictive inputs.

    The benchmark already labels the fiscal year immediately before a bankruptcy filing
    as the positive class. The loader therefore uses the public row-level label directly.
    """
    frame = _clean_columns(pd.read_csv(path))
    feature_ids = [f"X{index}" for index in range(1, 19)]
    year_column = "year" if "year" in frame.columns else "fyear" if "fyear" in frame.columns else None
    required = ["company_name", "status_label", *feature_ids]
    missing = [column for column in required if column not in frame.columns]
    if year_column is None:
        missing.append("year/fyear")
    if missing:
        raise ValueError(f"American data is missing columns: {missing}")

    data = frame.copy()
    data[year_column] = pd.to_numeric(data[year_column], errors="raise").astype(int)
    for feature in feature_ids:
        data[feature] = pd.to_numeric(data[feature], errors="raise")
    y = _binary_status(data["status_label"])

    # A positive row is the fiscal year immediately before filing, so the filing year is +1.
    event_year = np.where(y == 1, data[year_column].to_numpy(dtype=int) + 1, np.nan)
    output = data[feature_ids].copy()
    output["row_id"] = np.arange(len(output), dtype=int)
    output["company_name"] = data["company_name"].astype(str).to_numpy()
    output["year"] = data[year_column].to_numpy(dtype=int)
    output["event_year"] = event_year
    return output, y, tuple(feature_ids)


def _split_taiwan(frame: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    train_idx, held_out_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    validation_idx, test_idx = train_test_split(
        held_out_idx,
        test_size=0.50,
        random_state=seed,
        stratify=y[held_out_idx],
    )
    return np.asarray(train_idx), np.asarray(validation_idx), np.asarray(test_idx)


def _split_american(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_low, train_high = config.get("train_years", [1999, 2011])
    validation_low, validation_high = config.get("validation_years", [2012, 2014])
    test_low, test_high = config.get("test_years", [2015, 2018])
    years = frame["year"].to_numpy(dtype=int)
    train_idx = np.flatnonzero((years >= train_low) & (years <= train_high))
    validation_idx = np.flatnonzero((years >= validation_low) & (years <= validation_high))
    test_idx = np.flatnonzero((years >= test_low) & (years <= test_high))
    if min(len(train_idx), len(validation_idx), len(test_idx)) == 0:
        raise ValueError("One or more chronological American partitions are empty.")
    return train_idx, validation_idx, test_idx


def prepare_dataset(config: dict[str, Any], seed: int) -> PreparedDataset:
    """Load, split, preprocess, and attach the fixed feature hierarchy."""
    dataset_config = config["dataset"]
    dataset_name = str(dataset_config["name"]).lower()
    dataset_path = Path(dataset_config["path"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. See data/README.md or run scripts/download_data.py."
        )

    if dataset_name == "taiwan":
        frame, y, feature_labels = load_taiwan(dataset_path)
        feature_ids = tuple(f"T{index}" for index in range(1, 96))
        train_idx, validation_idx, test_idx = _split_taiwan(frame, y, seed)
        metadata_columns = ["row_id"]
    elif dataset_name == "american":
        frame, y, feature_labels = load_american(dataset_path)
        feature_ids = tuple(f"X{index}" for index in range(1, 19))
        train_idx, validation_idx, test_idx = _split_american(frame, dataset_config)
        metadata_columns = ["row_id", "company_name", "year", "event_year"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    hierarchy = build_hierarchy(dataset_config["mapping"], dataset_name, feature_ids)
    x_all = frame[list(feature_ids)].to_numpy(dtype=np.float64)
    preprocessor = TrainOnlyPreprocessor()
    x_train = preprocessor.fit_transform(x_all[train_idx])
    x_validation = preprocessor.transform(x_all[validation_idx])
    x_test = preprocessor.transform(x_all[test_idx])

    def pack(indices: np.ndarray, x: np.ndarray) -> SplitData:
        metadata = frame.iloc[indices][metadata_columns].reset_index(drop=True)
        return SplitData(
            x=x,
            y=y[indices].astype(np.int64),
            row_ids=frame.iloc[indices]["row_id"].to_numpy(dtype=np.int64),
            metadata=metadata,
        )

    raw_train = frame.iloc[train_idx].copy().reset_index(drop=True)
    raw_validation = frame.iloc[validation_idx].copy().reset_index(drop=True)
    raw_test = frame.iloc[test_idx].copy().reset_index(drop=True)
    raw_train["target"] = y[train_idx]
    raw_validation["target"] = y[validation_idx]
    raw_test["target"] = y[test_idx]

    return PreparedDataset(
        name=dataset_name,
        feature_ids=feature_ids,
        feature_labels=feature_labels,
        hierarchy=hierarchy,
        train=pack(train_idx, x_train),
        validation=pack(validation_idx, x_validation),
        test=pack(test_idx, x_test),
        preprocessor=preprocessor,
        raw_train_frame=raw_train,
        raw_validation_frame=raw_validation,
        raw_test_frame=raw_test,
    )
