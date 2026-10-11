from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_wilcoxon(proposed: Iterable[float], comparator: Iterable[float]) -> float:
    proposed_array = np.asarray(list(proposed), dtype=float)
    comparator_array = np.asarray(list(comparator), dtype=float)
    if proposed_array.shape != comparator_array.shape:
        raise ValueError("Paired samples must have identical shape.")
    if proposed_array.ndim != 1 or proposed_array.size < 2 or not np.isfinite(proposed_array).all() or not np.isfinite(comparator_array).all():
        raise ValueError("Wilcoxon requires at least two finite paired scores.")
    differences = proposed_array - comparator_array
    if np.allclose(differences, 0.0):
        return 1.0
    return float(wilcoxon(proposed_array, comparator_array, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and between zero and one.")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        candidate = min((m - rank) * p[index], 1.0)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_prediction_inference(predictions: pd.DataFrame, model_a: str, model_b: str, metric: str,
                                cluster_column: str | None = None, repetitions: int = 10_000,
                                seed: int = 2026, confidence: float = 0.95) -> tuple[dict[str, float | int], pd.DataFrame]:
    if metric not in {"f1", "mcc"}:
        raise ValueError("metric must be 'f1' or 'mcc'.")
    if repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("repetitions must be positive and confidence must be between zero and one.")
    required = {"split", "model", "seed", "row_id", "y_true", "score", "threshold"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    selected = predictions[(predictions["split"] == "test") & predictions["model"].isin([model_a, model_b])].copy()
    if model_a == model_b:
        raise ValueError("Two distinct model identifiers are required.")
    if selected[["seed", "row_id"]].isna().any().any():
        raise ValueError("Seed and row identifiers cannot be missing.")
    if "dataset" in selected and selected["dataset"].nunique(dropna=False) != 1:
        raise ValueError("A paired comparison must use exactly one dataset.")
    if "target_definition" in selected and selected["target_definition"].nunique(dropna=False) != 1:
        raise ValueError("Paired predictions must share one target definition.")
    if selected.groupby(["model", "seed"])["threshold"].nunique(dropna=False).gt(1).any():
        raise ValueError("Each model/seed must use one fixed operating threshold.")
    unit = cluster_column if cluster_column is not None else "row_id"
    if cluster_column is not None and cluster_column not in selected.columns:
        raise ValueError(f"Cluster field is missing from predictions: {cluster_column}")

    columns = ["seed", "row_id", "y_true", "score", "threshold"] + ([] if unit == "row_id" else [unit])
    rename_unit_a = {unit: "unit_a"} if unit != "row_id" else {}
    rename_unit_b = {unit: "unit_b"} if unit != "row_id" else {}
    left = selected[selected["model"] == model_a][columns].rename(columns={"y_true": "y_a", "score": "p_a", "threshold": "t_a", **rename_unit_a})
    right = selected[selected["model"] == model_b][columns].rename(columns={"y_true": "y_b", "score": "p_b", "threshold": "t_b", **rename_unit_b})
    paired = left.merge(right, on=["seed", "row_id"], validate="one_to_one")
    if len(left) != len(right) or len(paired) != len(left):
        raise ValueError("The compared models must have identical test rows for every seed.")
    if paired.empty or not np.array_equal(paired["y_a"].to_numpy(), paired["y_b"].to_numpy()):
        raise ValueError(f"No valid paired test predictions for {model_a} and {model_b}.")
    if unit == "row_id":
        paired["unit_a"] = paired["row_id"]
        paired["unit_b"] = paired["row_id"]
    if not paired["unit_a"].equals(paired["unit_b"]):
        raise ValueError("Paired predictions have inconsistent resampling-unit identifiers.")
    row_sets = paired.groupby("seed")["row_id"].apply(lambda values: tuple(sorted(values.unique())))
    if row_sets.nunique() != 1:
        raise ValueError("All paired seeds must cover the same test observations.")
    if paired.groupby("row_id")["y_a"].nunique(dropna=False).gt(1).any():
        raise ValueError("Test labels must be identical across seeds for each row.")
    if paired["unit_a"].isna().any() or paired.groupby("row_id")["unit_a"].nunique(dropna=False).gt(1).any():
        raise ValueError("Resampling-unit identifiers must be present and stable for every test row.")
    if not paired["y_a"].isin([0, 1]).all() or not np.isfinite(paired[["p_a", "p_b", "t_a", "t_b"]].to_numpy(dtype=float)).all():
        raise ValueError("Test labels, scores, and thresholds are invalid.")

    paired = paired.sort_values(["seed", "row_id"]).reset_index(drop=True)
    seeds = np.sort(paired["seed"].unique())
    units = pd.Index(pd.unique(paired["unit_a"]))
    unit_lookup = {value: index for index, value in enumerate(units)}
    seed_lookup = {int(value): index for index, value in enumerate(seeds)}
    counts = np.zeros((len(units), len(seeds), 2, 4), dtype=np.int64)
    y = paired["y_a"].to_numpy(dtype=int)
    unit_index = paired["unit_a"].map(unit_lookup).to_numpy(dtype=int)
    seed_index = paired["seed"].map(seed_lookup).to_numpy(dtype=int)
    for model_index, suffix in enumerate(("a", "b")):
        predicted = (paired[f"p_{suffix}"].to_numpy(dtype=float) >= paired[f"t_{suffix}"].to_numpy(dtype=float)).astype(int)
        confusion_index = np.where(y == 1, np.where(predicted == 1, 0, 2), np.where(predicted == 1, 1, 3))
        np.add.at(counts, (unit_index, seed_index, model_index, confusion_index), 1)

    def scores_from_counts(values: np.ndarray) -> np.ndarray:
        tp, fp, fn, tn = (values[..., index].astype(float) for index in range(4))
        if metric == "f1":
            return 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1e-12)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(denominator), where=denominator > 0)

    full_counts = counts.sum(axis=0)
    seed_scores = scores_from_counts(full_counts)
    seed_differences = seed_scores[:, 0] - seed_scores[:, 1]
    p_value = paired_wilcoxon(seed_scores[:, 0], seed_scores[:, 1]) if len(seeds) >= 2 else float("nan")

    rng = np.random.default_rng(seed)
    unit_matrix = counts.astype(np.float64).reshape(len(units), -1)
    bootstrap_differences = np.empty(int(repetitions), dtype=float)
    for start in range(0, int(repetitions), 128):
        size = min(128, int(repetitions) - start)
        weights = rng.multinomial(len(units), np.full(len(units), 1.0 / len(units)), size=size).astype(np.float64, copy=False)
        sampled_counts = (weights @ unit_matrix).reshape(size, len(seeds), 2, 4)
        sampled_scores = scores_from_counts(sampled_counts)
        bootstrap_differences[start:start + size] = (sampled_scores[:, :, 0] - sampled_scores[:, :, 1]).mean(axis=1)

    alpha = 1.0 - float(confidence)
    ci_low, ci_high = np.quantile(bootstrap_differences, [alpha / 2.0, 1.0 - alpha / 2.0])
    result = {"n_seeds": int(len(seeds)), "n_test_rows": int(paired["row_id"].nunique()),
        "n_resampling_units": int(len(units)), "bootstrap_unit": unit, "bootstrap_repetitions": int(repetitions),
        "confidence_level": float(confidence), "seed_mean_difference": float(seed_differences.mean()),
        "seed_std_difference": float(seed_differences.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "test_unit_bootstrap_mean_difference": float(seed_differences.mean()),
        "test_unit_bootstrap_ci_low": float(ci_low), "test_unit_bootstrap_ci_high": float(ci_high),
        "p_seed_wilcoxon": float(p_value)}
    seed_frame = pd.DataFrame({"seed": seeds, "model_a": model_a, "model_b": model_b, "metric": metric,
        "score_a": seed_scores[:, 0], "score_b": seed_scores[:, 1], "difference_a_minus_b": seed_differences})
    return result, seed_frame
