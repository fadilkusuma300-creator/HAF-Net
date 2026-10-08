from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import prepare_dataset
from .provenance import load_selected_config, runtime_manifest, data_contract
from .trainer import RunResult, train_one_seed
from .utils import ensure_dir, validate_disjoint_seeds, validate_seed_list, write_json

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "gmean"]


def run_to_record(result: RunResult) -> dict[str, float | int | str]:
    """Convert one run result to a flat table row."""
    record: dict[str, float | int | str] = {
        "seed": result.seed,
        "variant": result.variant,
        "epochs": result.epochs,
        "parameters": result.parameters,
        "threshold": result.threshold,
    }
    for metric in METRIC_COLUMNS:
        record[f"validation_{metric}"] = result.validation[metric]
        record[f"test_{metric}"] = result.test[metric]
    if result.alignment is not None:
        record["attention_kl"] = result.alignment["kl"]
        record["attention_cosine"] = result.alignment["cosine"]
    return record


def summarize_runs(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize run-level metrics by model configuration."""
    rows: list[dict[str, float | str]] = []
    for variant, block in records.groupby("variant", sort=False):
        row: dict[str, float | str] = {"variant": variant}
        for split in ("validation", "test"):
            for metric in METRIC_COLUMNS:
                values = block[f"{split}_{metric}"].astype(float)
                row[f"{split}_{metric}_mean"] = float(values.mean())
                row[f"{split}_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        thresholds = block["threshold"].astype(float)
        row["threshold_mean"] = float(thresholds.mean())
        row["threshold_std"] = float(thresholds.std(ddof=1)) if len(thresholds) > 1 else 0.0
        for metric in ("attention_kl", "attention_cosine"):
            if metric in block.columns and block[metric].notna().any():
                values = block[metric].dropna().astype(float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_hafnet(
    config_path: str | Path,
    variants: Iterable[str] = ("full",),
    seeds: Iterable[int] | None = None,
    output_dir: str | Path = "outputs",
    save_checkpoints: bool = False,
    parameter_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run HAF-Net configurations across the requested seeds."""
    config = load_config(config_path)
    if parameter_file is None:
        raise ValueError("Final evaluation requires validation-selected parameters.")
    config, selected_parameters = load_selected_config(config, parameter_file)
    selected_seeds = validate_seed_list(
        seeds if seeds is not None else config["training"]["seeds"], "Evaluation seeds"
    )
    tuning_seeds = config["training"].get("tuning_seeds", [])
    if tuning_seeds:
        validate_disjoint_seeds(selected_seeds, tuning_seeds)
    selected_variants = list(variants)
    if not selected_variants or len(selected_variants) != len(set(selected_variants)):
        raise ValueError("At least one unique HAF-Net variant is required.")
    output = ensure_dir(output_dir)
    if (output / "hafnet_runs.csv").exists():
        raise FileExistsError("Choose an empty evaluation output directory.")
    write_json(output / "config.json", config)
    write_json(output / "run_manifest.json", runtime_manifest(config,
        evaluation_seeds=selected_seeds, variants=selected_variants,
        parameter_selection=selected_parameters["_provenance"], output_kind="model_evaluation"))
    if parameter_file is not None:
        write_json(output / "selected_hafnet_parameters.json", selected_parameters)
    records: list[dict[str, float | int | str]] = []
    attention_records: list[dict[str, float | int | str]] = []
    row_attention_records: list[dict[str, float | int | str]] = []
    prediction_records: list[dict[str, float | int | str]] = []
    attention_distribution_records: list[dict[str, float | int | str]] = []

    for seed in selected_seeds:
        dataset = prepare_dataset(config)
        for variant in selected_variants:
            result = train_one_seed(dataset, config, seed=seed, variant=variant, evaluate_test=True)
            records.append(run_to_record(result))

            for split_name, split_data, probabilities in (
                ("validation", dataset.validation, result.validation_probability),
                ("test", dataset.test, result.test_probability),
            ):
                if probabilities is None:
                    continue
                for index, probability in enumerate(probabilities):
                    row: dict[str, float | int | str] = {
                        "dataset": dataset.name,
                        "split": split_name,
                        "model": variant,
                        "seed": seed,
                        "row_id": int(split_data.row_ids[index]),
                        "y_true": int(split_data.y[index]),
                        "score": float(probability),
                        "threshold": float(result.threshold),
                        "prediction": int(probability >= result.threshold),
                    }
                    row.update(split_data.metadata.iloc[index].to_dict())
                    prediction_records.append(row)

            if result.group_attention is not None:
                row: dict[str, float | int | str] = {"seed": seed, "variant": variant}
                for group_name, value in zip((group.name for group in dataset.hierarchy.groups), result.group_attention):
                    row[group_name] = float(value)
                attention_records.append(row)

            if result.test_attention is not None:
                attention = result.test_attention
                groups = dataset.hierarchy.groups
                for index, row_id in enumerate(dataset.test.row_ids):
                    row = {
                        "dataset": dataset.name,
                        "model": variant,
                        "seed": seed,
                        "row_id": int(row_id),
                        "y_true": int(dataset.test.y[index]),
                        "dominant_group": groups[int(np.argmax(attention[index]))].name,
                    }
                    row.update(dataset.test.metadata.iloc[index].to_dict())
                    for group_index, group in enumerate(groups):
                        row[f"group_attention__{group.name}"] = float(attention[index, group_index])
                    row_attention_records.append(row)

                for target in (0, 1):
                    mask = dataset.test.y == target
                    if not np.any(mask):
                        continue
                    block = attention[mask]
                    dominant = np.argmax(block, axis=1)
                    for group_index, group in enumerate(groups):
                        values = block[:, group_index]
                        attention_distribution_records.append(
                            {
                                "dataset": dataset.name,
                                "model": variant,
                                "seed": seed,
                                "y_true": target,
                                "group": group.name,
                                "mean": float(values.mean()),
                                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                                "q25": float(np.quantile(values, 0.25)),
                                "median": float(np.quantile(values, 0.50)),
                                "q75": float(np.quantile(values, 0.75)),
                                "dominant_rate": float(np.mean(dominant == group_index)),
                            }
                        )
            if save_checkpoints:
                checkpoint_dir = ensure_dir(output / "checkpoints")
                torch.save({"state_dict": result.model_state, "seed": seed, "variant": variant,
                    "config": config, "data_contract": data_contract(config),
                    "threshold": result.threshold, "best_epoch": result.epochs},
                    checkpoint_dir / f"{dataset.name}_{variant}_seed{seed}.pt")
            if result.prior is not None:
                write_json(
                    output / f"{dataset.name}_{variant}_seed{seed}_prior.json",
                    {group.name: float(value) for group, value in zip(dataset.hierarchy.groups, result.prior)},
                )

    runs = pd.DataFrame(records)
    summary = summarize_runs(runs)
    runs.to_csv(output / "hafnet_runs.csv", index=False)
    summary.to_csv(output / "hafnet_summary.csv", index=False)
    if attention_records:
        pd.DataFrame(attention_records).to_csv(output / "group_attention.csv", index=False)
    if row_attention_records:
        attention_frame = pd.DataFrame(row_attention_records)
        attention_frame.to_csv(output / "hafnet_attention.csv", index=False)
        stability_rows = []
        for (dataset_name, model, row_id, target), block in attention_frame.groupby(
            ["dataset", "model", "row_id", "y_true"], sort=False
        ):
            dominant = block["dominant_group"].value_counts()
            stability_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model,
                    "row_id": int(row_id),
                    "y_true": int(target),
                    "dominant_group": str(dominant.index[0]),
                    "dominant_rate_across_seeds": float(dominant.iloc[0] / len(block)),
                    "n_seeds": int(len(block)),
                }
            )
        stability_frame = pd.DataFrame(stability_rows)
        stability_frame.to_csv(output / "dominant_group_stability.csv", index=False)
        stability_summary = (
            stability_frame.groupby(["dataset", "model", "y_true"], as_index=False)
            .agg(
                n_rows=("row_id", "nunique"),
                n_seeds=("n_seeds", "max"),
                mean_row_dominant_rate=("dominant_rate_across_seeds", "mean"),
                sd_row_dominant_rate=("dominant_rate_across_seeds", "std"),
            )
        )
        stability_summary.to_csv(output / "dominant_group_stability_summary.csv", index=False)
    if prediction_records:
        pd.DataFrame(prediction_records).to_csv(output / "hafnet_predictions.csv", index=False)
    if attention_distribution_records:
        distribution_frame = pd.DataFrame(attention_distribution_records)
        distribution_frame.to_csv(output / "attention_distribution.csv", index=False)
        distribution_summary = (
            distribution_frame.groupby(["dataset", "model", "y_true", "group"], as_index=False)
            .agg(
                n_seeds=("seed", "nunique"),
                mean_attention=("mean", "mean"),
                sd_mean_attention_across_seeds=("mean", "std"),
                mean_within_seed_sd=("std", "mean"),
                mean_q25=("q25", "mean"),
                mean_median=("median", "mean"),
                mean_q75=("q75", "mean"),
                mean_dominant_rate=("dominant_rate", "mean"),
                sd_dominant_rate_across_seeds=("dominant_rate", "std"),
            )
        )
        distribution_summary.to_csv(output / "attention_distribution_summary.csv", index=False)
    return runs, summary
