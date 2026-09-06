from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from .config import load_config
from .data import prepare_dataset
from .trainer import RunResult, train_one_seed
from .utils import ensure_dir, write_json

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run HAF-Net configurations across the requested seeds."""
    config = load_config(config_path)
    selected_seeds = list(seeds if seeds is not None else config["training"]["seeds"])
    output = ensure_dir(output_dir)
    write_json(output / "config.json", config)
    records: list[dict[str, float | int | str]] = []
    attention_records: list[dict[str, float | int | str]] = []

    for seed in selected_seeds:
        dataset = prepare_dataset(config, seed)
        for variant in variants:
            result = train_one_seed(dataset, config, seed=seed, variant=variant)
            records.append(run_to_record(result))
            if result.group_attention is not None:
                row: dict[str, float | int | str] = {"seed": seed, "variant": variant}
                for group_name, value in zip((group.name for group in dataset.hierarchy.groups), result.group_attention):
                    row[group_name] = float(value)
                attention_records.append(row)
            if save_checkpoints:
                checkpoint_dir = ensure_dir(output / "checkpoints")
                torch.save(result.model_state, checkpoint_dir / f"{dataset.name}_{variant}_seed{seed}.pt")
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
    return runs, summary
