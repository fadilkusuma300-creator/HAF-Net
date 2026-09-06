#!/usr/bin/env python3
"""Evaluate HAF-Net validation performance over selected hyperparameter grids."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hafnet.config import load_config
from hafnet.data import prepare_dataset
from hafnet.experiment import run_to_record
from hafnet.trainer import train_one_seed
from hafnet.utils import ensure_dir

DEFAULT_GRIDS = {
    "lambda_tcar": [0.00, 0.02, 0.05, 0.08, 0.10],
    "beta": [0.990, 0.995, 0.999, 0.9995],
    "gamma": [0.0, 1.0, 2.0, 3.0, 4.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Taiwan HAF-Net hyperparameter sensitivity analysis.")
    parser.add_argument("--config", default="configs/taiwan.yaml")
    parser.add_argument("--parameter", choices=["lambda_tcar", "beta", "gamma", "all"], default="all")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output", default="outputs/sensitivity")
    return parser.parse_args()


def set_parameter(config: dict, parameter: str, value: float) -> dict:
    current = deepcopy(config)
    current["loss"][parameter] = float(value)
    return current


def run_parameter(config: dict, parameter: str, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for value in DEFAULT_GRIDS[parameter]:
        current = set_parameter(config, parameter, value)
        for seed in seeds:
            dataset = prepare_dataset(current, seed)
            result = train_one_seed(dataset, current, seed=seed, variant="full")
            row = run_to_record(result)
            row["parameter"] = parameter
            row["value"] = float(value)
            rows.append(row)
            print(
                f"{parameter}={value:g} | seed={seed} | "
                f"validation_f1={result.validation['f1']:.4f} | validation_pr_auc={result.validation['pr_auc']:.4f}"
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["parameter", "value"], as_index=False)
        .agg(
            validation_f1_mean=("validation_f1", "mean"),
            validation_f1_std=("validation_f1", "std"),
            validation_pr_auc_mean=("validation_pr_auc", "mean"),
            validation_pr_auc_std=("validation_pr_auc", "std"),
        )
        .fillna(0.0)
    )


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    parameters = [parameter for parameter in ("lambda_tcar", "beta", "gamma") if parameter in set(summary["parameter"])]
    if not parameters:
        return
    figure, axes = plt.subplots(1, len(parameters), figsize=(4.4 * len(parameters), 3.6), squeeze=False)
    for axis, parameter in zip(axes[0], parameters):
        block = summary[summary["parameter"] == parameter].sort_values("value")
        axis.errorbar(
            block["value"],
            block["validation_f1_mean"],
            yerr=block["validation_f1_std"],
            marker="o",
            capsize=3,
        )
        axis.set_xlabel(parameter)
        axis.set_ylabel("Validation F1")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "sensitivity.pdf", bbox_inches="tight")
    figure.savefig(output / "sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if str(config["dataset"]["name"]).lower() != "taiwan":
        raise ValueError("The manuscript sensitivity analysis is defined for the Taiwan benchmark.")

    seeds = list(args.seeds if args.seeds is not None else config["training"]["seeds"])
    parameters = list(DEFAULT_GRIDS) if args.parameter == "all" else [args.parameter]
    output = ensure_dir(args.output)
    blocks = [run_parameter(config, parameter, seeds) for parameter in parameters]
    runs = pd.concat(blocks, ignore_index=True)
    summary = summarize(runs)
    runs.to_csv(output / "sensitivity_runs.csv", index=False)
    summary.to_csv(output / "sensitivity_summary.csv", index=False)
    make_figure(summary, output)
    print(summary.to_string(index=False))
