#!/usr/bin/env python3
"""Evaluate HAF-Net validation performance over the prespecified sensitivity grids."""
from __future__ import annotations

import argparse
from copy import deepcopy

import pandas as pd

from hafnet.config import load_config
from hafnet.data import prepare_dataset

DEFAULT_GRIDS = {
    "lambda_tcar": [0.00, 0.02, 0.05, 0.10, 0.20],
    "beta": [0.90, 0.99, 0.999, 0.9999],
    "gamma": [0.0, 1.0, 2.0, 3.0, 5.0],
    "rho": [0.0, 0.25, 0.50, 0.75, 1.0],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Taiwan HAF-Net validation sensitivity analysis.")
    parser.add_argument("--config", default="configs/taiwan.yaml")
    parser.add_argument("--parameter", choices=[*DEFAULT_GRIDS, "all"], default="all")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--output", default="outputs/sensitivity")
    return parser.parse_args(argv)


def set_parameter(config: dict, parameter: str, value: float) -> dict:
    current = deepcopy(config)
    current["loss"][parameter] = float(value)
    return current


def run_parameter(config: dict, parameter: str, seeds: list[int]) -> pd.DataFrame:
    from hafnet.experiment import run_to_record
    from hafnet.trainer import train_one_seed

    rows: list[dict] = []
    for value in DEFAULT_GRIDS[parameter]:
        current = set_parameter(config, parameter, value)
        for seed in seeds:
            dataset = prepare_dataset(current)
            result = train_one_seed(dataset, current, seed=seed, variant="full", evaluate_test=False)
            row = run_to_record(result)
            row = {key: item for key, item in row.items() if not key.startswith("test_")}
            row["parameter"] = parameter
            row["value"] = float(value)
            rows.append(row)
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from hafnet.provenance import load_selected_config
    from hafnet.utils import ensure_dir, validate_disjoint_seeds, validate_seed_list, write_json

    config = load_config(args.config)
    config, _ = load_selected_config(config, args.parameters)
    if str(config["dataset"]["name"]).lower() != "taiwan":
        raise ValueError("Sensitivity analysis is defined for the Taiwan benchmark.")

    seeds = validate_seed_list(
        args.seeds if args.seeds is not None else config["training"].get("tuning_seeds", config["training"]["seeds"]),
        "Sensitivity seeds",
    )
    validate_disjoint_seeds(seeds, config["training"].get("seeds", []))
    parameters = list(DEFAULT_GRIDS) if args.parameter == "all" else [args.parameter]
    output = ensure_dir(args.output)

    blocks = [run_parameter(config, parameter, seeds) for parameter in parameters]
    runs = pd.concat(blocks, ignore_index=True)
    summary = summarize(runs)
    runs.to_csv(output / "sensitivity_runs.csv", index=False)
    summary.to_csv(output / "sensitivity_summary.csv", index=False)

    selected_values = {parameter: float(config["loss"][parameter]) for parameter in parameters}
    write_json(
        output / "sensitivity_manifest.json",
        {
            "dataset": "taiwan",
            "seeds": seeds,
            "grids": {parameter: DEFAULT_GRIDS[parameter] for parameter in parameters},
            "display_metric": "validation_average_precision",
            "parameter_selection": "none; configured reference values are highlighted",
            "selected_architecture": {
                key: config["model"][key] for key in ("subgroup_dim", "fusion_dim", "dropout")
            },
            "test_evaluated": False,
            "reference_values": selected_values,
        },
    )

    from plot_sensitivity import make_figure

    make_figure(summary, output, selected_values)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
