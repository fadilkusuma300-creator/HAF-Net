#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hafnet.baselines import applicable_baselines, candidate_parameters, fit_baseline, select_parameters
from hafnet.config import load_config
from hafnet.data import prepare_dataset
from hafnet.experiment import METRIC_COLUMNS
from hafnet.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparison methods under the common benchmark protocol.")
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml", help="Comparison-model search configuration")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names or 'all'")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output", default="outputs/baselines")
    parser.add_argument("--parameters", default=None, help="JSON file containing selected parameter dictionaries")
    parser.add_argument("--no-search", action="store_true", help="Use the first configured parameter candidate")
    return parser.parse_args()


def record_from_result(result) -> dict:
    row = {
        "model": result.name,
        "seed": result.seed,
        "threshold": result.threshold,
        "parameters": json.dumps(result.parameters, sort_keys=True),
    }
    for metric in METRIC_COLUMNS:
        row[f"validation_{metric}"] = result.validation[metric]
        row[f"test_{metric}"] = result.test[metric]
    return row


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, block in frame.groupby("model", sort=False):
        row = {"model": model}
        for split in ("validation", "test"):
            for metric in METRIC_COLUMNS:
                values = block[f"{split}_{metric}"].astype(float)
                row[f"{split}_{metric}_mean"] = values.mean()
                row[f"{split}_{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    baseline_config = load_config(args.baseline_config)
    output = ensure_dir(args.output)
    seeds = list(args.seeds if args.seeds is not None else config["training"]["seeds"])
    dataset_name = str(config["dataset"]["name"]).lower()
    allowed = applicable_baselines(dataset_name)
    selected_models = list(allowed) if args.models == ["all"] else args.models
    unsupported = [model for model in selected_models if model not in allowed]
    if unsupported:
        raise ValueError(f"Methods not applicable to {dataset_name}: {unsupported}")

    chosen_parameters: dict[str, dict] = {}
    if args.parameters:
        with Path(args.parameters).open("r", encoding="utf-8") as handle:
            chosen_parameters = json.load(handle)

    tuning_seed = seeds[0]
    tuning_dataset = prepare_dataset(config, tuning_seed)
    for model_name in selected_models:
        if model_name in chosen_parameters:
            continue
        candidates = candidate_parameters(model_name, baseline_config, tuning_dataset)
        if args.no_search:
            chosen_parameters[model_name] = candidates[0]
        else:
            chosen_parameters[model_name] = select_parameters(
                model_name,
                tuning_dataset,
                tuning_seed,
                baseline_config,
            )
    write_json(output / "selected_parameters.json", chosen_parameters)

    records = []
    for seed in seeds:
        dataset = prepare_dataset(config, seed)
        for model_name in selected_models:
            result = fit_baseline(
                model_name,
                dataset,
                seed,
                parameters=chosen_parameters.get(model_name, {}),
                evaluate_test=True,
            )
            records.append(record_from_result(result))
            print(f"{dataset_name} | {model_name} | seed={seed} | test_f1={result.test['f1']:.4f}")

    runs = pd.DataFrame(records)
    summary = summarize(runs)
    runs.to_csv(output / "baseline_runs.csv", index=False)
    summary.to_csv(output / "baseline_summary.csv", index=False)
    print(summary.to_string(index=False))
