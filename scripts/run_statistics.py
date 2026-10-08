#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hafnet.statistics import holm_adjust, paired_prediction_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired test-set inference for HAF-Net and its strongest comparator.")
    parser.add_argument("--taiwan-hafnet", required=True, help="HAF-Net prediction CSV")
    parser.add_argument("--taiwan-baselines", required=True, help="Baseline prediction CSV")
    parser.add_argument("--american-hafnet", required=True, help="HAF-Net prediction CSV")
    parser.add_argument("--american-baselines", required=True, help="Baseline prediction CSV")
    parser.add_argument("--comparator", default="two_stage_lightgbm")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", default="outputs/statistics/primary_comparisons.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    datasets = {
        "taiwan": (args.taiwan_hafnet, args.taiwan_baselines, None),
        "american": (args.american_hafnet, args.american_baselines, "company_name"),
    }
    rows = []
    seed_blocks = []

    for dataset_index, (dataset_name, (hafnet_path, baseline_path, cluster_column)) in enumerate(datasets.items()):
        predictions = pd.concat(
            [pd.read_csv(hafnet_path, float_precision="round_trip"), pd.read_csv(baseline_path, float_precision="round_trip")],
            ignore_index=True,
            sort=False,
        )
        if "dataset" not in predictions.columns:
            raise ValueError(f"Prediction files for {dataset_name} must include a dataset column.")
        observed_datasets = set(predictions["dataset"].dropna().astype(str).str.lower().unique())
        if observed_datasets != {dataset_name}:
            raise ValueError(
                f"Prediction files for {dataset_name} contain unexpected dataset values: {sorted(observed_datasets)}."
            )
        for metric_index, metric in enumerate(("f1", "mcc")):
            inference, seed_differences = paired_prediction_inference(
                predictions=predictions,
                model_a="full",
                model_b=args.comparator,
                metric=metric,
                cluster_column=cluster_column,
                repetitions=args.bootstrap,
                seed=2026 + dataset_index * 10 + metric_index,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "model_a": "full",
                    "model_b": args.comparator,
                    **inference,
                }
            )
            seed_differences.insert(0, "dataset", dataset_name)
            seed_blocks.append(seed_differences)

    adjusted = holm_adjust([row["p_seed_wilcoxon"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_seed_wilcoxon_holm"] = float(value)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.concat(seed_blocks, ignore_index=True).to_csv(
        output.with_name(f"{output.stem}_seed_differences.csv"), index=False
    )
    print(pd.DataFrame(rows).to_string(index=False))
