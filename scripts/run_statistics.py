#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hafnet.statistics import holm_adjust, paired_bootstrap_ci, paired_wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired HAF-Net vs Two-Stage LightGBM inference for F1 and MCC across both benchmarks."
    )
    parser.add_argument("--taiwan-hafnet", required=True)
    parser.add_argument("--taiwan-baselines", required=True)
    parser.add_argument("--american-hafnet", required=True)
    parser.add_argument("--american-baselines", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", default="outputs/statistics/primary_comparisons.csv")
    return parser.parse_args()


def paired_values(hafnet_path: str, baseline_path: str, metric: str) -> tuple[pd.Series, pd.Series]:
    hafnet = pd.read_csv(hafnet_path)
    baselines = pd.read_csv(baseline_path)
    hafnet = hafnet[hafnet["variant"] == "full"][["seed", f"test_{metric}"]]
    baselines = baselines[baselines["model"] == "two_stage_lightgbm"][["seed", f"test_{metric}"]]
    merged = hafnet.merge(baselines, on="seed", suffixes=("_hafnet", "_baseline"), validate="one_to_one")
    if len(merged) < 2:
        raise ValueError("At least two paired seeds are required.")
    return merged[f"test_{metric}_hafnet"], merged[f"test_{metric}_baseline"]


if __name__ == "__main__":
    args = parse_args()
    comparisons = [
        ("taiwan", args.taiwan_hafnet, args.taiwan_baselines, "f1"),
        ("taiwan", args.taiwan_hafnet, args.taiwan_baselines, "mcc"),
        ("american", args.american_hafnet, args.american_baselines, "f1"),
        ("american", args.american_hafnet, args.american_baselines, "mcc"),
    ]
    rows = []
    raw_p = []
    for index, (dataset, hafnet_path, baseline_path, metric) in enumerate(comparisons):
        proposed, comparator = paired_values(hafnet_path, baseline_path, metric)
        mean_difference, ci_low, ci_high = paired_bootstrap_ci(
            proposed,
            comparator,
            repetitions=args.bootstrap,
            seed=2026 + index,
        )
        p_value = paired_wilcoxon(proposed, comparator)
        raw_p.append(p_value)
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "pairs": len(proposed),
                "mean_difference": mean_difference,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
            }
        )

    adjusted = holm_adjust(raw_p)
    for row, value in zip(rows, adjusted):
        row["p_adjusted_holm"] = float(value)
        row["significant_0_05"] = bool(value < 0.05)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
