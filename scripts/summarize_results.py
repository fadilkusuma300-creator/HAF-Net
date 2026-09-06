#!/usr/bin/env python3
"""Create compact Markdown summaries from run-level experiment CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "gmean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HAF-Net or baseline run tables.")
    parser.add_argument("--input", required=True, help="CSV containing run-level metrics")
    parser.add_argument("--output", default=None, help="Optional Markdown output path")
    return parser.parse_args()


def summarize_hafnet(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, block in frame.groupby("variant", sort=False):
        row = {"model": variant}
        for metric in METRICS:
            values = block[f"test_{metric}"].astype(float)
            row[metric] = f"{values.mean():.4f} ± {values.std(ddof=1):.4f}" if len(values) > 1 else f"{values.mean():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, block in frame.groupby("model", sort=False):
        row = {"model": model}
        for metric in METRICS:
            values = block[f"test_{metric}"].astype(float)
            row[metric] = f"{values.mean():.4f} ± {values.std(ddof=1):.4f}" if len(values) > 1 else f"{values.mean():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = parse_args()
    frame = pd.read_csv(args.input)
    if "variant" in frame.columns:
        summary = summarize_hafnet(frame)
    elif "model" in frame.columns:
        summary = summarize_baselines(frame)
    else:
        raise ValueError("Input CSV must contain either a 'variant' or 'model' column.")

    markdown = summary.to_markdown(index=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
