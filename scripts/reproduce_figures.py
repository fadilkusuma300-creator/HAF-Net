#!/usr/bin/env python3
"""Regenerate manuscript Figures 3 and 4 from the values reported in the article."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plot_attention_cases import make_figure as make_attention_figure
from plot_sensitivity import make_figure as make_sensitivity_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the manuscript attention-case and sensitivity figures."
    )
    parser.add_argument(
        "--output",
        default="figures",
        help="Directory for generated PDF and PNG files (default: figures)",
    )
    return parser.parse_args()


def manuscript_attention_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    attention = pd.DataFrame(
        [
            ["Case A", "", "Solvency", "Solvency", False, 0.51, 0.18, 0.16, 0.15],
            ["Case B", "", "Profitability", "Profitability", False, 0.21, 0.42, 0.20, 0.17],
            ["Case C", "", "Operational efficiency", "Operational efficiency", False, 0.18, 0.22, 0.46, 0.14],
        ],
        columns=[
            "case", "test_row_id", "top_group", "requested_group", "fallback_used",
            "Solvency", "Profitability", "Operational efficiency", "Growth/capital capacity",
        ],
    )
    features = pd.DataFrame(
        [
            ["Case A", "Solvency", False, 1, "Solvency", "", "Debt/Asset", 2.1],
            ["Case A", "Solvency", False, 2, "Solvency", "", "Interest Cov.", -0.9],
            ["Case A", "Solvency", False, 3, "Solvency", "", "Current Ratio", -0.8],
            ["Case B", "Profitability", False, 1, "Profitability", "", "ROE", -1.7],
            ["Case B", "Profitability", False, 2, "Profitability", "", "Net Margin", -1.3],
            ["Case B", "Profitability", False, 3, "Profitability", "", "Gross Margin", -1.1],
            ["Case C", "Operational efficiency", False, 1, "Operational efficiency", "", "A/R Turnover", -0.4],
            ["Case C", "Operational efficiency", False, 2, "Operational efficiency", "", "Inv. Turnover", -0.6],
            ["Case C", "Operational efficiency", False, 3, "Operational efficiency", "", "Asset Turnover", -1.1],
        ],
        columns=[
            "case", "requested_group", "fallback_used", "rank", "group", "feature_id",
            "feature_label", "standardized_value",
        ],
    )
    return attention, features


def manuscript_sensitivity_table() -> pd.DataFrame:
    rows = [
        ("lambda_tcar", 0.00, 0.614, 0.006),
        ("lambda_tcar", 0.02, 0.638, 0.005),
        ("lambda_tcar", 0.05, 0.655, 0.004),
        ("lambda_tcar", 0.10, 0.650, 0.005),
        ("lambda_tcar", 0.20, 0.623, 0.007),
        ("beta", 0.90, 0.608, 0.007),
        ("beta", 0.99, 0.633, 0.005),
        ("beta", 0.999, 0.654, 0.004),
        ("beta", 0.9999, 0.639, 0.006),
        ("gamma", 0.0, 0.608, 0.006),
        ("gamma", 1.0, 0.643, 0.005),
        ("gamma", 2.0, 0.654, 0.004),
        ("gamma", 3.0, 0.629, 0.006),
        ("gamma", 5.0, 0.591, 0.008),
        ("rho", 0.00, 0.625, 0.006),
        ("rho", 0.25, 0.643, 0.005),
        ("rho", 0.50, 0.655, 0.004),
        ("rho", 0.75, 0.652, 0.004),
        ("rho", 1.00, 0.638, 0.006),
    ]
    return pd.DataFrame(
        rows,
        columns=["parameter", "value", "validation_pr_auc_mean", "validation_pr_auc_std"],
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    attention, features = manuscript_attention_tables()
    make_attention_figure(attention, features, output_dir, prefix="Fig3")

    sensitivity = manuscript_sensitivity_table()
    selected = {"lambda_tcar": 0.05, "beta": 0.999, "gamma": 2.0, "rho": 0.50}
    make_sensitivity_figure(sensitivity, output_dir, selected, prefix="Fig4")

    print(output_dir / "Fig3.pdf")
    print(output_dir / "Fig4.pdf")


if __name__ == "__main__":
    main()
