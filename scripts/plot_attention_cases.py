#!/usr/bin/env python3
"""Create the attention-case figure from exported case tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXCLUDED_COLUMNS = {"case", "test_row_id", "top_group", "requested_group", "fallback_used"}
ABBREVIATIONS = {
    "Solvency": "Sol.",
    "Profitability": "Prof.",
    "Operational efficiency": "Oper.",
    "Growth/capital capacity": "Grw.",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HAF-Net attention cases from exported CSV files.")
    parser.add_argument("--attention", required=True, help="case_attention.csv")
    parser.add_argument("--features", required=True, help="case_features.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--prefix", default="attention_cases")
    return parser.parse_args(argv)


def _clean_group_name(name: str) -> str:
    return str(name).replace("Operational efficiency", "Operating-efficiency")


def make_figure(
    attention_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    output_dir: Path,
    prefix: str = "attention_cases",
) -> tuple[Path, Path]:
    group_columns = [column for column in attention_frame.columns if column not in EXCLUDED_COLUMNS]
    if len(attention_frame) != 3:
        raise ValueError("The case figure requires exactly three selected cases.")
    if len(group_columns) != 4:
        raise ValueError("The case figure requires four level-1 attention groups.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Tinos", "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.6,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(10.45, 4.48), squeeze=False)
    dominant_color = "#c43d2b"
    neutral_color = "#ededed"
    negative_color = "#6b96ca"
    positive_color = "#ca6b5d"

    for column, (_, row) in enumerate(attention_frame.reset_index(drop=True).iterrows()):
        case = str(row["case"])
        values = np.asarray([float(row[name]) for name in group_columns], dtype=float)
        dominant_index = int(np.argmax(values))
        colors = [dominant_color if index == dominant_index else neutral_color for index in range(4)]
        bars = axes[0, column].bar(
            np.arange(4), values, width=0.58, color=colors, edgecolor="#a9a9a9", linewidth=0.65
        )
        axes[0, column].set_xticks(
            np.arange(4), [ABBREVIATIONS.get(name, name) for name in group_columns]
        )
        axes[0, column].set_ylim(0.0, 0.65)
        actual_group = _clean_group_name(str(row["top_group"]))
        axes[0, column].set_title(
            f"({chr(97 + column)}) {case}: {actual_group}-dominant",
            fontsize=9.6,
            fontweight="bold",
            pad=6,
        )
        if column == 0:
            axes[0, column].set_ylabel(r"Group attention weight $\alpha_k$", labelpad=4)
        axes[0, column].grid(axis="y", color="#e3e3e3", linewidth=0.55)
        axes[0, column].spines["top"].set_visible(False)
        axes[0, column].spines["right"].set_visible(False)
        for index, bar in enumerate(bars):
            axes[0, column].text(
                bar.get_x() + bar.get_width() / 2,
                values[index] + 0.015,
                f"{values[index]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color=dominant_color if index == dominant_index else "#555555",
                fontweight="bold" if index == dominant_index else "normal",
            )

        block = feature_frame[feature_frame["case"].astype(str) == case].copy().sort_values("rank")
        if len(block) != 3:
            raise ValueError(f"{case} must contain exactly three indicator rows.")
        values_feature = block["standardized_value"].to_numpy(dtype=float)
        positions = np.arange(3)
        colors_feature = [positive_color if value >= 0 else negative_color for value in values_feature]
        axes[1, column].barh(
            positions, values_feature, color=colors_feature, height=0.56, edgecolor="none"
        )
        axes[1, column].set_yticks(positions, [str(label) for label in block["feature_label"]], fontsize=7.9)
        axes[1, column].invert_yaxis()
        axes[1, column].axvline(0.0, color="#777777", linewidth=0.75)
        limit = max(3.0, float(np.ceil((np.max(np.abs(values_feature)) + 0.15) * 2) / 2))
        axes[1, column].set_xlim(-limit, limit)
        axes[1, column].grid(axis="x", color="#ececec", linewidth=0.5)
        axes[1, column].spines["top"].set_visible(False)
        axes[1, column].spines["right"].set_visible(False)
        if column == 1:
            axes[1, column].set_xlabel("Standardized indicator value (z-score)", labelpad=4)
        for y_position, value in zip(positions, values_feature):
            offset = 0.10 if value >= 0 else -0.10
            axes[1, column].text(
                value + offset,
                y_position,
                f"{value:+.1f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8.2,
                color="#444444",
            )

    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.91, wspace=0.38, hspace=0.46)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{prefix}.pdf"
    png_path = output_dir / f"{prefix}.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=600)
    plt.close(figure)
    return pdf_path, png_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    attention_path = Path(args.attention)
    feature_path = Path(args.features)
    output = Path(args.output) if args.output else attention_path.parent
    attention = pd.read_csv(attention_path)
    features = pd.read_csv(feature_path)
    pdf_path, png_path = make_figure(attention, features, output, args.prefix)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
