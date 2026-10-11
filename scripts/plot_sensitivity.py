#!/usr/bin/env python3
"""Create the HAF-Net hyperparameter sensitivity figure from summary results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARAMETER_ORDER = ("lambda_tcar", "beta", "gamma", "rho")
PARAMETER_LABELS = {
    "lambda_tcar": r"$\lambda_{\mathrm{TCAR}}$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
    "rho": r"$\rho$",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HAF-Net validation sensitivity from a summary CSV.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--prefix", default="sensitivity")
    return parser.parse_args(argv)


def _selected_values(manifest_path: str | None, summary: pd.DataFrame) -> dict[str, float]:
    if manifest_path:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        values = payload.get("reference_values", {})
        return {str(key): float(value) for key, value in values.items()}
    defaults = {"lambda_tcar": 0.05, "beta": 0.999, "gamma": 2.0, "rho": 0.50}
    available = set(summary["parameter"].astype(str))
    return {key: value for key, value in defaults.items() if key in available}


def _shared_y_limits(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0, np.linspace(0.0, 1.0, 6)
    step = 0.02
    lower = np.floor((values.min() - 0.008) / step) * step
    upper = np.ceil((values.max() + 0.008) / step) * step
    if upper - lower < 0.08:
        midpoint = (upper + lower) / 2.0
        lower = np.floor((midpoint - 0.04) / step) * step
        upper = np.ceil((midpoint + 0.04) / step) * step
    ticks = np.arange(lower, upper + step / 2.0, step)
    return float(lower), float(upper), ticks


def make_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    selected_values: dict[str, float],
    prefix: str = "sensitivity",
) -> tuple[Path, Path]:
    required = {"parameter", "value", "validation_pr_auc_mean"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Sensitivity summary is missing columns: {sorted(missing)}")
    parameters = [p for p in PARAMETER_ORDER if p in set(summary["parameter"].astype(str))]
    if not parameters:
        raise ValueError("Sensitivity summary does not contain any supported parameter.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Tinos", "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    figure, axes = plt.subplots(1, len(parameters), figsize=(10.12, 2.28), sharey=True, squeeze=False)
    axes = axes[0]
    y_min, y_max, y_ticks = _shared_y_limits(summary["validation_pr_auc_mean"].to_numpy(dtype=float))

    line_color = "#3b6fb6"
    selected_color = "#d14a2f"
    for panel_index, (axis, parameter) in enumerate(zip(axes, parameters)):
        block = summary[summary["parameter"].astype(str) == parameter].copy().sort_values("value").reset_index(drop=True)
        means = block["validation_pr_auc_mean"].to_numpy(dtype=float)
        if parameter == "beta":
            x = np.arange(len(block), dtype=float)
            x_labels = [f"{value:g}" for value in block["value"].to_numpy(dtype=float)]
        else:
            x = block["value"].to_numpy(dtype=float)
            x_labels = None

        axis.plot(
            x,
            means,
            color=line_color,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=line_color,
            markeredgewidth=1.1,
            linewidth=1.45,
            markersize=4.2,
        )
        selected_value = selected_values.get(parameter)
        if selected_value is not None:
            mask = np.isclose(block["value"].to_numpy(dtype=float), float(selected_value))
            if np.any(mask):
                index = int(np.flatnonzero(mask)[0])
                axis.scatter(x[index], means[index], s=28, color=selected_color, zorder=4)

        if x_labels is not None:
            axis.set_xticks(x, x_labels)
        elif parameter in {"lambda_tcar", "rho"}:
            axis.set_xticks(block["value"].to_numpy(dtype=float), [f"{v:.2f}" for v in block["value"]])
        else:
            axis.set_xticks(block["value"].to_numpy(dtype=float), [f"{v:g}" for v in block["value"]])

        axis.set_ylim(y_min, y_max)
        axis.set_yticks(y_ticks)
        axis.set_xlabel(PARAMETER_LABELS[parameter], labelpad=2)
        axis.text(
            0.02,
            0.98,
            f"({chr(97 + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
        )
        axis.grid(axis="y", color="#dedede", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=8.2, pad=2)
        if panel_index == 0:
            axis.set_ylabel("Validation average precision", labelpad=4)

    figure.subplots_adjust(left=0.073, right=0.995, bottom=0.27, top=0.975, wspace=0.18)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{prefix}.pdf"
    png_path = output_dir / f"{prefix}.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=600)
    plt.close(figure)
    return pdf_path, png_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary_path = Path(args.summary)
    summary = pd.read_csv(summary_path)
    selected_values = _selected_values(args.manifest, summary)
    output = Path(args.output) if args.output else summary_path.parent
    pdf_path, png_path = make_figure(summary, output, selected_values, args.prefix)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
