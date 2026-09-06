#!/usr/bin/env python3
"""Export representative Taiwan positive cases and their group-attention profiles."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hafnet.config import load_config
from hafnet.data import prepare_dataset
from hafnet.model import HAFNet
from hafnet.trainer import predict_probabilities
from hafnet.utils import ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export positive-case group attention from a trained Taiwan HAF-Net model.")
    parser.add_argument("--config", default="configs/taiwan.yaml")
    parser.add_argument("--checkpoint", required=True, help="State dictionary produced by scripts/run_experiment.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/attention_cases")
    return parser.parse_args()


def build_model(dataset, config, checkpoint: Path, device: torch.device) -> HAFNet:
    model = HAFNet(
        input_dim=len(dataset.feature_ids),
        hierarchy=dataset.hierarchy,
        subgroup_dim=int(config["model"].get("subgroup_dim", 32)),
        fusion_dim=int(config["model"].get("fusion_dim", 64)),
        dropout=float(config["model"].get("dropout", 0.1)),
        use_residual=True,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device)


def select_cases(attention: np.ndarray, y: np.ndarray, desired_groups: tuple[int, ...] = (0, 1, 2)) -> list[int]:
    positive_indices = np.flatnonzero(y == 1)
    if len(positive_indices) == 0:
        raise ValueError("The test partition contains no positive observations.")
    selected: list[int] = []
    dominant = np.argmax(attention, axis=1)
    for group_index in desired_groups:
        candidates = positive_indices[dominant[positive_indices] == group_index]
        if len(candidates):
            case = int(candidates[np.argmax(attention[candidates, group_index])])
        else:
            remaining = np.asarray([index for index in positive_indices if int(index) not in selected], dtype=int)
            if len(remaining) == 0:
                remaining = positive_indices
            case = int(remaining[np.argmax(attention[remaining, group_index])])
        selected.append(case)
    return selected


def group_feature_indices(dataset, group_index: int) -> list[int]:
    group = dataset.hierarchy.groups[group_index]
    return [index for subgroup in group.subgroups for index in subgroup.indices]


def export_tables(dataset, attention: np.ndarray, selected: list[int], output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_names = [group.name for group in dataset.hierarchy.groups]
    attention_rows: list[dict] = []
    feature_rows: list[dict] = []

    for case_number, row_index in enumerate(selected, start=1):
        weights = attention[row_index]
        top_group = int(np.argmax(weights))
        attention_row = {
            "case": f"Case {case_number}",
            "test_row_id": int(dataset.test.row_ids[row_index]),
            "top_group": group_names[top_group],
        }
        attention_row.update({name: float(value) for name, value in zip(group_names, weights)})
        attention_rows.append(attention_row)

        indices = group_feature_indices(dataset, top_group)
        values = dataset.test.x[row_index, indices]
        ordering = np.argsort(np.abs(values))[::-1][:3]
        for rank, local_position in enumerate(ordering, start=1):
            feature_position = indices[int(local_position)]
            feature_rows.append(
                {
                    "case": f"Case {case_number}",
                    "rank": rank,
                    "group": group_names[top_group],
                    "feature_id": dataset.feature_ids[feature_position],
                    "feature_label": dataset.feature_labels[feature_position],
                    "standardized_value": float(dataset.test.x[row_index, feature_position]),
                }
            )

    attention_frame = pd.DataFrame(attention_rows)
    feature_frame = pd.DataFrame(feature_rows)
    attention_frame.to_csv(output / "case_attention.csv", index=False)
    feature_frame.to_csv(output / "case_features.csv", index=False)
    return attention_frame, feature_frame


def make_figure(attention_frame: pd.DataFrame, feature_frame: pd.DataFrame, output: Path) -> None:
    group_columns = [column for column in attention_frame.columns if column not in {"case", "test_row_id", "top_group"}]
    figure, axes = plt.subplots(len(attention_frame), 2, figsize=(10, 3.2 * len(attention_frame)), squeeze=False)
    for row, case in enumerate(attention_frame["case"]):
        attention_row = attention_frame.iloc[row]
        axes[row, 0].bar(group_columns, [attention_row[column] for column in group_columns])
        axes[row, 0].set_ylim(0.0, 1.0)
        axes[row, 0].set_ylabel("Group attention")
        axes[row, 0].set_title(f"{case}: {attention_row['top_group']}")
        axes[row, 0].tick_params(axis="x", rotation=25)

        block = feature_frame[feature_frame["case"] == case]
        axes[row, 1].barh(block["feature_id"], block["standardized_value"])
        axes[row, 1].axvline(0.0, linewidth=0.8)
        axes[row, 1].set_xlabel("Standardized value")
        axes[row, 1].set_title("Largest absolute indicators in the dominant group")

    figure.tight_layout()
    figure.savefig(output / "attention_cases.pdf", bbox_inches="tight")
    figure.savefig(output / "attention_cases.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if str(config["dataset"]["name"]).lower() != "taiwan":
        raise ValueError("The case-analysis figure is defined for the Taiwan benchmark.")

    dataset = prepare_dataset(config, args.seed)
    device = resolve_device(str(config["training"].get("device", "auto")))
    model = build_model(dataset, config, Path(args.checkpoint), device)
    _, attention = predict_probabilities(model, dataset.test, device)
    if attention is None:
        raise RuntimeError("The selected model does not expose group attention.")

    output = ensure_dir(args.output)
    selected = select_cases(attention, dataset.test.y)
    attention_frame, feature_frame = export_tables(dataset, attention, selected, output)
    make_figure(attention_frame, feature_frame, output)
    print(attention_frame.to_string(index=False))
