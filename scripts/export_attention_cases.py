#!/usr/bin/env python3
"""Export selected Taiwan positive cases and their group-attention profiles."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hafnet.attention_cases import resolve_checkpoint_seed, select_cases
from hafnet.config import load_config
from hafnet.data import prepare_dataset
from hafnet.model import HAFNet
from hafnet.trainer import predict_probabilities
from hafnet.utils import ensure_dir, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export positive-case group attention from a trained Taiwan HAF-Net model."
    )
    parser.add_argument("--config", default="configs/taiwan.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--seed", type=int, default=None)
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
    from hafnet.provenance import data_contract

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise ValueError("A HAF-Net checkpoint bundle is required.")
    if bundle["data_contract"] != data_contract(config) or bundle["config"]["model"] != config["model"]:
        raise ValueError("Checkpoint and requested data/model configuration differ.")
    if bundle["variant"] != "full":
        raise ValueError("Attention-case export requires the full HAF-Net configuration.")
    expected_seed = resolve_checkpoint_seed(checkpoint, None)
    if int(bundle["seed"]) != expected_seed:
        raise ValueError("Checkpoint seed metadata and filename disagree.")
    model.load_state_dict(bundle["state_dict"])
    return model.to(device)


def group_feature_indices(dataset, group_index: int) -> list[int]:
    group = dataset.hierarchy.groups[group_index]
    return [index for subgroup in group.subgroups for index in subgroup.indices]


def export_tables(
    dataset,
    attention: np.ndarray,
    selected: list[tuple[int, int, bool]],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_names = [group.name for group in dataset.hierarchy.groups]
    attention_rows: list[dict] = []
    feature_rows: list[dict] = []

    for case_number, (row_index, requested_group, fallback) in enumerate(selected, start=1):
        case_label = f"Case {chr(64 + case_number)}"
        weights = attention[row_index]
        top_group = int(np.argmax(weights))
        attention_row = {
            "case": case_label,
            "test_row_id": int(dataset.test.row_ids[row_index]),
            "top_group": group_names[top_group],
            "requested_group": group_names[requested_group],
            "fallback_used": bool(fallback),
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
                    "case": case_label,
                    "requested_group": group_names[requested_group],
                    "fallback_used": bool(fallback),
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    from hafnet.provenance import load_selected_config

    config, _ = load_selected_config(config, args.parameters)
    if str(config["dataset"]["name"]).lower() != "taiwan":
        raise ValueError("Attention-case analysis is defined for the Taiwan benchmark.")

    checkpoint = Path(args.checkpoint)
    selection_seed = resolve_checkpoint_seed(checkpoint, args.seed)
    dataset = prepare_dataset(config)
    device = resolve_device(str(config["training"].get("device", "auto")))
    model = build_model(dataset, config, checkpoint, device)
    _, attention, _ = predict_probabilities(model, dataset.test, device)
    if attention is None:
        raise RuntimeError("The selected model does not expose group attention.")

    output = ensure_dir(args.output)
    selected = select_cases(attention, dataset.test.y)
    attention_frame, feature_frame = export_tables(dataset, attention, selected, output)
    write_json(
        output / "case_selection.json",
        {
            "split": "test",
            "target_class": 1,
            "selection_rule": (
                "highest attention among positives dominated by each prespecified group; "
                "if none exists, highest attention for that group among remaining positives"
            ),
            "seed": selection_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model_parameters": config["model"],
            "target_groups": [dataset.hierarchy.groups[index].name for index in (0, 1, 2)],
            "selection": [
                {
                    "case": f"Case {chr(64 + case_number)}",
                    "target_group": dataset.hierarchy.groups[group_index].name,
                    "row_id": int(dataset.test.row_ids[row_index]),
                    "dominant_group": dataset.hierarchy.groups[int(np.argmax(attention[row_index]))].name,
                    "fallback_used": bool(fallback),
                }
                for case_number, (row_index, group_index, fallback) in enumerate(selected, start=1)
            ],
        },
    )
    from plot_attention_cases import make_figure

    make_figure(attention_frame, feature_frame, output)
    print(attention_frame.to_string(index=False))


if __name__ == "__main__":
    main()
