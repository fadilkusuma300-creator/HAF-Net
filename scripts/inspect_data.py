#!/usr/bin/env python3
"""Inspect dataset dimensions, class balance, split sizes, and feature hierarchy."""
from __future__ import annotations

import argparse

import numpy as np

from hafnet.config import load_config
from hafnet.data import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display HAF-Net dataset and hierarchy information.")
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the Taiwan stratified split")
    return parser.parse_args()


def class_summary(y: np.ndarray) -> str:
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    rate = positive / max(len(y), 1)
    return f"n={len(y)}, positive={positive}, negative={negative}, positive_rate={rate:.4%}"


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    dataset = prepare_dataset(config, args.seed)

    print(f"dataset: {dataset.name}")
    print(f"features: {len(dataset.feature_ids)}")
    print(f"level-1 groups: {len(dataset.hierarchy.groups)}")
    print(f"level-2 subgroups: {dataset.hierarchy.subgroup_count}")
    print(f"level-1 group sizes: {dataset.hierarchy.group_sizes}")
    print(f"level-2 subgroup sizes: {dataset.hierarchy.subgroup_sizes}")
    print(f"train: {class_summary(dataset.train.y)}")
    print(f"validation: {class_summary(dataset.validation.y)}")
    print(f"test: {class_summary(dataset.test.y)}")
    print("groups:")
    for group in dataset.hierarchy.groups:
        print(f"  - {group.name}")
        for subgroup in group.subgroups:
            print(f"      {subgroup.level2}: {len(subgroup.indices)} features")
