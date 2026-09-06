from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL_GROUP_ORDER = {
    "taiwan": [
        "Solvency",
        "Profitability",
        "Operational efficiency",
        "Growth/capital capacity",
    ],
    "american": [
        "Solvency/obligations",
        "Profitability/earnings",
        "Operating activity",
        "Scale/market position",
    ],
}

CANONICAL_SUBGROUP_ORDER = {
    "taiwan": {
        "Solvency": ["Liquidity and short-term coverage", "Leverage and debt service"],
        "Profitability": ["Returns and margins", "Earnings quality and cost burden"],
        "Operational efficiency": [
            "Asset and working-capital turnover",
            "Resource productivity and cash conversion",
        ],
        "Growth/capital capacity": ["Growth and reinvestment", "Capital base and per-share capacity"],
    },
    "american": {
        "Solvency/obligations": ["Current liquidity position", "Long-term and total obligations"],
        "Profitability/earnings": ["Operating earnings", "Net and accumulated earnings"],
        "Operating activity": ["Cost structure", "Working-capital activity"],
        "Scale/market position": ["Market and asset scale", "Revenue and sales scale"],
    },
}


@dataclass(frozen=True)
class SubgroupSpec:
    """One level-2 subgroup and its feature positions."""

    level1: str
    level2: str
    feature_ids: tuple[str, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True)
class GroupSpec:
    """One level-1 financial group."""

    name: str
    subgroups: tuple[SubgroupSpec, ...]


@dataclass(frozen=True)
class HierarchySpec:
    """Complete two-level feature hierarchy for one dataset."""

    dataset: str
    feature_ids: tuple[str, ...]
    groups: tuple[GroupSpec, ...]

    @property
    def subgroup_count(self) -> int:
        return sum(len(group.subgroups) for group in self.groups)

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(sum(len(subgroup.indices) for subgroup in group.subgroups) for group in self.groups)

    @property
    def subgroup_sizes(self) -> tuple[int, ...]:
        return tuple(len(subgroup.indices) for group in self.groups for subgroup in group.subgroups)


def load_mapping(mapping_path: str | Path, dataset: str) -> pd.DataFrame:
    """Load the machine-readable feature hierarchy for a selected dataset."""
    frame = pd.read_csv(mapping_path)
    required = {"dataset", "feature_id", "feature_label", "level1_group", "level2_subgroup"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Mapping file is missing columns: {sorted(missing)}")
    selected = frame[frame["dataset"].str.lower() == dataset.lower()].copy()
    if selected.empty:
        raise ValueError(f"No feature mapping found for dataset '{dataset}'.")
    if selected["feature_id"].duplicated().any():
        duplicates = selected.loc[selected["feature_id"].duplicated(), "feature_id"].tolist()
        raise ValueError(f"Duplicate feature IDs in mapping: {duplicates}")
    return selected.reset_index(drop=True)


def build_hierarchy(mapping_path: str | Path, dataset: str, feature_ids: Iterable[str]) -> HierarchySpec:
    """Construct the fixed four-group/eight-subgroup hierarchy used by HFGE."""
    feature_ids = tuple(str(feature_id) for feature_id in feature_ids)
    positions = {feature_id: index for index, feature_id in enumerate(feature_ids)}
    mapping = load_mapping(mapping_path, dataset)

    expected = set(mapping["feature_id"])
    observed = set(feature_ids)
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Feature IDs do not match the mapping. Missing={missing}; extra={extra}")

    grouped: OrderedDict[str, OrderedDict[str, list[str]]] = OrderedDict()
    for row in mapping.itertuples(index=False):
        grouped.setdefault(row.level1_group, OrderedDict()).setdefault(row.level2_subgroup, []).append(row.feature_id)

    dataset_key = dataset.lower()
    level1_order = CANONICAL_GROUP_ORDER[dataset_key]
    if set(level1_order) != set(grouped.keys()):
        raise ValueError(f"Level-1 group names do not match the expected {dataset} hierarchy.")

    group_specs: list[GroupSpec] = []
    for level1 in level1_order:
        subgroup_map = grouped[level1]
        level2_order = CANONICAL_SUBGROUP_ORDER[dataset_key][level1]
        if set(level2_order) != set(subgroup_map.keys()):
            raise ValueError(f"Subgroup names do not match the expected hierarchy for {level1}.")
        subgroup_specs: list[SubgroupSpec] = []
        for level2 in level2_order:
            ids = tuple(subgroup_map[level2])
            subgroup_specs.append(
                SubgroupSpec(
                    level1=level1,
                    level2=level2,
                    feature_ids=ids,
                    indices=tuple(positions[feature_id] for feature_id in ids),
                )
            )
        group_specs.append(GroupSpec(name=level1, subgroups=tuple(subgroup_specs)))

    hierarchy = HierarchySpec(dataset=dataset_key, feature_ids=feature_ids, groups=tuple(group_specs))
    if len(hierarchy.groups) != 4 or hierarchy.subgroup_count != 8:
        raise ValueError("HAF-Net requires four level-1 groups and eight level-2 subgroups.")
    flat_indices = [index for group in hierarchy.groups for subgroup in group.subgroups for index in subgroup.indices]
    if sorted(flat_indices) != list(range(len(feature_ids))):
        raise ValueError("Every predictive feature must occur exactly once in the hierarchy.")
    return hierarchy


def randomized_hierarchy(hierarchy: HierarchySpec, seed: int) -> HierarchySpec:
    """Randomize feature assignments while preserving every subgroup cardinality."""
    rng = np.random.default_rng(seed)
    shuffled_ids = list(hierarchy.feature_ids)
    rng.shuffle(shuffled_ids)
    original_positions = {feature_id: index for index, feature_id in enumerate(hierarchy.feature_ids)}

    offset = 0
    randomized_groups: list[GroupSpec] = []
    for group in hierarchy.groups:
        randomized_subgroups: list[SubgroupSpec] = []
        for subgroup in group.subgroups:
            size = len(subgroup.feature_ids)
            selected_ids = tuple(shuffled_ids[offset : offset + size])
            offset += size
            randomized_subgroups.append(
                SubgroupSpec(
                    level1=group.name,
                    level2=subgroup.level2,
                    feature_ids=selected_ids,
                    indices=tuple(original_positions[feature_id] for feature_id in selected_ids),
                )
            )
        randomized_groups.append(GroupSpec(name=group.name, subgroups=tuple(randomized_subgroups)))

    return HierarchySpec(
        dataset=hierarchy.dataset,
        feature_ids=hierarchy.feature_ids,
        groups=tuple(randomized_groups),
    )
