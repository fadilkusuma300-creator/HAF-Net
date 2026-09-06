from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .grouping import HierarchySpec


class SubgroupEncoder(nn.Module):
    """Two-layer MLP used independently for each level-2 feature subgroup."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdditiveAttention(nn.Module):
    """Additive attention over a collection of equal-width vectors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim, bias=True)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(torch.tanh(self.projection(values))).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(values * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class HFGE(nn.Module):
    """Hierarchical Feature Grouping Encoder."""

    def __init__(
        self,
        input_dim: int,
        hierarchy: HierarchySpec,
        subgroup_dim: int = 32,
        fusion_dim: int = 64,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if input_dim != len(hierarchy.feature_ids):
            raise ValueError("Input dimension and hierarchy feature count must match.")
        self.hierarchy = hierarchy
        self.subgroup_dim = int(subgroup_dim)
        self.fusion_dim = int(fusion_dim)
        self.use_residual = bool(use_residual)

        self.subgroup_encoders = nn.ModuleList()
        self.intra_attention = nn.ModuleList()
        self._subgroup_indices: list[list[torch.Tensor]] = []

        for group in hierarchy.groups:
            group_encoders = nn.ModuleList()
            group_indices: list[torch.Tensor] = []
            for subgroup in group.subgroups:
                group_encoders.append(SubgroupEncoder(len(subgroup.indices), self.subgroup_dim, dropout))
                group_indices.append(torch.tensor(subgroup.indices, dtype=torch.long))
            self.subgroup_encoders.append(group_encoders)
            self.intra_attention.append(AdditiveAttention(self.subgroup_dim))
            self._subgroup_indices.append(group_indices)

        self.inter_attention = AdditiveAttention(self.subgroup_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.subgroup_dim, self.fusion_dim),
            nn.Dropout(dropout),
        )
        self.residual = nn.Linear(input_dim, self.fusion_dim) if self.use_residual else None

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        group_vectors: list[torch.Tensor] = []
        subgroup_weights: list[torch.Tensor] = []

        for group_index, group_encoders in enumerate(self.subgroup_encoders):
            encoded_subgroups: list[torch.Tensor] = []
            for subgroup_index, encoder in enumerate(group_encoders):
                indices = self._subgroup_indices[group_index][subgroup_index].to(x.device)
                subgroup_input = torch.index_select(x, dim=1, index=indices)
                encoded_subgroups.append(encoder(subgroup_input))
            subgroup_stack = torch.stack(encoded_subgroups, dim=1)
            group_vector, weights = self.intra_attention[group_index](subgroup_stack)
            group_vectors.append(group_vector)
            subgroup_weights.append(weights)

        group_stack = torch.stack(group_vectors, dim=1)
        pooled, group_attention = self.inter_attention(group_stack)
        z = self.fusion(pooled)
        if self.residual is not None:
            z = z + self.residual(x)
        return {
            "z": z,
            "group_attention": group_attention,
            "subgroup_attention": subgroup_weights,
            "group_vectors": group_stack,
        }


class HAFNet(nn.Module):
    """HAF-Net encoder followed by a single linear binary classifier."""

    def __init__(
        self,
        input_dim: int,
        hierarchy: HierarchySpec,
        subgroup_dim: int = 32,
        fusion_dim: int = 64,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = HFGE(
            input_dim=input_dim,
            hierarchy=hierarchy,
            subgroup_dim=subgroup_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            use_residual=use_residual,
        )
        self.classifier = nn.Linear(fusion_dim, 1)

    def encode(self, x: torch.Tensor) -> dict[str, Any]:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        encoded = self.encoder(x)
        logits = self.classifier(encoded["z"]).squeeze(-1)
        return {**encoded, "logits": logits, "probability": torch.sigmoid(logits)}


class FlatMLP(nn.Module):
    """Parameter-scale-matched flat MLP used in the representation controls."""

    def __init__(
        self,
        input_dim: int,
        fusion_dim: int = 64,
        dropout: float = 0.1,
        target_parameters: int | None = None,
    ) -> None:
        super().__init__()
        hidden1, hidden2 = self._choose_widths(input_dim, fusion_dim, target_parameters)
        self.hidden_widths = (hidden1, hidden2)
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, fusion_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(fusion_dim, 1)

    @staticmethod
    def _choose_widths(input_dim: int, fusion_dim: int, target_parameters: int | None) -> tuple[int, int]:
        if target_parameters is None:
            return fusion_dim, fusion_dim
        best: tuple[int, int, int, int] | None = None
        for hidden1 in range(16, 257):
            for hidden2 in range(16, 257):
                count = (
                    (input_dim + 1) * hidden1
                    + (hidden1 + 1) * hidden2
                    + (hidden2 + 1) * fusion_dim
                    + (fusion_dim + 1)
                )
                candidate = (abs(count - target_parameters), abs(hidden1 - hidden2), hidden1, hidden2)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        return int(best[2]), int(best[3])

    def encode(self, x: torch.Tensor) -> dict[str, Any]:
        return {"z": self.encoder_net(x), "group_attention": None, "subgroup_attention": None}

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        encoded = self.encode(x)
        logits = self.classifier(encoded["z"]).squeeze(-1)
        return {**encoded, "logits": logits, "probability": torch.sigmoid(logits)}
