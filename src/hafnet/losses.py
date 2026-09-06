from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss with mean-one normalized class weights."""

    def __init__(self, class_counts: tuple[int, int], beta: float = 0.999, gamma: float = 2.0) -> None:
        super().__init__()
        if not 0 <= beta < 1:
            raise ValueError("beta must satisfy 0 <= beta < 1.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        counts = torch.tensor(class_counts, dtype=torch.float32)
        if torch.any(counts <= 0):
            raise ValueError("Both classes must be present in the training partition.")

        beta_tensor = torch.tensor(float(beta), dtype=torch.float32)
        raw_weights = (1.0 - beta_tensor) / (1.0 - torch.pow(beta_tensor, counts))
        normalized_weights = len(class_counts) * raw_weights / raw_weights.sum()
        self.gamma = float(gamma)
        self.register_buffer("class_weights", normalized_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probability = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probability, 1.0 - probability).clamp(1e-7, 1.0 - 1e-7)
        weights = self.class_weights[targets.long()]
        loss = -weights * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        return loss.mean()


def tcar_loss(group_attention: torch.Tensor | None, targets: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Compute KL(prior || attention) over positive observations in the current batch."""
    if group_attention is None:
        return targets.new_zeros((), dtype=torch.float32)
    positive_mask = targets > 0.5
    if not bool(positive_mask.any()):
        return group_attention.new_zeros(())

    attention = group_attention[positive_mask].clamp_min(1e-8)
    prior = prior.to(group_attention.device, dtype=group_attention.dtype).clamp_min(1e-8)
    prior = prior / prior.sum()
    return torch.sum(prior * (torch.log(prior) - torch.log(attention)), dim=1).mean()


def temporal_infonce(
    anchor_z: torch.Tensor,
    positive_z: torch.Tensor,
    negative_z: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE with one positive and J same-year healthy negatives per anchor."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    anchor = F.normalize(anchor_z, dim=-1)
    positive = F.normalize(positive_z, dim=-1)
    negative = F.normalize(negative_z, dim=-1)
    positive_logit = torch.sum(anchor * positive, dim=-1, keepdim=True) / temperature
    negative_logits = torch.einsum("bd,bjd->bj", anchor, negative) / temperature
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
