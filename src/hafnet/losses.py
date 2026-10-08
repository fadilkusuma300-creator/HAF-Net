from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def effective_number_weights(class_counts: tuple[int, int], beta: float) -> torch.Tensor:
    """Evaluate effective-number weights in float64, including beta near one."""
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("beta must satisfy 0 <= beta < 1.")
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    if counts.shape != (2,) or not torch.isfinite(counts).all() or torch.any(counts <= 0) or not torch.equal(counts, counts.floor()):
        raise ValueError("Exactly two positive integer class counts are required.")
    if beta == 0:
        raw = torch.ones_like(counts)
    else:
        raw = (1.0 - beta) / (-torch.expm1(counts * math.log(beta)))
    return (2.0 * raw / raw.sum()).to(torch.float32)


def _binary_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape != targets.shape or logits.numel() == 0:
        raise ValueError("Logits and targets must have the same nonempty shape.")
    if not bool(torch.isfinite(logits).all()) or not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("Finite logits and binary targets are required.")
    targets = targets.to(dtype=logits.dtype)
    return targets, F.binary_cross_entropy_with_logits(logits, targets, reduction="none")


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss with mean-one normalized class weights."""

    def __init__(self, class_counts: tuple[int, int], beta: float = 0.999, gamma: float = 2.0) -> None:
        super().__init__()
        if not 0 <= beta < 1:
            raise ValueError("beta must satisfy 0 <= beta < 1.")
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative.")
        normalized_weights = effective_number_weights(class_counts, beta)
        self.gamma = float(gamma)
        self.register_buffer("class_weights", normalized_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets, cross_entropy = _binary_cross_entropy(logits, targets)
        pt = torch.exp(-cross_entropy)
        weights = self.class_weights[targets.long()]
        return (weights * (1.0 - pt).pow(self.gamma) * cross_entropy).mean()


class ClassBalancedBCELoss(nn.Module):
    """Binary cross-entropy with effective-number class weights."""

    def __init__(self, class_counts: tuple[int, int], beta: float = 0.999) -> None:
        super().__init__()
        if not 0 <= beta < 1:
            raise ValueError("beta must satisfy 0 <= beta < 1.")
        weights = effective_number_weights(class_counts, beta)
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets, losses = _binary_cross_entropy(logits, targets)
        return (self.class_weights[targets.long()] * losses).mean()


class FocalLoss(nn.Module):
    """Focal loss with equal class weights."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative.")
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _, cross_entropy = _binary_cross_entropy(logits, targets)
        return ((1.0 - torch.exp(-cross_entropy)).pow(self.gamma) * cross_entropy).mean()


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
    """InfoNCE with one positive and J same-year negative-labeled rows per anchor."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    if anchor_z.ndim != 2 or positive_z.shape != anchor_z.shape or negative_z.ndim != 3:
        raise ValueError("InfoNCE expects anchor/positive [B,D] and negatives [B,J,D].")
    if anchor_z.shape[0] == 0 or negative_z.shape[1] < 1 or negative_z.shape[0] != anchor_z.shape[0] or negative_z.shape[2] != anchor_z.shape[1]:
        raise ValueError("InfoNCE requires compatible nonempty batches and at least one negative.")
    anchor = F.normalize(anchor_z, dim=-1)
    positive = F.normalize(positive_z, dim=-1)
    negative = F.normalize(negative_z, dim=-1)
    positive_logit = torch.sum(anchor * positive, dim=-1, keepdim=True) / temperature
    negative_logits = torch.einsum("bd,bjd->bj", anchor, negative) / temperature
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
