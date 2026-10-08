from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import PreparedDataset, SplitData
from .grouping import randomized_hierarchy
from .losses import (
    ClassBalancedBCELoss,
    ClassBalancedFocalLoss,
    FocalLoss,
    tcar_loss,
    temporal_infonce,
)
from .metrics import classification_metrics, select_f1_threshold
from .model import FlatMLP, HAFNet
from .prior import combined_prior, permute_prior
from .temporal import TemporalPairBank
from .utils import count_trainable_parameters, resolve_device, set_seed


@dataclass
class RunResult:
    """Outputs from one seed and one HAF-Net configuration."""

    seed: int
    variant: str
    validation: dict[str, float]
    test: dict[str, float]
    epochs: int
    parameters: int
    threshold: float
    group_attention: np.ndarray | None
    test_attention: np.ndarray | None
    subgroup_attention: tuple[np.ndarray, ...] | None
    validation_probability: np.ndarray
    test_probability: np.ndarray | None
    prior: np.ndarray | None
    alignment: dict[str, float] | None
    model_state: dict[str, torch.Tensor]


def _reference_hafnet(dataset: PreparedDataset, config: dict[str, Any]) -> HAFNet:
    model_config = config["model"]
    return HAFNet(
        input_dim=len(dataset.feature_ids),
        hierarchy=dataset.hierarchy,
        subgroup_dim=int(model_config.get("subgroup_dim", 32)),
        fusion_dim=int(model_config.get("fusion_dim", 64)),
        dropout=float(model_config.get("dropout", 0.1)),
        use_residual=True,
    )


def _build_model(dataset: PreparedDataset, config: dict[str, Any], variant: str, seed: int) -> nn.Module:
    model_config = config["model"]
    fusion_dim = int(model_config.get("fusion_dim", 64))
    dropout = float(model_config.get("dropout", 0.1))

    if variant.startswith("flat_"):
        target_parameters = count_trainable_parameters(_reference_hafnet(dataset, config))
        return FlatMLP(
            input_dim=len(dataset.feature_ids),
            fusion_dim=fusion_dim,
            dropout=dropout,
            target_parameters=target_parameters,
        )

    control_seed = int(config.get("controls", {}).get("random_hfge_seed", 2026))
    hierarchy = randomized_hierarchy(dataset.hierarchy, control_seed) if variant == "random_hfge" else dataset.hierarchy
    return HAFNet(
        input_dim=len(dataset.feature_ids),
        hierarchy=hierarchy,
        subgroup_dim=int(model_config.get("subgroup_dim", 32)),
        fusion_dim=fusion_dim,
        dropout=dropout,
        use_residual=variant != "no_residual",
        equal_intragroup=variant == "equal_intragroup",
    )


def predict_probabilities(
    model: nn.Module,
    split: SplitData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, tuple[np.ndarray, ...] | None]:
    """Return probabilities and available attention weights."""
    model.eval()
    x = torch.as_tensor(split.x, dtype=torch.float32)
    probabilities: list[np.ndarray] = []
    attention: list[np.ndarray] = []
    subgroup_attention: list[list[np.ndarray]] = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(device)
            output = model(batch)
            probabilities.append(output["probability"].detach().cpu().numpy())
            if output.get("group_attention") is not None:
                attention.append(output["group_attention"].detach().cpu().numpy())
                if not subgroup_attention:
                    subgroup_attention = [[] for _ in output["subgroup_attention"]]
                for group_index, weights in enumerate(output["subgroup_attention"]):
                    subgroup_attention[group_index].append(weights.detach().cpu().numpy())
    probability = np.concatenate(probabilities)
    group_attention = np.concatenate(attention) if attention else None
    subgroup_weights = (
        tuple(np.concatenate(group_weights) for group_weights in subgroup_attention)
        if subgroup_attention
        else None
    )
    return probability, group_attention, subgroup_weights


def _loss_switches(config: dict[str, Any], variant: str) -> tuple[str, float, float]:
    loss_config = config["loss"]
    loss_mode = "cb_focal"
    if variant.endswith("_class_balanced_bce"):
        loss_mode = "class_balanced_bce"
    elif variant.endswith("_focal_unweighted"):
        loss_mode = "focal_unweighted"
    elif variant.endswith("_bce"):
        loss_mode = "bce"
    lambda_tcar = float(loss_config.get("lambda_tcar", 0.0))
    lambda_tcl = float(loss_config.get("lambda_tcl", 0.0))

    no_auxiliary = {
        "core",
        "hfge_bce",
        "flat_cbf",
        "flat_bce",
        "hfge_class_balanced_bce",
        "hfge_focal_unweighted",
        "random_hfge",
        "equal_intragroup",
    }
    if variant in no_auxiliary:
        lambda_tcar = 0.0
        lambda_tcl = 0.0
    return loss_mode, lambda_tcar, lambda_tcl


def train_one_seed(
    dataset: PreparedDataset,
    config: dict[str, Any],
    seed: int,
    variant: str = "full",
    evaluate_test: bool = True,
) -> RunResult:
    """Train one HAF-Net configuration and evaluate it with a validation-selected threshold."""
    supported = {
        "full",
        "core",
        "hfge_bce",
        "flat_cbf",
        "flat_bce",
        "hfge_class_balanced_bce",
        "hfge_focal_unweighted",
        "random_hfge",
        "no_residual",
        "permuted_prior",
        "uniform_prior",
        "equal_intragroup",
    }
    if variant not in supported:
        raise ValueError(f"Unsupported HAF-Net variant: {variant}")

    set_seed(seed)
    training_config = config["training"]
    loss_config = config["loss"]
    device = resolve_device(str(training_config.get("device", "auto")))
    model = _build_model(dataset, config, variant, seed).to(device)

    y_train = dataset.train.y
    class_counts = (int(np.sum(y_train == 0)), int(np.sum(y_train == 1)))
    cb_focal = ClassBalancedFocalLoss(
        class_counts=class_counts,
        beta=float(loss_config.get("beta", 0.999)),
        gamma=float(loss_config.get("gamma", 2.0)),
    ).to(device)
    class_balanced_bce = ClassBalancedBCELoss(
        class_counts=class_counts,
        beta=float(loss_config.get("beta", 0.999)),
    ).to(device)
    focal_unweighted = FocalLoss(gamma=float(loss_config.get("gamma", 2.0))).to(device)
    loss_mode, lambda_tcar, lambda_tcl = _loss_switches(config, variant)
    if not np.isfinite(lambda_tcar) or not np.isfinite(lambda_tcl) or min(lambda_tcar, lambda_tcl) < 0:
        raise ValueError("Auxiliary-loss weights must be finite and non-negative.")
    if lambda_tcar > 0 and dataset.name != "taiwan":
        raise ValueError("TCAR is only configured for the Taiwan benchmark.")
    if lambda_tcl > 0 and dataset.name != "american":
        raise ValueError("TCL requires firm-year metadata from the American benchmark.")
    if variant in {"permuted_prior", "uniform_prior"} and dataset.name != "taiwan":
        raise ValueError("TCAR prior controls are only defined for Taiwan.")
    if variant in {"permuted_prior", "uniform_prior"} and lambda_tcar <= 0:
        raise ValueError("TCAR prior controls require a positive TCAR loss weight.")
    if variant == "equal_intragroup" and dataset.name != "american":
        raise ValueError("The fixed intra-group aggregation control is only defined for American.")

    canonical_prior: torch.Tensor | None = None
    training_prior: torch.Tensor | None = None
    if dataset.name == "taiwan":
        domain_prior = loss_config.get("domain_prior", [0.35, 0.30, 0.20, 0.15])
        canonical_prior = combined_prior(
            dataset.train.x,
            y_train,
            dataset.hierarchy,
            domain_prior=domain_prior,
            rho=float(loss_config.get("rho", 0.5)),
        ).to(device)
        training_prior = canonical_prior
        if variant == "uniform_prior":
            uniform_domain_prior = loss_config.get(
                "uniform_domain_prior",
                [1.0 / len(dataset.hierarchy.groups)] * len(dataset.hierarchy.groups),
            )
            training_prior = combined_prior(
                dataset.train.x,
                y_train,
                dataset.hierarchy,
                domain_prior=uniform_domain_prior,
                rho=float(loss_config.get("rho", 0.5)),
            ).to(device)
        elif variant == "permuted_prior":
            training_prior = permute_prior(
                canonical_prior,
                loss_config.get("prior_permutation", [1, 3, 0, 2]),
            )
    if lambda_tcar > 0:
        if canonical_prior is None:
            raise ValueError("TCAR requires a financial prior for the selected dataset.")
        if variant in {"uniform_prior", "permuted_prior"} and training_prior is None:
            raise ValueError("TCAR prior controls require a training prior.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-5)),
    )

    x_train_cpu = torch.as_tensor(dataset.train.x, dtype=torch.float32)
    y_train_cpu = torch.from_numpy(dataset.train.y.astype(np.float32))
    local_indices = torch.arange(len(dataset.train.y), dtype=torch.long)
    train_dataset = TensorDataset(x_train_cpu, y_train_cpu, local_indices)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(training_config.get("batch_size", 128)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    temporal_bank: TemporalPairBank | None = None
    if lambda_tcl > 0:
        anchor_gap = loss_config.get("anchor_gap", 1)
        if isinstance(anchor_gap, bool) or int(anchor_gap) != anchor_gap or anchor_gap < 1:
            raise ValueError("anchor_gap must be a positive integer in observed fiscal years.")
        if int(loss_config.get("negatives", 8)) < 1:
            raise ValueError("TCL requires at least one negative sample.")
        if float(loss_config.get("temperature", 0.1)) <= 0:
            raise ValueError("TCL temperature must be positive.")
        temporal_bank = TemporalPairBank(
            dataset.train.metadata,
            y_train,
            anchor_gap=int(anchor_gap),
        )
    temporal_rng = np.random.default_rng(seed + 1701)
    negative_count = int(loss_config.get("negatives", 8))
    temperature = float(loss_config.get("temperature", 0.1))

    max_epochs = int(training_config.get("max_epochs", 100))
    patience = int(training_config.get("patience", 15))
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")
    best_pr_auc = -np.inf
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x_batch, y_batch, batch_indices in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_indices_np = batch_indices.numpy()
            optimizer.zero_grad(set_to_none=True)
            output = model(x_batch)

            if loss_mode == "cb_focal":
                primary_loss = cb_focal(output["logits"], y_batch)
            elif loss_mode == "class_balanced_bce":
                primary_loss = class_balanced_bce(output["logits"], y_batch)
            elif loss_mode == "focal_unweighted":
                primary_loss = focal_unweighted(output["logits"], y_batch)
            else:
                primary_loss = F.binary_cross_entropy_with_logits(output["logits"], y_batch)

            total_loss = primary_loss
            if lambda_tcar > 0 and training_prior is not None:
                total_loss = total_loss + lambda_tcar * tcar_loss(
                    output.get("group_attention"),
                    y_batch,
                    training_prior,
                )

            if lambda_tcl > 0 and temporal_bank is not None:
                sampled = temporal_bank.sample_for_batch(batch_indices_np, negative_count, temporal_rng)
                if sampled is not None:
                    anchor_index, positive_index, negative_index = sampled
                    anchor_x = x_train_cpu[anchor_index].to(device)
                    positive_x = x_train_cpu[positive_index].to(device)
                    negative_x = x_train_cpu[negative_index.reshape(-1)].to(device)
                    anchor_z = model.encode(anchor_x)["z"]
                    positive_z = model.encode(positive_x)["z"]
                    negative_z = model.encode(negative_x)["z"].reshape(len(anchor_index), negative_count, -1)
                    total_loss = total_loss + lambda_tcl * temporal_infonce(
                        anchor_z,
                        positive_z,
                        negative_z,
                        temperature=temperature,
                    )

            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("The training objective became non-finite.")
            total_loss.backward()
            optimizer.step()

        validation_probability, _, _ = predict_probabilities(model, dataset.validation, device)
        validation_pr_auc = average_precision_score(dataset.validation.y, validation_probability)
        if validation_pr_auc > best_pr_auc + 1e-8:
            best_pr_auc = float(validation_pr_auc)
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    validation_probability, _, _ = predict_probabilities(model, dataset.validation, device)
    threshold, _ = select_f1_threshold(dataset.validation.y, validation_probability)
    validation_metrics = classification_metrics(dataset.validation.y, validation_probability, threshold)
    test_probability: np.ndarray | None = None
    test_attention: np.ndarray | None = None
    subgroup_attention: tuple[np.ndarray, ...] | None = None
    group_attention_mean: np.ndarray | None = None
    alignment: dict[str, float] | None = None
    test_metrics = {name: float("nan") for name in validation_metrics}
    if evaluate_test:
        test_probability, test_attention, subgroup_attention = predict_probabilities(model, dataset.test, device)
        test_metrics = classification_metrics(dataset.test.y, test_probability, threshold)
        if test_attention is not None and np.any(dataset.test.y == 1):
            positive_attention = test_attention[dataset.test.y == 1]
            group_attention_mean = positive_attention.mean(axis=0)
            if canonical_prior is not None:
                reference = canonical_prior.detach().cpu().numpy().astype(float)
                reference = np.clip(reference, 1e-8, 1.0)
                attention = np.clip(positive_attention.astype(float), 1e-8, 1.0)
                kl = np.sum(reference[None, :] * (np.log(reference[None, :]) - np.log(attention)), axis=1)
                cosine = np.sum(attention * reference[None, :], axis=1) / (
                    np.linalg.norm(attention, axis=1) * np.linalg.norm(reference) + 1e-12
                )
                alignment = {"kl": float(np.mean(kl)), "cosine": float(np.mean(cosine))}

    return RunResult(
        seed=seed,
        variant=variant,
        validation=validation_metrics,
        test=test_metrics,
        epochs=best_epoch,
        parameters=count_trainable_parameters(model),
        threshold=threshold,
        group_attention=group_attention_mean,
        test_attention=test_attention,
        subgroup_attention=subgroup_attention,
        validation_probability=validation_probability,
        test_probability=test_probability,
        prior=None if training_prior is None else training_prior.detach().cpu().numpy(),
        alignment=alignment,
        model_state={name: value.detach().cpu() for name, value in best_state.items()},
    )
