from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .data import PreparedDataset
from .metrics import classification_metrics, select_f1_threshold
from .utils import resolve_device, set_seed


@dataclass
class BaselineResult:
    """Metrics and threshold from one comparison-model run."""

    name: str
    seed: int
    validation: dict[str, float]
    test: dict[str, float]
    threshold: float
    parameters: dict[str, Any]


BASELINE_NAMES = (
    "logistic",
    "altman",
    "xgboost",
    "lightgbm",
    "catboost",
    "smote_xgboost",
    "tabnet",
    "tabtransformer",
    "ft_transformer",
    "trompt",
    "cs_stacking",
    "two_stage_lightgbm",
)


def _tree_model(name: str, seed: int, parameters: dict[str, Any]):
    params = dict(parameters)
    if name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
            **params,
        )
    if name == "catboost":
        return CatBoostClassifier(
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    raise KeyError(name)


def _evaluate_probabilities(
    name: str,
    seed: int,
    dataset: PreparedDataset,
    validation_probability: np.ndarray,
    test_probability: np.ndarray | None,
    parameters: dict[str, Any],
) -> BaselineResult:
    threshold, _ = select_f1_threshold(dataset.validation.y, validation_probability)
    validation = classification_metrics(dataset.validation.y, validation_probability, threshold)
    if test_probability is None:
        test = {key: float("nan") for key in validation.keys()}
    else:
        test = classification_metrics(dataset.test.y, test_probability, threshold)
    return BaselineResult(
        name=name,
        seed=seed,
        validation=validation,
        test=test,
        threshold=threshold,
        parameters=parameters,
    )


def _fit_logistic(dataset: PreparedDataset, seed: int, evaluate_test: bool) -> BaselineResult:
    model = LogisticRegression(max_iter=4000, random_state=seed)
    model.fit(dataset.train.x, dataset.train.y)
    validation_probability = model.predict_proba(dataset.validation.x)[:, 1]
    test_probability = model.predict_proba(dataset.test.x)[:, 1] if evaluate_test else None
    return _evaluate_probabilities("logistic", seed, dataset, validation_probability, test_probability, {})


def _fit_tree(
    name: str,
    dataset: PreparedDataset,
    seed: int,
    parameters: dict[str, Any],
    evaluate_test: bool,
) -> BaselineResult:
    model = _tree_model(name, seed, parameters)
    model.fit(dataset.train.x, dataset.train.y)
    validation_probability = model.predict_proba(dataset.validation.x)[:, 1]
    test_probability = model.predict_proba(dataset.test.x)[:, 1] if evaluate_test else None
    return _evaluate_probabilities(name, seed, dataset, validation_probability, test_probability, parameters)


def _fit_smote_xgboost(
    dataset: PreparedDataset,
    seed: int,
    parameters: dict[str, Any],
    evaluate_test: bool,
) -> BaselineResult:
    sampler = SMOTE(random_state=seed)
    x_resampled, y_resampled = sampler.fit_resample(dataset.train.x, dataset.train.y)
    model = _tree_model("xgboost", seed, parameters)
    model.fit(x_resampled, y_resampled)
    validation_probability = model.predict_proba(dataset.validation.x)[:, 1]
    test_probability = model.predict_proba(dataset.test.x)[:, 1] if evaluate_test else None
    return _evaluate_probabilities(
        "smote_xgboost",
        seed,
        dataset,
        validation_probability,
        test_probability,
        parameters,
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=float)
    tiny = np.where(denominator < 0.0, -1e-12, 1e-12)
    adjusted = np.where(np.abs(denominator) < 1e-12, tiny, denominator)
    return np.asarray(numerator, dtype=float) / adjusted


def altman_risk_score(frame) -> np.ndarray:
    """Compute -Z from the original public-firm Altman formulation using American X1-X18 fields."""
    working_capital = frame["X1"].to_numpy(dtype=float) - frame["X14"].to_numpy(dtype=float)
    retained_earnings = frame["X15"].to_numpy(dtype=float)
    ebit = frame["X12"].to_numpy(dtype=float)
    market_value = frame["X8"].to_numpy(dtype=float)
    sales = frame["X9"].to_numpy(dtype=float)
    total_assets = frame["X10"].to_numpy(dtype=float)
    total_liabilities = frame["X17"].to_numpy(dtype=float)
    z_score = (
        1.2 * _safe_ratio(working_capital, total_assets)
        + 1.4 * _safe_ratio(retained_earnings, total_assets)
        + 3.3 * _safe_ratio(ebit, total_assets)
        + 0.6 * _safe_ratio(market_value, total_liabilities)
        + 1.0 * _safe_ratio(sales, total_assets)
    )
    return -z_score


def _fit_altman(dataset: PreparedDataset, seed: int, evaluate_test: bool) -> BaselineResult:
    if dataset.name != "american":
        raise ValueError("The exact Altman Z-Score baseline is defined for the American benchmark in this package.")
    validation_risk = altman_risk_score(dataset.raw_validation_frame)
    test_risk = altman_risk_score(dataset.raw_test_frame) if evaluate_test else None
    threshold = -2.675
    validation = classification_metrics(dataset.validation.y, validation_risk, threshold)
    if test_risk is None:
        test = {key: float("nan") for key in validation.keys()}
    else:
        test = classification_metrics(dataset.test.y, test_risk, threshold)
    return BaselineResult(
        name="altman",
        seed=seed,
        validation=validation,
        test=test,
        threshold=threshold,
        parameters={"z_cutoff": 2.675},
    )


def _fit_two_stage_lightgbm(
    dataset: PreparedDataset,
    seed: int,
    parameters: dict[str, Any],
    evaluate_test: bool,
) -> BaselineResult:
    """Two-stage LightGBM: full-feature ranking followed by validation-selected feature refitting."""
    stage_one = _tree_model("lightgbm", seed, parameters)
    stage_one.fit(dataset.train.x, dataset.train.y)
    ranking = np.argsort(stage_one.feature_importances_)[::-1]
    dimension = dataset.train.x.shape[1]
    feature_counts = sorted(set(min(dimension, value) for value in (5, 8, 12, 16, dimension)))

    best_pr_auc = -np.inf
    best_model = None
    best_features = None
    for feature_count in feature_counts:
        selected = ranking[:feature_count]
        model = _tree_model("lightgbm", seed, parameters)
        model.fit(dataset.train.x[:, selected], dataset.train.y)
        validation_probability = model.predict_proba(dataset.validation.x[:, selected])[:, 1]
        score = average_precision_score(dataset.validation.y, validation_probability)
        if score > best_pr_auc + 1e-12:
            best_pr_auc = float(score)
            best_model = model
            best_features = selected

    assert best_model is not None and best_features is not None
    validation_probability = best_model.predict_proba(dataset.validation.x[:, best_features])[:, 1]
    test_probability = (
        best_model.predict_proba(dataset.test.x[:, best_features])[:, 1]
        if evaluate_test
        else None
    )
    reported_parameters = {**parameters, "selected_features": int(len(best_features))}
    return _evaluate_probabilities(
        "two_stage_lightgbm",
        seed,
        dataset,
        validation_probability,
        test_probability,
        reported_parameters,
    )


def _fit_cs_stacking(dataset: PreparedDataset, seed: int, evaluate_test: bool) -> BaselineResult:
    """Cost-sensitive stacking with out-of-fold tree predictions and a balanced logistic meta-learner."""
    negatives = max(int(np.sum(dataset.train.y == 0)), 1)
    positives = max(int(np.sum(dataset.train.y == 1)), 1)
    scale_positive = negatives / positives

    def base_models(fold_seed: int):
        return [
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_positive,
                eval_metric="logloss",
                random_state=fold_seed,
                n_jobs=-1,
            ),
            LGBMClassifier(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.04,
                class_weight="balanced",
                random_state=fold_seed,
                verbosity=-1,
                n_jobs=-1,
            ),
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.04,
                auto_class_weights="Balanced",
                random_seed=fold_seed,
                verbose=False,
                allow_writing_files=False,
            ),
        ]

    x_train = dataset.train.x
    y_train = dataset.train.y
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(y_train), 3), dtype=float)

    for fold_index, (fit_index, holdout_index) in enumerate(splitter.split(x_train, y_train)):
        for model_index, model in enumerate(base_models(seed + fold_index + 1)):
            model.fit(x_train[fit_index], y_train[fit_index])
            oof[holdout_index, model_index] = model.predict_proba(x_train[holdout_index])[:, 1]

    full_models = base_models(seed)
    for model in full_models:
        model.fit(x_train, y_train)

    meta_learner = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    meta_learner.fit(oof, y_train)

    validation_stack = np.column_stack(
        [model.predict_proba(dataset.validation.x)[:, 1] for model in full_models]
    )
    validation_probability = meta_learner.predict_proba(validation_stack)[:, 1]

    test_probability = None
    if evaluate_test:
        test_stack = np.column_stack([model.predict_proba(dataset.test.x)[:, 1] for model in full_models])
        test_probability = meta_learner.predict_proba(test_stack)[:, 1]

    return _evaluate_probabilities(
        "cs_stacking",
        seed,
        dataset,
        validation_probability,
        test_probability,
        {"folds": 5, "meta_learner": "balanced_logistic"},
    )


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax transformation used for TabNet-style feature masks."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits, _ = torch.sort(shifted, descending=True, dim=dim)
    cumulative = sorted_logits.cumsum(dim) - 1
    range_shape = [1] * logits.dim()
    range_shape[dim] = logits.size(dim)
    ranks = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype).view(range_shape)
    support = ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = cumulative.gather(dim, support_size.long() - 1) / support_size.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


class TabNetNumeric(torch.nn.Module):
    """Compact TabNet-style network for continuous financial features."""

    def __init__(self, n_features: int, width: int, dropout: float, steps: int = 3) -> None:
        super().__init__()
        self.steps = int(steps)
        self.initial = torch.nn.Sequential(torch.nn.Linear(n_features, width), torch.nn.GELU())
        self.attention = torch.nn.ModuleList(torch.nn.Linear(width, n_features) for _ in range(self.steps))
        self.transform = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(n_features, width),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(width, width),
                torch.nn.GELU(),
            )
            for _ in range(self.steps)
        )
        self.head = torch.nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.initial(x)
        accumulated = torch.zeros_like(context)
        for attention_layer, transform_layer in zip(self.attention, self.transform):
            mask = _sparsemax(attention_layer(context), dim=1)
            decision = transform_layer(x * mask)
            accumulated = accumulated + decision
            context = decision
        return self.head(accumulated / self.steps).squeeze(-1)


class TabTransformerNumeric(torch.nn.Module):
    """Contextual feature-token Transformer for continuous tabular inputs."""

    def __init__(self, n_features: int, width: int, dropout: float, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.value_weight = torch.nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.value_bias = torch.nn.Parameter(torch.zeros(n_features, width))
        self.feature_embedding = torch.nn.Parameter(torch.randn(n_features, width) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=layers)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.value_weight.unsqueeze(0) + self.value_bias.unsqueeze(0)
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class FTTransformerNumeric(torch.nn.Module):
    """Feature-tokenizer Transformer with a learnable classification token."""

    def __init__(self, n_features: int, width: int, dropout: float, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.bias = torch.nn.Parameter(torch.zeros(n_features, width))
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, width))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=layers)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(-1)


class TromptNumeric(torch.nn.Module):
    """Prompt-conditioned tabular network with repeated prompt-to-feature attention."""

    def __init__(
        self,
        n_features: int,
        width: int,
        dropout: float,
        prompt_count: int = 4,
        cycles: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.bias = torch.nn.Parameter(torch.zeros(n_features, width))
        self.feature_embedding = torch.nn.Parameter(torch.randn(n_features, width) * 0.02)
        self.prompts = torch.nn.Parameter(torch.randn(1, prompt_count, width) * 0.02)
        self.cross_attention = torch.nn.ModuleList(
            torch.nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            for _ in range(cycles)
        )
        self.norms = torch.nn.ModuleList(
            torch.nn.ModuleList([torch.nn.LayerNorm(width), torch.nn.LayerNorm(width)])
            for _ in range(cycles)
        )
        self.feed_forward = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(width, width * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(width * 2, width),
            )
            for _ in range(cycles)
        )
        self.head = torch.nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        features = features + self.feature_embedding.unsqueeze(0)
        prompts = self.prompts.expand(x.shape[0], -1, -1)
        for attention, norms, feed_forward in zip(self.cross_attention, self.norms, self.feed_forward):
            attended, _ = attention(prompts, features, features, need_weights=False)
            prompts = norms[0](prompts + attended)
            prompts = norms[1](prompts + feed_forward(prompts))
        return self.head(prompts.mean(dim=1)).squeeze(-1)


def _neural_factory(name: str, n_features: int, width: int, dropout: float) -> Callable[[], torch.nn.Module]:
    if name == "tabnet":
        return lambda: TabNetNumeric(n_features, width, dropout)
    if name == "tabtransformer":
        return lambda: TabTransformerNumeric(n_features, width, dropout)
    if name == "ft_transformer":
        return lambda: FTTransformerNumeric(n_features, width, dropout)
    if name == "trompt":
        return lambda: TromptNumeric(n_features, width, dropout)
    raise KeyError(name)


def _fit_neural(
    name: str,
    dataset: PreparedDataset,
    seed: int,
    parameters: dict[str, Any],
    evaluate_test: bool,
) -> BaselineResult:
    set_seed(seed)
    device = resolve_device("auto")
    width = int(parameters.get("width", 64))
    dropout = float(parameters.get("dropout", 0.1))
    learning_rate = float(parameters.get("learning_rate", 1e-3))
    max_epochs = int(parameters.get("max_epochs", 100))
    patience = int(parameters.get("patience", 15))
    batch_size = int(parameters.get("batch_size", 128))

    model = _neural_factory(name, dataset.train.x.shape[1], width, dropout)().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    x_train = torch.from_numpy(dataset.train.x)
    y_train = torch.from_numpy(dataset.train.y.astype(np.float32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_pr_auc = -np.inf
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0
    validation_x = torch.from_numpy(dataset.validation.x).to(device)

    for _ in range(max_epochs):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_probability = torch.sigmoid(model(validation_x)).cpu().numpy()
        validation_pr_auc = average_precision_score(dataset.validation.y, validation_probability)
        if validation_pr_auc > best_pr_auc + 1e-8:
            best_pr_auc = float(validation_pr_auc)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_probability = torch.sigmoid(model(validation_x)).cpu().numpy()
        test_probability = (
            torch.sigmoid(model(torch.from_numpy(dataset.test.x).to(device))).cpu().numpy()
            if evaluate_test
            else None
        )
    return _evaluate_probabilities(name, seed, dataset, validation_probability, test_probability, parameters)


def candidate_parameters(name: str, search_config: dict[str, Any], dataset: PreparedDataset) -> list[dict[str, Any]]:
    """Return the validation search candidates for one comparison method."""
    if name in {"logistic", "altman", "cs_stacking"}:
        return [{}]
    if name in {"xgboost", "lightgbm", "catboost"}:
        return [dict(item) for item in search_config["classical"][name]]
    if name == "smote_xgboost":
        return [dict(item) for item in search_config["classical"]["xgboost"]]
    if name == "two_stage_lightgbm":
        return [dict(item) for item in search_config["classical"]["lightgbm"]]
    if name in {"tabnet", "tabtransformer", "ft_transformer", "trompt"}:
        neural = search_config["neural"]
        batch_size = 128 if dataset.name == "taiwan" else 256
        return [
            {
                "learning_rate": float(learning_rate),
                "width": int(width),
                "dropout": float(dropout),
                "max_epochs": int(neural.get("max_epochs", 100)),
                "patience": int(neural.get("patience", 15)),
                "batch_size": batch_size,
            }
            for learning_rate, width, dropout in product(
                neural["learning_rate"],
                neural["width"],
                neural["dropout"],
            )
        ]
    raise KeyError(name)


def fit_baseline(
    name: str,
    dataset: PreparedDataset,
    seed: int,
    parameters: dict[str, Any] | None = None,
    evaluate_test: bool = True,
) -> BaselineResult:
    """Fit one comparison method under the common data and threshold protocol."""
    if name not in BASELINE_NAMES:
        raise ValueError(f"Unknown comparison method: {name}")
    params = dict(parameters or {})
    if name == "logistic":
        return _fit_logistic(dataset, seed, evaluate_test)
    if name == "altman":
        return _fit_altman(dataset, seed, evaluate_test)
    if name in {"xgboost", "lightgbm", "catboost"}:
        return _fit_tree(name, dataset, seed, params, evaluate_test)
    if name == "smote_xgboost":
        return _fit_smote_xgboost(dataset, seed, params, evaluate_test)
    if name == "two_stage_lightgbm":
        return _fit_two_stage_lightgbm(dataset, seed, params, evaluate_test)
    if name == "cs_stacking":
        return _fit_cs_stacking(dataset, seed, evaluate_test)
    return _fit_neural(name, dataset, seed, params, evaluate_test)


def select_parameters(
    name: str,
    dataset: PreparedDataset,
    seed: int,
    search_config: dict[str, Any],
) -> dict[str, Any]:
    """Choose one parameter setting using validation PR-AUC only."""
    candidates = candidate_parameters(name, search_config, dataset)
    if len(candidates) == 1:
        return candidates[0]

    best_parameters = candidates[0]
    best_score = -np.inf
    for parameters in candidates:
        result = fit_baseline(name, dataset, seed, parameters, evaluate_test=False)
        score = float(result.validation["pr_auc"])
        if score > best_score + 1e-12:
            best_score = score
            best_parameters = parameters
    return dict(best_parameters)


def applicable_baselines(dataset_name: str) -> tuple[str, ...]:
    """Return the comparison methods applicable to a benchmark."""
    if dataset_name.lower() == "taiwan":
        return tuple(name for name in BASELINE_NAMES if name != "altman")
    return BASELINE_NAMES
