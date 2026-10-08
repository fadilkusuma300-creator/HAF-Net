from __future__ import annotations

from copy import deepcopy
import os
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, accuracy_score, recall_score, balanced_accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

from .data import PreparedDataset
from .metrics import classification_metrics, select_f1_threshold
from .preprocess import TrainOnlyPreprocessor
from .utils import resolve_device, set_seed
from .validation import binary_labels
from .published_models import TabNet, TabTransformerNumericAdaptation, FTTransformer, Trompt


@dataclass
class BaselineResult:
    """Metrics and threshold from one comparison-model run."""

    name: str
    seed: int
    validation: dict[str, float]
    test: dict[str, float]
    threshold: float
    parameters: dict[str, Any]
    validation_probability: np.ndarray
    test_probability: np.ndarray | None


BASELINE_NAMES = (
    "logistic",
    "altman",
    "xgboost",
    "lightgbm",
    "catboost",
    "smote_xgboost",
    "tabnet",
    "tabtransformer_numeric_adapted",
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
            n_jobs=int(os.environ.get("HAFNET_N_JOBS", "1")),
            **params,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            random_state=seed,
            verbosity=-1,
            n_jobs=int(os.environ.get("HAFNET_N_JOBS", "1")),
            subsample_freq=1,
            **params,
        )
    if name == "catboost":
        return CatBoostClassifier(
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=int(os.environ.get("HAFNET_N_JOBS", "1")),
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
        validation_probability=np.asarray(validation_probability, dtype=float),
        test_probability=None if test_probability is None else np.asarray(test_probability, dtype=float),
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
    x_resampled, y_resampled = _resample_training_data(dataset.train.x, dataset.train.y, "smote", seed)
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
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all() or np.any(denominator == 0):
        raise ValueError("Altman ratios require finite inputs and nonzero denominators.")
    return numerator / denominator


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
        raise ValueError("The Altman Z-Score formula is defined for the American benchmark in this package.")
    validation_risk = altman_risk_score(dataset.raw_validation_frame)
    test_risk = altman_risk_score(dataset.raw_test_frame) if evaluate_test else None
    threshold = float(np.nextafter(-2.675, np.inf))  # risk >= this is exactly Z < 2.675
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
        validation_probability=np.asarray(validation_risk, dtype=float),
        test_probability=None if test_risk is None else np.asarray(test_risk, dtype=float),
    )


def _resample_training_data(x: np.ndarray, y: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = binary_labels(y, require_both=True)
    if method == "none":
        return x, y
    minority = int(np.bincount(y, minlength=2).min())
    if method in {"smote", "smote_enn"} and minority < 2:
        raise ValueError("SMOTE requires at least two minority observations in the fitting fold.")
    if method == "smote":
        sampler = SMOTE(random_state=seed, k_neighbors=min(5, minority - 1))
    elif method == "smote_enn":
        sampler = SMOTEENN(random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=min(5, minority - 1)))
    elif method == "kmeans_centroids":
        sampler = ClusterCentroids(random_state=seed)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    x_fit, y_fit = sampler.fit_resample(x, y)
    binary_labels(y_fit, require_both=True)
    return x_fit, y_fit


def inner_splits(x: np.ndarray, y: np.ndarray, folds: int, seed: int, groups=None):
    """Keep firms intact in American inner folds; stratify rows for Taiwan."""
    if folds < 2:
        raise ValueError("At least two inner folds are required.")
    if groups is None:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(x, y))
    else:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("Group identifiers and targets must have equal lengths.")
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(x, y, groups))
    for fit, held in splits:
        binary_labels(y[fit], require_both=True)
        binary_labels(y[held], require_both=True)
        if groups is not None and set(groups[fit]).intersection(groups[held]):
            raise ValueError("An inner fold contains a company in both subsets.")
    return splits


class _ResampledLightGBMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, parameters: dict[str, Any], resampling: str, seed: int) -> None:
        self.parameters = parameters
        self.resampling = resampling
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.preprocessor_ = TrainOnlyPreprocessor().fit(x)
        x = self.preprocessor_.transform(x)
        x_fit, y_fit = _resample_training_data(x, np.asarray(y), self.resampling, int(self.seed))
        self.model_ = _tree_model("lightgbm", int(self.seed), dict(self.parameters))
        self.model_.fit(x_fit, y_fit)
        self.classes_ = self.model_.classes_
        self.feature_importances_ = self.model_.feature_importances_
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict(self.preprocessor_.transform(np.asarray(x, dtype=np.float64)))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(self.preprocessor_.transform(np.asarray(x, dtype=np.float64)))


def _fit_two_stage_lightgbm(
    dataset: PreparedDataset,
    seed: int,
    parameters: dict[str, Any],
    evaluate_test: bool,
) -> BaselineResult:
    resampling = str(parameters.get("resampling", "smote_enn"))
    excluded = {"resampling", "cv_folds", "importance_tail_mass", "wrapper_min_features",
        "accuracy_min_improvement", "wrapper_patience"}
    lightgbm_parameters = {key: value for key, value in parameters.items() if key not in excluded}
    feature_columns = list(dataset.feature_ids)
    x_train = dataset.raw_train_frame[feature_columns].to_numpy(dtype=np.float64)
    x_validation = dataset.raw_validation_frame[feature_columns].to_numpy(dtype=np.float64)
    x_test = dataset.raw_test_frame[feature_columns].to_numpy(dtype=np.float64) if evaluate_test else None
    y = dataset.train.y
    groups = dataset.train.metadata["company_name"].to_numpy() if dataset.name == "american" else None
    folds = inner_splits(x_train, y, int(parameters.get("cv_folds", 3)), seed, groups)
    tail_mass = float(parameters.get("importance_tail_mass", 0.01))
    min_features = int(parameters.get("wrapper_min_features", 2))
    tolerance = float(parameters.get("accuracy_min_improvement", 0.005))
    patience = int(parameters.get("wrapper_patience", 3))
    if not 0 < tail_mass < 1 or not 1 <= min_features <= x_train.shape[1] or tolerance < 0 or patience < 1:
        raise ValueError("Invalid wrapper elimination or stopping settings.")

    selected = np.arange(x_train.shape[1])
    best_features = selected.copy()
    best_key = (-np.inf, -np.inf, -np.inf)
    best_accuracy = -np.inf
    stale = 0
    trace = []
    stage1_model = None
    stage1_ap = None
    while True:
        probabilities = np.empty(len(y), dtype=float)
        importance = np.zeros(len(selected), dtype=float)
        for fold_id, (fit, held) in enumerate(folds):
            model = _ResampledLightGBMClassifier(lightgbm_parameters, resampling, seed + fold_id)
            model.fit(x_train[fit][:, selected], y[fit])
            probabilities[held] = model.predict_proba(x_train[held][:, selected])[:, 1]
            raw_importance = model.feature_importances_.astype(float)
            importance += raw_importance / max(float(raw_importance.sum()), 1.0)
        prediction = (probabilities >= 0.5).astype(int)
        accuracy = float(accuracy_score(y, prediction))
        recall = float(recall_score(y, prediction, zero_division=0))
        tp = int(np.sum((y == 1) & (prediction == 1)))
        trace.append({"feature_count": len(selected), "features": [feature_columns[i] for i in selected],
            "inner_true_positives": tp, "inner_recall": recall, "inner_accuracy": accuracy,
            "inner_f1": float(f1_score(y, prediction, zero_division=0)),
            "inner_kappa": float(cohen_kappa_score(y, prediction))})
        # Recall selects the subset; fewer features break a recall tie, then accuracy.
        key = (recall, -len(selected), accuracy)
        if key > best_key:
            best_key, best_features = key, selected.copy()
        if stage1_model is None:
            stage1_model = _ResampledLightGBMClassifier(lightgbm_parameters, resampling, seed)
            stage1_model.fit(x_train, y)
            stage1_ap = float(average_precision_score(dataset.validation.y, stage1_model.predict_proba(x_validation)[:, 1]))
        if accuracy - best_accuracy >= tolerance:
            stale, best_accuracy = 0, accuracy
        else:
            stale += 1
            best_accuracy = max(best_accuracy, accuracy)
        if len(selected) <= min_features or stale >= patience:
            break
        importance /= max(importance.sum(), 1.0)
        order = np.lexsort((selected, importance))
        remove = max(1, int(np.sum(np.cumsum(importance[order]) <= tail_mass)))
        remove = min(remove, len(selected) - min_features)
        selected = np.sort(np.delete(selected, order[:remove]))

    selected = best_features
    final_model = _ResampledLightGBMClassifier(lightgbm_parameters, resampling, seed)
    final_model.fit(x_train[:, selected], y)
    validation_probability = final_model.predict_proba(x_validation[:, selected])[:, 1]
    test_probability = final_model.predict_proba(x_test[:, selected])[:, 1] if x_test is not None else None
    return _evaluate_probabilities("two_stage_lightgbm", seed, dataset,
        validation_probability, test_probability,
        {**parameters, "stage1_validation_average_precision": stage1_ap,
         "stage1_full_feature_count": int(x_train.shape[1]), "wrapper_trace": trace,
         "selected_features": [feature_columns[i] for i in selected],
         "selected_feature_count": len(selected), "wrapper_best_cv_score": float(best_key[0]),
         "inner_split_unit": "company" if groups is not None else "row"})


class _TrainingXGBoost(ClassifierMixin, BaseEstimator):
    """Fit preprocessing and class costs on the fitting subset of each wrapper fold."""
    def __init__(self, seed: int = 42, trees: int = 50, depth: int = 3):
        self.seed, self.trees, self.depth = seed, trees, depth

    def fit(self, x, y):
        y = binary_labels(y, require_both=True)
        self.preprocessor_ = TrainOnlyPreprocessor().fit(x)
        self.model_ = XGBClassifier(n_estimators=self.trees, max_depth=self.depth,
            learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=float(np.sum(y == 0) / np.sum(y == 1)), eval_metric="logloss",
            random_state=self.seed, n_jobs=int(os.environ.get("HAFNET_N_JOBS", "1")))
        self.model_.fit(self.preprocessor_.transform(x), y)
        self.classes_ = self.model_.classes_
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, x):
        return self.model_.predict(self.preprocessor_.transform(x))


def _select_cs_features(x: np.ndarray, y: np.ndarray, seed: int, parameters: dict[str, Any], groups=None) -> np.ndarray:
    """XGBoost ranking plus forward search; choose the BACC-best visited subset."""
    ranker = _TrainingXGBoost(seed, trees=100, depth=4).fit(x, y)
    prefilter_count = min(int(parameters.get("prefilter_features", 12)), x.shape[1])
    cap = min(int(parameters.get("feature_count", 6)), prefilter_count)
    if min(prefilter_count, cap) < 1:
        raise ValueError("Feature screening sizes must be positive.")
    candidates = list(np.lexsort((np.arange(x.shape[1]), -ranker.feature_importances_))[:prefilter_count])
    folds = inner_splits(x, y, int(parameters.get("feature_selection_folds", 2)), seed, groups)
    selected = []
    best_subset = []
    best_score = -np.inf
    for _ in range(cap):
        choice, score = None, -np.inf
        for feature in sorted(candidates):
            indices = selected + [feature]
            fold_scores = []
            for fold_index, (fit, held) in enumerate(folds):
                model = _TrainingXGBoost(seed + fold_index).fit(x[fit][:, indices], y[fit])
                fold_scores.append(balanced_accuracy_score(y[held], model.predict(x[held][:, indices])))
            current = float(np.mean(fold_scores))
            if current > score:
                choice, score = feature, current
        selected.append(choice)
        candidates.remove(choice)
        if score > best_score:
            best_score, best_subset = score, selected.copy()
    return np.asarray(best_subset, dtype=int)


def _fit_cs_stacking(
    dataset: PreparedDataset,
    seed: int,
    parameters: dict[str, Any],
    evaluate_test: bool,
) -> BaselineResult:
    """Select features within each stack fold, then fit cost-sensitive out-of-fold stacking."""
    y_train = dataset.train.y
    feature_columns = list(dataset.feature_ids)
    x_train = dataset.raw_train_frame[feature_columns].to_numpy(dtype=np.float64)
    x_validation_raw = dataset.raw_validation_frame[feature_columns].to_numpy(dtype=np.float64)
    x_test_raw = (
        dataset.raw_test_frame[feature_columns].to_numpy(dtype=np.float64)
        if evaluate_test
        else None
    )

    def base_models(fold_seed: int, positive_weight: float):
        return [
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=positive_weight,
                eval_metric="logloss",
                random_state=fold_seed,
                n_jobs=int(os.environ.get("HAFNET_N_JOBS", "1")),
            ),
            LGBMClassifier(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.04,
                class_weight="balanced",
                random_state=fold_seed,
                verbosity=-1,
                n_jobs=int(os.environ.get("HAFNET_N_JOBS", "1")),
                subsample_freq=1,
            ),
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.04,
                auto_class_weights="Balanced",
                random_seed=fold_seed,
                verbose=False,
                allow_writing_files=False,
            thread_count=int(os.environ.get("HAFNET_N_JOBS", "1")),
            ),
        ]

    stacking_folds = int(parameters.get("stacking_folds", 3))
    groups = dataset.train.metadata["company_name"].to_numpy() if dataset.name == "american" else None
    splits = inner_splits(x_train, y_train, stacking_folds, seed, groups)
    oof = np.zeros((len(y_train), 3), dtype=float)
    fold_selected_features: list[list[str]] = []

    for fold_index, (fit_index, holdout_index) in enumerate(splits):
        fold_seed = seed + fold_index + 1
        fold_preprocessor = TrainOnlyPreprocessor().fit(x_train[fit_index])
        fold_x_train = fold_preprocessor.transform(x_train[fit_index])
        fold_x_holdout = fold_preprocessor.transform(x_train[holdout_index])
        selected = _select_cs_features(x_train[fit_index], y_train[fit_index], fold_seed, parameters, None if groups is None else groups[fit_index])
        fold_selected_features.append([dataset.feature_ids[index] for index in selected])
        fold_negatives = max(int(np.sum(y_train[fit_index] == 0)), 1)
        fold_positives = max(int(np.sum(y_train[fit_index] == 1)), 1)
        fold_positive_weight = fold_negatives / fold_positives
        fold_x = fold_x_train[:, selected]
        holdout_x = fold_x_holdout[:, selected]
        for model_index, model in enumerate(base_models(fold_seed, fold_positive_weight)):
            model.fit(fold_x, y_train[fit_index])
            oof[holdout_index, model_index] = model.predict_proba(holdout_x)[:, 1]

    full_preprocessor = TrainOnlyPreprocessor().fit(x_train)
    processed_x_train = full_preprocessor.transform(x_train)
    processed_x_validation = full_preprocessor.transform(x_validation_raw)
    processed_x_test = full_preprocessor.transform(x_test_raw) if x_test_raw is not None else None
    selected_features = _select_cs_features(x_train, y_train, seed, parameters, groups)
    full_negatives = max(int(np.sum(y_train == 0)), 1)
    full_positives = max(int(np.sum(y_train == 1)), 1)
    full_models = base_models(seed, full_negatives / full_positives)
    for model in full_models:
        model.fit(processed_x_train[:, selected_features], y_train)

    meta_learner = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    meta_learner.fit(oof, y_train)

    validation_stack = np.column_stack(
        [model.predict_proba(processed_x_validation[:, selected_features])[:, 1] for model in full_models]
    )
    validation_probability = meta_learner.predict_proba(validation_stack)[:, 1]

    test_probability = None
    if processed_x_test is not None:
        test_stack = np.column_stack(
            [model.predict_proba(processed_x_test[:, selected_features])[:, 1] for model in full_models]
        )
        test_probability = meta_learner.predict_proba(test_stack)[:, 1]

    return _evaluate_probabilities(
        "cs_stacking",
        seed,
        dataset,
        validation_probability,
        test_probability,
        {
            **parameters,
            "fold_selected_features": fold_selected_features,
            "selected_features": [dataset.feature_ids[index] for index in selected_features],
            "selected_feature_count": int(len(selected_features)),
            "meta_learner": "balanced_logistic",
        },
    )


def _neural_factory(name: str, n_features: int, parameters: dict[str, Any]) -> Callable[[], torch.nn.Module]:
    """Build a deep-tabular comparator using its published/reference architecture.

    TabTransformer remains explicitly labelled as a numerical-input adaptation
    because the two benchmarks contain no categorical columns, whereas the
    published architecture contextualizes categorical embeddings.
    """
    width = int(parameters.get("width", 64))
    dropout = float(parameters.get("dropout", 0.1))
    if name == "tabnet":
        return lambda: TabNet(
            n_features,
            d_hidden=width,
            n_steps=int(parameters.get("n_steps", 4)),
            gamma=float(parameters.get("gamma", 1.5)),
        )
    if name == "tabtransformer_numeric_adapted":
        heads = 8 if width % 8 == 0 else 4
        return lambda: TabTransformerNumericAdaptation(
            n_features,
            d_model=width,
            dropout=dropout,
            layers=int(parameters.get("layers", 4)),
            heads=heads,
        )
    if name == "ft_transformer":
        heads = 8 if width % 8 == 0 else 4
        return lambda: FTTransformer(
            n_features,
            d_model=width,
            dropout=dropout,
            layers=int(parameters.get("layers", 3)),
            heads=heads,
        )
    if name == "trompt":
        return lambda: Trompt(
            n_features,
            channels=width,
            n_prompts=int(parameters.get("num_prompts", width)),
            n_layers=int(parameters.get("layers", 6)),
        )
    raise KeyError(name)


def _model_logits(model: torch.nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
    raw = model(x)
    if name == "trompt":
        return Trompt.inference_logits(raw)
    return raw


def _training_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, name: str) -> torch.Tensor:
    raw = model(x)
    if name == "trompt":
        # Trompt is supervised at every cell, matching the layer-wise objective
        # used by the published architecture.
        targets = y.unsqueeze(1).expand_as(raw)
        return F.binary_cross_entropy_with_logits(raw, targets)
    return F.binary_cross_entropy_with_logits(raw, y)


def _fit_neural(
    name: str,
    dataset: PreparedDataset,
    seed: int,
    parameters: dict[str, Any],
    evaluate_test: bool,
    device_name: str,
) -> BaselineResult:
    set_seed(seed)
    device = resolve_device(device_name)
    width = int(parameters.get("width", 64))
    dropout = float(parameters.get("dropout", 0.1))
    learning_rate = float(parameters.get("learning_rate", 1e-3))
    max_epochs = int(parameters.get("max_epochs", 100))
    patience = int(parameters.get("patience", 15))
    batch_size = int(parameters.get("batch_size", 128))

    model = _neural_factory(name, dataset.train.x.shape[1], parameters)().to(device)
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
            loss = _training_loss(model, x_batch, y_batch, name)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_probability = torch.sigmoid(_model_logits(model, validation_x, name)).cpu().numpy()
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
        validation_probability = torch.sigmoid(_model_logits(model, validation_x, name)).cpu().numpy()
        test_probability = (
            torch.sigmoid(_model_logits(model, torch.from_numpy(dataset.test.x).to(device), name)).cpu().numpy()
            if evaluate_test
            else None
        )
    return _evaluate_probabilities(name, seed, dataset, validation_probability, test_probability, parameters)


def candidate_parameters(name: str, search_config: dict[str, Any], dataset: PreparedDataset) -> list[dict[str, Any]]:
    """Return the validation search candidates for one comparison method."""
    def expand_grid(specification: dict[str, Any]) -> list[dict[str, Any]]:
        keys = list(specification)
        values = [value if isinstance(value, list) else [value] for value in specification.values()]
        return [dict(zip(keys, combination)) for combination in product(*values)]

    if name in {"logistic", "altman"}:
        return [{}]
    if name == "cs_stacking":
        return [
            {
                "prefilter_features": int(prefilter),
                "feature_count": int(count),
                "feature_selection_folds": int(search_config["cs_stacking"]["feature_selection_folds"]),
                "stacking_folds": int(search_config["cs_stacking"]["stacking_folds"]),
            }
            for prefilter, count in product(
                search_config["cs_stacking"]["prefilter_features"],
                search_config["cs_stacking"]["feature_count"],
            )
        ]
    if name in {"xgboost", "lightgbm", "catboost"}:
        return expand_grid(search_config["classical"][name])
    if name == "smote_xgboost":
        return expand_grid(search_config["classical"]["xgboost"])
    if name == "two_stage_lightgbm":
        wrapper = search_config["two_stage_lightgbm"]
        candidates = []
        for item in wrapper["base_settings"]:
            for resampling in wrapper["resampling"]:
                candidates.append(
                    {
                        **dict(item),
                        "resampling": resampling,
                        "cv_folds": int(wrapper["cv_folds"]),
                        "importance_tail_mass": float(wrapper["importance_tail_mass"]),
                        "accuracy_min_improvement": float(wrapper["accuracy_min_improvement"]),
                        "wrapper_patience": int(wrapper["wrapper_patience"]),
                        "wrapper_min_features": int(wrapper["wrapper_min_features"]),
                    }
                )
        return candidates
    if name in {"tabnet", "tabtransformer_numeric_adapted", "ft_transformer", "trompt"}:
        specification = search_config["neural"][name]
        batch_size = 128 if dataset.name == "taiwan" else 256
        candidates = expand_grid({k: v for k, v in specification.items() if k not in {"max_epochs", "patience", "batch_size"}})
        return [
            {
                **candidate,
                "max_epochs": int(specification.get("max_epochs", 100)),
                "patience": int(specification.get("patience", 15)),
                "batch_size": int(specification.get("batch_size", batch_size)),
            }
            for candidate in candidates
        ]
    raise KeyError(name)


def fit_baseline(
    name: str,
    dataset: PreparedDataset,
    seed: int,
    parameters: dict[str, Any] | None = None,
    evaluate_test: bool = True,
    device: str = "auto",
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
        return _fit_cs_stacking(dataset, seed, params, evaluate_test)
    return _fit_neural(name, dataset, seed, params, evaluate_test, device)


def select_parameters(
    name: str,
    dataset: PreparedDataset,
    seeds: Iterable[int],
    search_config: dict[str, Any],
    return_trials: bool = False,
    device: str = "auto",
) -> dict[str, Any] | tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose one parameter setting using validation average precision only."""
    candidates = candidate_parameters(name, search_config, dataset)
    if len(candidates) == 1:
        return (candidates[0], []) if return_trials else candidates[0]

    selected_seeds = [int(seed) for seed in seeds]
    if not selected_seeds:
        raise ValueError("At least one tuning seed is required.")
    best_parameters = candidates[0]
    best_score = -np.inf
    trial_rows: list[dict[str, Any]] = []
    for candidate_id, parameters in enumerate(candidates, start=1):
        scores = []
        for seed in selected_seeds:
            result = fit_baseline(name, dataset, seed, parameters, evaluate_test=False, device=device)
            score_value = float(result.validation["pr_auc"])
            scores.append(score_value)
            trial_rows.append(
                {
                    "model": name,
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "parameters": parameters,
                    "validation_average_precision": score_value,
                    "stage1_validation_average_precision": result.parameters.get(
                        "stage1_validation_average_precision", float("nan")
                    ),
                    "wrapper_best_cv_score": result.parameters.get("wrapper_best_cv_score", float("nan")),
                    "selected_feature_count": result.parameters.get("selected_feature_count", float("nan")),
                }
            )
        score = float(np.mean(scores))
        if score > best_score + 1e-12:
            best_score = score
            best_parameters = parameters
    if return_trials:
        return dict(best_parameters), trial_rows
    return dict(best_parameters)


def applicable_baselines(dataset_name: str) -> tuple[str, ...]:
    """Return the comparison methods applicable to a benchmark."""
    if dataset_name.lower() == "taiwan":
        return tuple(name for name in BASELINE_NAMES if name != "altman")
    return BASELINE_NAMES
