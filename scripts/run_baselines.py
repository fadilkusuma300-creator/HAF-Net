#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparison methods under the common benchmark protocol.")
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml", help="Comparison-model search configuration")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names or 'all'")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output", default="outputs/baselines")
    parser.add_argument("--parameters", default=None, help="JSON file containing selected parameter dictionaries")
    parser.add_argument("--no-search", action="store_true", help="Unsupported: selection requires the validation search or its saved parameter file")
    return parser.parse_args()


def record_from_result(result) -> dict:
    row = {
        "model": result.name,
        "seed": result.seed,
        "threshold": result.threshold,
        "parameters": json.dumps(result.parameters, sort_keys=True),
    }
    for metric in METRIC_COLUMNS:
        row[f"validation_{metric}"] = result.validation[metric]
        row[f"test_{metric}"] = result.test[metric]
    return row


def prediction_rows(result, dataset, split_name: str, probabilities) -> list[dict]:
    if probabilities is None:
        return []
    split = getattr(dataset, split_name)
    rows = []
    for index, probability in enumerate(probabilities):
        row = {
            "dataset": dataset.name,
            "split": split_name,
            "model": result.name,
            "seed": result.seed,
            "row_id": int(split.row_ids[index]),
            "y_true": int(split.y[index]),
            "score": float(probability),
            "threshold": float(result.threshold),
            "prediction": int(probability >= result.threshold),
        }
        row.update(split.metadata.iloc[index].to_dict())
        rows.append(row)
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, block in frame.groupby("model", sort=False):
        row = {"model": model}
        for split in ("validation", "test"):
            for metric in METRIC_COLUMNS:
                values = block[f"{split}_{metric}"].astype(float)
                row[f"{split}_{metric}_mean"] = values.mean()
                row[f"{split}_{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = parse_args()
    from hafnet.baselines import applicable_baselines, candidate_parameters, fit_baseline, select_parameters
    from hafnet.config import load_config
    from hafnet.data import prepare_dataset
    from hafnet.experiment import METRIC_COLUMNS
    from hafnet.utils import ensure_dir, validate_disjoint_seeds, validate_seed_list, write_json

    from hafnet.provenance import load_selected_config, selection_metadata, runtime_manifest
    config = load_config(args.config)
    baseline_config = load_config(args.baseline_config)
    output = ensure_dir(args.output)
    if (output / "baseline_runs.csv").exists():
        raise FileExistsError("Choose an empty baseline output directory.")
    if args.no_search:
        raise ValueError("First-candidate evaluation is not a selected benchmark. Omit --no-search.")
    seeds = validate_seed_list(
        args.seeds if args.seeds is not None else config["training"]["seeds"], "Evaluation seeds"
    )
    dataset_name = str(config["dataset"]["name"]).lower()
    allowed = applicable_baselines(dataset_name)
    selected_models = list(allowed) if args.models == ["all"] else args.models
    if not selected_models or len(selected_models) != len(set(selected_models)):
        raise ValueError("At least one unique comparison method is required.")
    unsupported = [model for model in selected_models if model not in allowed]
    if unsupported:
        raise ValueError(f"Methods not applicable to {dataset_name}: {unsupported}")

    chosen_parameters: dict[str, dict] = {}
    if args.parameters:
        _, chosen_parameters = load_selected_config(config, args.parameters)
        absent = set(selected_models).difference(chosen_parameters)
        if absent:
            raise ValueError(f"Selected parameter file lacks requested methods: {sorted(absent)}")

    tuning_seeds = validate_seed_list(
        config["training"].get("tuning_seeds", [31415]), "Tuning seeds"
    )
    tuning_dataset = prepare_dataset(config)
    candidate_counts: dict[str, int] = {}
    tuning_trials: list[dict] = []
    selection_by_model: dict[str, str] = {}
    for model_name in selected_models:
        candidates = candidate_parameters(model_name, baseline_config, tuning_dataset)
        candidate_counts[model_name] = len(candidates)
        if model_name in chosen_parameters:
            selection_by_model[model_name] = "provided"
            continue
        if model_name not in chosen_parameters:
            selected, trials = select_parameters(
                model_name,
                tuning_dataset,
                tuning_seeds,
                baseline_config,
                return_trials=True,
                device=str(config["training"].get("device", "auto")),
            )
            chosen_parameters[model_name] = selected
            selection_by_model[model_name] = (
                "mean_validation_average_precision" if len(candidates) > 1 else "single_candidate"
            )
            for trial in trials:
                tuning_trials.append(
                    {
                        **trial,
                        "parameters": json.dumps(trial["parameters"], sort_keys=True),
                    }
                )
    searched = [name for name, method in selection_by_model.items() if method == "mean_validation_average_precision"]
    if searched or any(method == "provided" for method in selection_by_model.values()):
        validate_disjoint_seeds(seeds, tuning_seeds)
    trials_path = output / "tuning_trials.csv"
    if args.parameters:
        source = Path(args.parameters).parent / chosen_parameters["_provenance"]["trial_file"]
        import shutil
        shutil.copyfile(source, trials_path)
        chosen_parameters["_provenance"]["trial_file"] = trials_path.name
    else:
        pd.DataFrame(tuning_trials, columns=["model", "candidate_id", "seed", "parameters", "validation_average_precision",
            "stage1_validation_average_precision", "wrapper_best_cv_score", "selected_feature_count"]).to_csv(trials_path, index=False)
        chosen_parameters["_provenance"] = selection_metadata(config, tuning_seeds,
            sum(candidate_counts.values()), trials_path)
    write_json(output / "selected_parameters.json", chosen_parameters)
    write_json(output / "run_manifest.json", runtime_manifest(config,
        evaluation_seeds=seeds, selected_models=selected_models,
        parameter_selection=chosen_parameters["_provenance"], output_kind="model_evaluation"))
    write_json(
        output / "tuning_manifest.json",
        {
            "dataset": dataset_name,
            "split_seed": config["dataset"].get("split_seed", "chronological"),
            "tuning_seeds": tuning_seeds,
            "selection_by_model": selection_by_model,
            "selection_metric": "validation_average_precision" if searched else None,
            "test_evaluated_during_tuning": False,
            "candidate_counts": candidate_counts,
            "candidate_seed_evaluations": {
                name: count * len(tuning_seeds)
                if selection_by_model[name] == "mean_validation_average_precision"
                else 0
                for name, count in candidate_counts.items()
            },
            "selected_models": selected_models,
        },
    )

    records = []
    prediction_records = []
    for seed in seeds:
        dataset = prepare_dataset(config)
        for model_name in selected_models:
            result = fit_baseline(
                model_name,
                dataset,
                seed,
                parameters=chosen_parameters.get(model_name, {}),
                evaluate_test=True,
                device=str(config["training"].get("device", "auto")),
            )
            records.append(record_from_result(result))
            prediction_records.extend(
                prediction_rows(result, dataset, "validation", result.validation_probability)
            )
            prediction_records.extend(prediction_rows(result, dataset, "test", result.test_probability))

    runs = pd.DataFrame(records)
    summary = summarize(runs)
    runs.to_csv(output / "baseline_runs.csv", index=False)
    summary.to_csv(output / "baseline_summary.csv", index=False)
    pd.DataFrame(prediction_records).to_csv(output / "baseline_predictions.csv", index=False)
    print(summary.to_string(index=False))
