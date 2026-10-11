#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from hafnet.config import load_config
from hafnet.statistics import paired_prediction_inference

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAF-Net component and structure controls.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/ablation")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    from hafnet.experiment import run_hafnet
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    config = load_config(args.config)
    dataset_name = str(config["dataset"]["name"]).lower()
    if dataset_name == "taiwan":
        variants = ("full", "core", "hfge_bce", "flat_cbf", "flat_bce", "hfge_class_balanced_bce",
                    "hfge_focal_unweighted", "random_hfge", "no_residual", "permuted_prior", "uniform_prior")
        contrasts = [("full", "core"), ("full", "uniform_prior"), ("full", "permuted_prior"),
                     ("core", "flat_cbf"), ("hfge_bce", "flat_bce"), ("core", "hfge_bce"),
                     ("flat_cbf", "flat_bce"), ("hfge_class_balanced_bce", "hfge_bce"),
                     ("hfge_focal_unweighted", "hfge_bce"), ("core", "random_hfge"), ("full", "no_residual")]
    elif dataset_name == "american":
        variants = ("full", "core", "hfge_bce", "flat_cbf", "flat_bce", "hfge_class_balanced_bce",
                    "hfge_focal_unweighted", "equal_intragroup")
        contrasts = [("full", "core"), ("core", "flat_cbf"), ("hfge_bce", "flat_bce"),
                     ("core", "hfge_bce"), ("flat_cbf", "flat_bce"),
                     ("hfge_class_balanced_bce", "hfge_bce"), ("hfge_focal_unweighted", "hfge_bce"),
                     ("core", "equal_intragroup")]
    else:
        raise ValueError(f"Unsupported ablation dataset: {dataset_name}")
    _, summary = run_hafnet(args.config, variants, args.seeds, args.output, False, args.parameters)
    predictions = pd.read_csv(Path(args.output) / "hafnet_predictions.csv", float_precision="round_trip")
    inference_rows, seed_blocks = [], []
    cluster_column = "company_name" if dataset_name == "american" else None
    for contrast_index, (model_a, model_b) in enumerate(contrasts):
        row = {"dataset": dataset_name, "model_a": model_a, "model_b": model_b}
        for metric_index, metric in enumerate(("f1", "mcc")):
            inference, seed_differences = paired_prediction_inference(predictions, model_a, model_b, metric,
                cluster_column=cluster_column, repetitions=args.bootstrap, seed=3100 + contrast_index * 10 + metric_index)
            for key in ("test_unit_bootstrap_mean_difference", "test_unit_bootstrap_ci_low", "test_unit_bootstrap_ci_high",
                        "seed_mean_difference", "seed_std_difference"):
                row[f"{metric}_{key}"] = inference[key]
            seed_differences.insert(0, "dataset", dataset_name)
            seed_blocks.append(seed_differences)
        inference_rows.append(row)
    pd.DataFrame(inference_rows).to_csv(Path(args.output) / "ablation_inference.csv", index=False)
    pd.concat(seed_blocks, ignore_index=True).to_csv(Path(args.output) / "ablation_seed_differences.csv", index=False)
    print(summary.to_string(index=False))
