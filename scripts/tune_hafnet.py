#!/usr/bin/env python3
from __future__ import annotations
import argparse
from copy import deepcopy
from itertools import product
import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select HAF-Net settings using validation average precision.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--search", default="configs/hafnet_search.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output", default="outputs/hafnet_tuning")
    return parser.parse_args()

def candidates(search: dict) -> list[dict[str, int | float]]:
    model = search["model"]
    return [{"subgroup_dim": int(d), "fusion_dim": int(model["fusion_dim"]), "dropout": float(p), "learning_rate": float(lr)}
            for d, p, lr in product(model["subgroup_dim"], model["dropout"], search["training"]["learning_rate"])]

if __name__ == "__main__":
    args = parse_args()
    from hafnet.config import load_config
    from hafnet.data import prepare_dataset
    from hafnet.trainer import train_one_seed
    from hafnet.utils import ensure_dir, validate_disjoint_seeds, validate_seed_list, write_json
    from hafnet.provenance import selection_metadata
    config = load_config(args.config)
    search = load_config(args.search)
    seeds = validate_seed_list(args.seeds or config["training"]["tuning_seeds"], "Tuning seeds")
    validate_disjoint_seeds(seeds, config["training"]["seeds"])
    dataset = prepare_dataset(config)
    settings = candidates(search)
    output = ensure_dir(args.output)
    if (output / "tuning_runs.csv").exists():
        raise FileExistsError("Choose an empty tuning output directory.")
    rows = []
    for candidate_id, parameters in enumerate(settings, start=1):
        for seed in seeds:
            current = deepcopy(config)
            current["model"].update({k: v for k, v in parameters.items() if k != "learning_rate"})
            current["training"]["learning_rate"] = parameters["learning_rate"]
            result = train_one_seed(dataset, current, seed=seed, variant="full", evaluate_test=False)
            rows.append({"candidate_id": candidate_id, "seed": seed, **parameters,
                         "validation_average_precision": result.validation["pr_auc"],
                         "validation_f1": result.validation["f1"], "epochs": result.epochs})
    runs = pd.DataFrame(rows)
    summary = runs.groupby("candidate_id", as_index=False).agg(
        subgroup_dim=("subgroup_dim", "first"), fusion_dim=("fusion_dim", "first"), dropout=("dropout", "first"),
        learning_rate=("learning_rate", "first"), validation_average_precision_mean=("validation_average_precision", "mean"),
        validation_average_precision_std=("validation_average_precision", "std"), validation_f1_mean=("validation_f1", "mean"),
        validation_f1_std=("validation_f1", "std")).fillna(0.0)
    winner = summary.sort_values(["validation_average_precision_mean", "candidate_id"], ascending=[False, True]).iloc[0]
    selected = {"model": {"subgroup_dim": int(winner["subgroup_dim"]), "fusion_dim": int(winner["fusion_dim"]),
                           "dropout": float(winner["dropout"])}, "training": {"learning_rate": float(winner["learning_rate"])}}
    runs.to_csv(output / "tuning_runs.csv", index=False)
    summary.to_csv(output / "tuning_summary.csv", index=False)
    selected["_provenance"] = selection_metadata(config, seeds, len(settings), output / "tuning_runs.csv")
    write_json(output / "selected_parameters.json", selected)
    write_json(output / "tuning_manifest.json", {"dataset": dataset.name,
        "split_seed": config["dataset"].get("split_seed", "chronological"), "tuning_seeds": seeds,
        "candidate_count": len(settings), "candidate_seed_runs": len(settings) * len(seeds),
        "selection_metric": "validation_average_precision", "test_evaluated_during_tuning": False, "search_grid": search})
