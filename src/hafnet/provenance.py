"""Data, configuration, selection, and source fingerprints for scientific outputs."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version, PackageNotFoundError

def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def object_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def data_contract(config: dict) -> dict:
    dataset = deepcopy(config["dataset"])
    raw = file_sha256(dataset.pop("path"))
    mapping = file_sha256(dataset.pop("mapping"))
    dataset.pop("sha256", None)
    return {"dataset": dataset, "raw_sha256": raw, "mapping_sha256": mapping,
        "preprocessing": "training-only 1/99 percentile winsorization; clipped-training mean and population SD"}

def design_contract(config: dict) -> dict:
    training = {k: v for k, v in config["training"].items() if k not in {"device", "seeds", "tuning_seeds"}}
    return {"data": data_contract(config), "model": config["model"], "loss": config["loss"],
        "training": training, "controls": config.get("controls", {})}

def selection_metadata(config: dict, tuning_seeds, candidates: int, trials_path) -> dict:
    return {"schema": 1, "design_sha256": object_sha256(design_contract(config)),
        "data_contract": data_contract(config), "tuning_seeds": list(tuning_seeds),
        "candidate_count": int(candidates), "selection_metric": "validation_average_precision",
        "test_evaluated": False, "trial_file": Path(trials_path).name,
        "trial_sha256": file_sha256(trials_path)}

def load_selected_config(config: dict, path) -> tuple[dict, dict]:
    path = Path(path)
    selected = json.loads(path.read_text(encoding="utf-8"))
    metadata = selected.get("_provenance")
    if not isinstance(metadata, dict) or metadata.get("schema") != 1:
        raise ValueError("Selected parameters require a validation-selection provenance record.")
    if metadata["design_sha256"] != object_sha256(design_contract(config)):
        raise ValueError("Selected parameters do not belong to this data/design configuration.")
    if metadata.get("test_evaluated") is not False or metadata.get("selection_metric") != "validation_average_precision":
        raise ValueError("Parameter selection must be validation-only average precision.")
    trial_file = path.parent / metadata["trial_file"]
    if file_sha256(trial_file) != metadata["trial_sha256"]:
        raise ValueError("The parameter-selection trial record does not match its fingerprint.")
    result = deepcopy(config)
    for section in ("model", "training", "loss"):
        result.setdefault(section, {}).update(selected.get(section, {}))
    return result, selected

def runtime_manifest(config: dict, **details) -> dict:
    dependencies = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "lightgbm", "catboost", "imbalanced-learn"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not installed"
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src" / "hafnet").glob("*.py"))
    paths += list((root / "scripts").glob("*.py"))
    paths += list((root / "configs").glob("*.yaml"))
    sources = {str(path.relative_to(root)): file_sha256(path) for path in sorted(paths)}
    return {"schema": 1, "data_contract": data_contract(config), "design_sha256": object_sha256(design_contract(config)),
        "python": platform.python_version(), "dependencies": dependencies, "source_sha256": sources, **details}
