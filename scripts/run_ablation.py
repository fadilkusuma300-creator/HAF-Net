#!/usr/bin/env python3
from __future__ import annotations

import argparse

from hafnet.config import load_config
from hafnet.experiment import run_hafnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HAF-Net component and structure controls.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/ablation")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if str(config["dataset"]["name"]).lower() == "taiwan":
        variants = (
            "full",
            "core",
            "hfge_bce",
            "flat_cbf",
            "flat_bce",
            "random_hfge",
            "no_residual",
            "no_tcar",
            "permuted_prior",
        )
    else:
        variants = ("full", "no_tcl")
    _, summary = run_hafnet(
        config_path=args.config,
        variants=variants,
        seeds=args.seeds,
        output_dir=args.output,
    )
    print(summary.to_string(index=False))
