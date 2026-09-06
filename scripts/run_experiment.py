#!/usr/bin/env python3
from __future__ import annotations

import argparse

from hafnet.experiment import run_hafnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAF-Net across configured random seeds.")
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument("--output", default="outputs/hafnet", help="Output directory")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seed subset")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save model state dictionaries")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runs, summary = run_hafnet(
        config_path=args.config,
        variants=("full",),
        seeds=args.seeds,
        output_dir=args.output,
        save_checkpoints=args.save_checkpoints,
    )
    print(summary.to_string(index=False))
