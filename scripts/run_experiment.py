#!/usr/bin/env python3
from __future__ import annotations
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAF-Net across configured random seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/hafnet")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    from hafnet.experiment import run_hafnet
    _, summary = run_hafnet(args.config, ("full",), args.seeds, args.output, args.save_checkpoints, args.parameters)
    print(summary.to_string(index=False))
