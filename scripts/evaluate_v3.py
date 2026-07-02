"""Standalone V3 evaluation entry point."""

from __future__ import annotations

import argparse

from stockxpert.v3.config import load_v3_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_nifty100.yaml")
    args = parser.parse_args()

    cfg = load_v3_config(args.config)
    print(f"Loaded V3 evaluation config: {cfg.run.name}")
    print("Use stockxpert.v3.reporting.V3Evaluator for statistical tests and breakdowns.")


if __name__ == "__main__":
    main()

