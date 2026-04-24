"""CLI wrapper for metric-ranked XGBoost tuning.

Core tuning logic lives in ``football_ai.tuning``.  This script only handles
argparse, YAML loading, and CLI overrides.
"""
from __future__ import annotations

import argparse
from typing import Any

from football_ai.config import load_config, setup_logging
from football_ai.tuning import DEFAULT_CONFIG_PATH, run_xgboost_tuning


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tuning entrypoint."""
    parser = argparse.ArgumentParser(
        description="Bayesian tuning for XGBoost on engineered VAEP features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file.",
    )
    parser.add_argument("--target-col", type=str, default=None, help="Target column: scores or concedes")
    parser.add_argument("--data-file", type=str, default=None, help="Path to HDF5 data file")
    parser.add_argument("--output-dir", type=str, default=None, help="Root directory for run outputs")
    parser.add_argument("--n-trials", type=int, default=None, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random state")
    parser.add_argument("--device", type=str, default=None, help="XGBoost device: cpu or cuda")
    return parser.parse_args()


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_col": args.target_col,
        "data_file": args.data_file,
        "output_dir": args.output_dir,
        "n_trials": args.n_trials,
        "timeout_seconds": args.timeout,
        "seed": args.seed,
        "device": args.device,
    }


def main() -> int:
    """Load YAML config, apply CLI overrides, and run tuning."""
    args = parse_args()
    setup_logging("tune_xgboost")
    cfg = load_config(args.config)
    result = run_xgboost_tuning(cfg, cli_overrides=_cli_overrides(args))
    print(f"Best trial: {result.best_trial_number}")
    print(f"Best source-val {result.selected_metric}: {result.best_score:.6f}")
    print(f"Results saved in: {result.run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
