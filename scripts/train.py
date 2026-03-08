"""Train a single sklearn model (rf / logreg / mlp) on merged SPADL features.

Uses ``load_xy_competition_split`` to split train/val/test by competition
name (all seasons for each specified league).  Reads a single HDF5 file
containing a ``full_data`` key.

Usage examples
--------------
# Using YAML config
python -m scripts.train --config configs/train_sklearn.yaml

# Override model from CLI
python -m scripts.train --config configs/train_sklearn.yaml --model rf

# Legacy: runs with hardcoded module-level defaults (no --config needed)
python -m scripts.train
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_ai.config import load_config, merge_cli_overrides, resolve_random_state
from football_ai.evaluation import (
    evaluate_binary,
    get_positive_class_scores,
    print_metrics,
)
from football_ai.training import (
    build_sklearn_model,
    load_xy_competition_split,
)

# =========================
# Global config
# =========================
DATA_FILE = Path("data/spadl_full_data")
DATASET_NAME = "women_leagues.h5"          # configurable filename
KEY_CANDIDATES = ["full_data"]
TARGET_COL = "scores"                      # or "concedes"


TRAIN_COMPETITIONS = ["FA Women's Super League"]
VALIDATION_COMPETITIONS = ["UEFA Women's Euro"]
TEST_COMPETITIONS = ["Women's World Cup"]

RANDOM_STATE: int | None = 20260306

MODEL_NAME = "mlp"  # rf, logreg, mlp
RESULTS_PATH = Path(f"results/{MODEL_NAME}_women")
MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "rf": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
        "criterion": "gini",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "logreg": {
        "C": 0.1,
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": RANDOM_STATE,
    },
    "mlp": {
        "hidden_layer_sizes": (256, 128, 64),
        "alpha": 1e-3,
        "solver": "adam",
        "early_stopping": True,
        "n_iter_no_change": 10,
        "tol": 1e-5,
        "max_iter": 200,
        "verbose": True,
        "random_state": RANDOM_STATE,
    },
}
# =========================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a single sklearn model on merged SPADL features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file (e.g. configs/train_sklearn.yaml)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name: rf, logreg, mlp")
    parser.add_argument("--target-col", type=str, default=None, help="Target column: scores or concedes")
    parser.add_argument("--data-file", type=str, default=None, help="Path to HDF5 data file")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="HDF5 filename inside data dir (e.g. women_leagues.h5)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing the H5 file (default: data/spadl_full_data)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for result CSVs")
    parser.add_argument("--seed", type=int, default=None, help="Random state")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Build effective config: module defaults -> YAML -> CLI overrides
    cfg: dict[str, Any] = {
        "data": {
            "file": str(DATA_FILE / DATASET_NAME),
            "dir": str(DATA_FILE),
            "dataset_name": DATASET_NAME,
            "key_candidates": KEY_CANDIDATES,
            "target_col": TARGET_COL,
        },
        "split": {
            "train_competitions": TRAIN_COMPETITIONS,
            "validation_competitions": VALIDATION_COMPETITIONS,
            "test_competitions": TEST_COMPETITIONS,
        },
        "model": {
            "name": MODEL_NAME,
            "random_state": RANDOM_STATE,
            "config": MODEL_CONFIG,
        },
        "output": {"dir": str(RESULTS_PATH)},
    }
    if args.config is not None:
        cfg.update(load_config(args.config))
    cfg = merge_cli_overrides(cfg, {
        "model.name": args.model,
        "data.target_col": args.target_col,
        "data.file": args.data_file,
        "data.dataset_name": args.dataset_name,
        "data.dir": args.data_dir,
        "output.dir": args.output_dir,
        "model.random_state": args.seed,
    })

    # Unpack effective config
    data_cfg = cfg["data"]
    # Resolve data file: explicit --data-file wins, then YAML data.file, then dir + dataset_name
    if args.data_file:
        data_file = Path(args.data_file)
    elif "file" in data_cfg and data_cfg["file"]:
        data_file = Path(data_cfg["file"])
    else:
        data_file = Path(data_cfg["dir"]) / data_cfg["dataset_name"]

    key_candidates: list[str] = data_cfg.get("key_candidates", KEY_CANDIDATES)
    target_col: str = data_cfg["target_col"]

    split_cfg = cfg.get("split", {})
    train_competitions: list[str] = split_cfg.get("train_competitions", TRAIN_COMPETITIONS)
    validation_competitions: list[str] = split_cfg.get("validation_competitions", VALIDATION_COMPETITIONS)
    test_competitions: list[str] = split_cfg.get("test_competitions", TEST_COMPETITIONS)

    model_name: str = cfg["model"]["name"]
    random_state: int = resolve_random_state(
        cfg["model"].get("random_state"), RANDOM_STATE,
    )
    model_configs: dict = cfg["model"].get("config", MODEL_CONFIG)
    results_path = Path(cfg["output"]["dir"])
    results_path.mkdir(parents=True, exist_ok=True)

    # Inject random_state into the selected model config
    selected_model_cfg = dict(model_configs.get(model_name, {}))
    selected_model_cfg.setdefault("random_state", random_state)
    if "hidden_layer_sizes" in selected_model_cfg and isinstance(
        selected_model_cfg["hidden_layer_sizes"], list
    ):
        selected_model_cfg["hidden_layer_sizes"] = tuple(selected_model_cfg["hidden_layer_sizes"])

    if not data_file.exists():
        raise FileNotFoundError(f"Data file does not exist: {data_file.resolve()}")

    # ---- Load data (competition split) ----
    print(f"Loading data from {data_file} (keys: {key_candidates})")
    print(f"  train competitions: {train_competitions}")
    print(f"  val   competitions: {validation_competitions}")
    print(f"  test  competitions: {test_competitions}")

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_comps, val_comps, test_comps,
    ) = load_xy_competition_split(
        target_col=target_col,
        data_file=data_file,
        key_candidates=key_candidates,
        train_competitions=train_competitions,
        validation_competitions=validation_competitions,
        test_competitions=test_competitions,
    )

    print(f"  Train:  {len(X_train):>8,} rows  {train_comps}")
    print(f"  Val:    {len(X_val):>8,} rows  {val_comps}")
    print(f"  Test:   {len(X_test):>8,} rows  {test_comps}")

    # ---- Build + train model ----
    model = build_sklearn_model(
        model_name=model_name,
        model_config=selected_model_cfg,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # ---- Evaluate ----
    y_proba_train = get_positive_class_scores(model, X_train)
    y_proba_val = get_positive_class_scores(model, X_val)
    y_proba_test = get_positive_class_scores(model, X_test)

    train_metrics = evaluate_binary(y_proba_train, y_train)
    val_metrics = evaluate_binary(y_proba_val, y_val)
    test_metrics = evaluate_binary(y_proba_test, y_test)

    print(f"\nModel: {model_name}  |  Target: {target_col}")
    print_metrics("TRAIN", train_metrics)
    print_metrics("VALIDATION", val_metrics)
    print_metrics("TEST", test_metrics)

    # ---- Save results ----
    results_df = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "competitions": [
                ", ".join(train_comps),
                ", ".join(val_comps),
                ", ".join(test_comps),
            ],
            **{k: [train_metrics[k], val_metrics[k], test_metrics[k]] for k in train_metrics},
        }
    )
    results_df.to_csv(results_path / f"metrics_{model_name}_{target_col}.csv", index=False)

    config_df = pd.DataFrame(
        {
            "model": [model_name],
            "target": [target_col],
            "data_file": [str(data_file)],
            "train_competitions": [str(train_competitions)],
            "validation_competitions": [str(validation_competitions)],
            "test_competitions": [str(test_competitions)],
            "random_state": [random_state],
            "model_config": [str(selected_model_cfg)],
        }
    )
    config_df.to_csv(results_path / f"config_{model_name}_{target_col}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
