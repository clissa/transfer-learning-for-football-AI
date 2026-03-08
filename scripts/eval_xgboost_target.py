"""Evaluate a pre-trained XGBoost model on target competitions (0-shot).

Reads a merged VAEP HDF5 dataset, loads a pickled XGBoost classifier, filters
rows by TARGET_COMPETITIONS, and prints binary-classification metrics including
naive baselines — matching the format used in train_xgboost.py logs.

Configure the three globals below; no CLI or YAML needed.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from football_ai.data import read_h5_table
from football_ai.evaluation import (
    evaluate_binary_with_baselines,
    get_positive_class_scores,
)
from football_ai.training import select_vaep_feature_cols

# ═══════════════════════════════════════════════
# Configurable globals
# ═══════════════════════════════════════════════
DATA_FILE: str | Path = "data/vaep_data/major_leagues_vaep.h5"
DATA_KEY: str = "vaep_data"
MODEL_FILE: str | Path = "models/xgboost_scores_20260307.pkl"

TARGET_COL: str = "scores"
TARGET_COMPETITIONS: list[str] = ["Champions League", "UEFA Europa League"]
THRESHOLD: float = 0.5


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════

def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """Pretty-print a metrics dict, right-aligning keys to the longest."""
    width = max(len(k) for k in metrics)
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:>{width}}: {v:.6f}")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main() -> None:
    # ── Load data ──────────────────────────────
    print(f"Loading data from {DATA_FILE} (key={DATA_KEY!r}) ...")
    df = read_h5_table(data_file=DATA_FILE, key_candidates=[DATA_KEY])

    if "competition_name" not in df.columns:
        raise KeyError("Expected 'competition_name' column in dataset")

    df["competition_name"] = df["competition_name"].astype(str)
    df_target = df[df["competition_name"].isin(TARGET_COMPETITIONS)].copy()
    print(
        f"Target rows: {len(df_target)}  "
        f"(competitions: {sorted(df_target['competition_name'].unique())})"
    )
    if df_target.empty:
        raise ValueError("No rows found for TARGET_COMPETITIONS")

    # ── Features / labels ──────────────────────
    feature_cols = select_vaep_feature_cols(df_target.columns)
    X_target = df_target[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_target = df_target[TARGET_COL].astype(int)

    # ── Load model ─────────────────────────────
    model_path = Path(MODEL_FILE)
    print(f"Loading model from {model_path} ...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── Predict & evaluate ─────────────────────
    y_proba = get_positive_class_scores(model, X_target)
    metrics = evaluate_binary_with_baselines(y_proba, y_target, threshold=THRESHOLD)

    print_metrics("TARGET (0-shot)", metrics)


if __name__ == "__main__":
    main()
