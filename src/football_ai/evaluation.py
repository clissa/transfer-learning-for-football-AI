"""Evaluation utilities for binary classification models.

Provides scoring, metrics computation, threshold sweeps, and visualisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ──────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """Pretty-print a metrics dict under a section header."""
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:>14}: {v:.6f}")


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────


def get_positive_class_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Extract positive-class predicted probabilities.

    Args:
        model: Fitted classifier with ``predict_proba``.
        X: Feature DataFrame.

    Returns:
        1-D array of P(y=1).
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Model has no predict_proba; threshold sweep requires probabilistic scores."
        )
    scores = model.predict_proba(X)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(
            "predict_proba did not return 2-class probabilities. "
            "Use an objective that supports probabilities (e.g. 'binary:logistic')."
        )
    return scores[:, 1]


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────


def evaluate_binary(
    y_proba: np.ndarray,
    y_true: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate binary classification from predicted probabilities.

    Args:
        y_proba: Predicted P(y=1) scores.
        y_true: True binary labels.
        threshold: Decision threshold.

    Returns:
        Dict with ``rows``, ``positive_rate``, ``precision``, ``recall``,
        ``f1``, ``roc_auc``, ``pr_auc``, ``brier``, ``logloss``.
    """
    y_arr = np.asarray(y_true)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "rows": float(len(y_arr)),
        "positive_rate": float(y_arr.mean()),
        "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_arr, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_arr, y_proba))
            if len(np.unique(y_arr)) > 1
            else float("nan")
        ),
        "pr_auc": float(average_precision_score(y_arr, y_proba)),
        "brier": float(brier_score_loss(y_arr, y_proba)),
        "logloss": float(log_loss(y_arr, y_proba)),
    }


def evaluate_binary_with_baselines(
    y_proba: np.ndarray,
    y_true: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Like :func:`evaluate_binary` but also computes naive-baseline metrics.

    Extra keys: ``smart_naive_*``, ``positives_*``.
    """
    metrics = evaluate_binary(y_proba, y_true, threshold)

    y_arr = np.asarray(y_true)
    y_pred_allpositives = np.ones_like(y_arr, dtype=int)
    rng = np.random.default_rng()
    y_pred_smart_naive = rng.choice(
        [0, 1], size=len(y_arr), p=[1 - y_arr.mean(), y_arr.mean()]
    )
    metrics.update({
        "smart_naive_precision": float(precision_score(y_arr, y_pred_smart_naive, zero_division=0)),
        "smart_naive_recall": float(recall_score(y_arr, y_pred_smart_naive, zero_division=0)),
        "smart_naive_f1": float(f1_score(y_arr, y_pred_smart_naive, zero_division=0)),
        "positives_precision": float(precision_score(y_arr, y_pred_allpositives, zero_division=0)),
        "positives_recall": float(recall_score(y_arr, y_pred_allpositives, zero_division=0)),
        "positives_f1": float(f1_score(y_arr, y_pred_allpositives, zero_division=0)),
    })
    return metrics


# ──────────────────────────────────────────────
# Threshold sweep
# ──────────────────────────────────────────────


def sweep_thresholds_for_f1(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_steps: int = 90,
) -> tuple[pd.DataFrame, float]:
    """Sweep decision thresholds and pick the one that maximises F1.

    Args:
        y_true: True binary labels.
        y_score: Predicted P(y=1) scores.
        threshold_min: Lowest threshold in grid.
        threshold_max: Highest threshold in grid.
        threshold_steps: Number of thresholds in grid.

    Returns:
        ``(sweep_df, best_threshold)``
    """
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
    rows: list[dict[str, float]] = []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "pred_positive_rate": float(y_pred.mean()),
        })

    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_idx = (
        sweep_df.sort_values(
            ["f1", "precision", "recall"], ascending=[False, False, False]
        ).index[0]
    )
    best_threshold = float(sweep_df.loc[best_idx, "threshold"])
    return sweep_df, best_threshold


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    split_name: str,
    save_path: str | Path,
) -> None:
    """Plot and save a confusion matrix figure.

    Args:
        cm: Confusion matrix (from ``sklearn.metrics.confusion_matrix``).
        split_name: Label for the split (e.g. ``'train'``, ``'test'``).
        save_path: Full path for the output PNG file.
    """
    from matplotlib import pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {split_name.upper()}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    print(f"\n=== {split_name.upper()} ===")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
