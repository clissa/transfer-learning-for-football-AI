from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier


# =========================
# Global config
# =========================
DATA_FILE = Path("data/spadl_data_rich/major_leagues.h5")
RICH_ACTIONS_KEYS = ["rich_action", "rich_actions"]
TARGET_COL = "scores"  # or "concedes"

RESULTS_PATH = Path("results/debug_train_xgboost_rich")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

VALIDATION_COMPETITIONS = [
    "Serie A",
    "Ligue 1",
]
TEST_COMPETITIONS = [
    "Champions League",
    "UEFA Europa League",
]
# Keep None to automatically use all remaining competitions as train.
TRAIN_COMPETITIONS: list[str] | None = None
RANDOM_STATE = 20260305

# XGBoost objective/loss examples:
# - "binary:logistic" (binary probabilities)
# - "binary:hinge" (binary labels)
# - "reg:logistic" (regression-style logistic)
# OBJECTIVE = "binary:logistic"
OBJECTIVE = "binary:logistic"

# Eval metric examples:
# - "logloss", "auc", "aucpr", "error"
# EVAL_METRIC: str | list[str] = ["logloss", "auc", "aucpr"]
EVAL_METRIC: str | list[str] = ["aucpr", "auc", "logloss"]

# Which validation set to pass into fit(eval_set=...):
# - "train_val_split" -> use predefined validation competitions
# - "none" -> do not pass eval_set to fit
VALIDATION_SET_MODE = "train_val_split"

# If True, include training data in eval_set as first tuple: [(X_train, y_train), (X_val, y_val)]
INCLUDE_TRAIN_IN_EVAL_SET = False

# Decision threshold for final metrics from predicted probabilities
# PRED_THRESHOLD = 0.5
PRED_THRESHOLD = 0.5
# If True, select threshold by maximizing validation F1.
ENABLE_THRESHOLD_SWEEP = True
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEPS = 90

# Keep only non-None params when building estimator / fit kwargs.
XGB_MODEL_CONFIG: dict[str, Any] = {
    # General
    "objective": OBJECTIVE,
    "booster": "gbtree",  # gbtree, gblinear, dart
    "tree_method": "hist",  # auto, exact, approx, hist
    "device": "cuda",  # cpu, cuda
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "verbosity": 1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,

    # Tree complexity / regularization
    "max_depth": 6,
    "max_leaves": 0,
    "max_bin": 256,
    "grow_policy": "lossguide",  # depthwise, lossguide
    "gamma": 0.0,
    "min_child_weight": 1.0,
    "max_delta_step": 0.0,

    # Subsampling
    "subsample": 0.8,
    "sampling_method": "uniform",  # uniform, gradient_based (gpu_hist)
    "colsample_bytree": 0.8,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,

    # L1/L2 and imbalance
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    # "scale_pos_weight": None,
    "scale_pos_weight": 90,

    # Base / missing
    "base_score": None,
    "missing": np.nan,

    # Random forest / multiclass extras
    "num_parallel_tree": 1,
    "multi_strategy": None,

    # Constraints
    "monotone_constraints": None,
    "interaction_constraints": None,

    # Categorical handling
    "enable_categorical": False,
    "feature_types": None,
    "max_cat_to_onehot": None,
    "max_cat_threshold": None,

    # sklearn wrapper controls
    "importance_type": "gain",
    "validate_parameters": True,
    "eval_metric": EVAL_METRIC,
    "early_stopping_rounds": 130,
    "callbacks": None,
}

XGB_FIT_CONFIG: dict[str, Any] = {
    # Sample weights
    "sample_weight": None,
    "base_margin": None,

    # Per-eval-set optional weights/margins
    "sample_weight_eval_set": None,
    "base_margin_eval_set": None,

    # Continue training from previous model/booster
    "xgb_model": None,

    # Logging in fit
    "verbose": True,
}
# =========================


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _read_rich_actions(data_file: Path, key_candidates: list[str]) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(f"Missing H5 dataset file: {data_file}")

    with pd.HDFStore(str(data_file), mode="r") as store:
        available_keys = set(store.keys())
        selected_key = None
        for key in key_candidates:
            key_with_slash = f"/{key}"
            if key_with_slash in available_keys:
                selected_key = key
                break
        if selected_key is None:
            raise KeyError(
                f"None of keys {key_candidates} found in {data_file}. "
                f"Available keys: {sorted(available_keys)}"
            )
        print(f"Using H5 key '{selected_key}' from {data_file}")
        return store.get(f"/{selected_key}")


def load_xy_from_rich_actions(
    target_col: str,
    data_file: Path,
    key_candidates: list[str],
    validation_competitions: list[str],
    test_competitions: list[str],
    train_competitions: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be 'scores' or 'concedes'")

    df = _read_rich_actions(data_file=data_file, key_candidates=key_candidates).copy()
    if "competition_name" not in df.columns:
        raise KeyError("Expected 'competition_name' column in rich actions dataset")

    df["competition_name"] = df["competition_name"].astype(str)
    all_competitions = sorted(df["competition_name"].dropna().unique().tolist())

    val_set = set(validation_competitions)
    test_set = set(test_competitions)
    if train_competitions is None:
        train_set = set(all_competitions) - val_set - test_set
    else:
        train_set = set(train_competitions)

    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlap:
        raise ValueError(f"Train/val/test competition sets overlap: {sorted(overlap)}")

    df_train = df[df["competition_name"].isin(train_set)].copy()
    df_val = df[df["competition_name"].isin(val_set)].copy()
    df_test = df[df["competition_name"].isin(test_set)].copy()

    if df_train.empty:
        raise ValueError("Train split is empty after applying competition filters")
    if df_val.empty:
        raise ValueError("Validation split is empty after applying competition filters")
    if df_test.empty:
        raise ValueError("Test split is empty after applying competition filters")

    excluded_cols = {"game_id", "action_id", "scores", "concedes", "competition_name"}
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number, bool]).columns
        if col not in excluded_cols
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns available in rich actions dataset")

    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = df_train[target_col].astype(int)

    X_val = df_val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = df_val[target_col].astype(int)

    X_test = df_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = df_test[target_col].astype(int)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        sorted(train_set),
        sorted(val_set),
        sorted(test_set),
    )


def build_eval_set(
    mode: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    train_feature_cols: list[str],
) -> tuple[list[tuple[pd.DataFrame, pd.Series]] | None, str]:
    if mode == "none":
        return None, "none"

    if mode == "train_val_split":
        eval_set: list[tuple[pd.DataFrame, pd.Series]] = []
        if INCLUDE_TRAIN_IN_EVAL_SET:
            eval_set.append((X_train, y_train))
        eval_set.append((X_val.reindex(columns=train_feature_cols, fill_value=0), y_val))
        return eval_set, "train_val_split"

    raise ValueError(
        "VALIDATION_SET_MODE must be one of: 'train_val_split', 'none'"
    )


def evaluate_binary(
    y_proba: np.ndarray,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    
    # Naive classifier: always predict majority class
    # y_pred_naive = np.full_like(y, y.mode()[0], dtype=int)
    y_pred_allpositives = np.ones_like(y, dtype=int)
    rng = np.random.default_rng()
    y_pred_smart_naive = rng.choice(
        [0, 1],
        size=len(y),
        p=[1 - y.mean(), y.mean()]
    )
    return {
        "rows": float(len(y)),
        "positive_rate": float(y.mean()),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)) if y.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "brier": float(brier_score_loss(y, y_proba)),
        "smart_naive_precision": float(precision_score(y, y_pred_smart_naive, zero_division=0)),
        "smart_naive_recall": float(recall_score(y, y_pred_smart_naive, zero_division=0)),
        "smart_naive_f1": float(f1_score(y, y_pred_smart_naive, zero_division=0)),
        "positives_precision": float(precision_score(y, y_pred_allpositives, zero_division=0)),
        "positives_recall": float(recall_score(y, y_pred_allpositives, zero_division=0)),
        "positives_f1": float(f1_score(y, y_pred_allpositives, zero_division=0)),
    
    }


def get_positive_class_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores for thresholding."""
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


def sweep_thresholds_for_f1(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> tuple[pd.DataFrame, float]:
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
    rows: list[dict[str, float]] = []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "pred_positive_rate": float(y_pred.mean()),
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_idx = (
        sweep_df.sort_values(["f1", "precision", "recall"], ascending=[False, False, False])
        .index[0]
    )
    best_threshold = float(sweep_df.loc[best_idx, "threshold"])
    return sweep_df, best_threshold


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:>14}: {v:.6f}")


def plot_confusion_matrix(cm, split_name):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {split_name.upper()}")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"confusion_matrix_{split_name}_{TARGET_COL}.png", dpi=300)
    plt.close()
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n=== {split_name.upper()} ===")    
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

def main() -> int:
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_competitions_used,
        val_competitions_used,
        test_competitions_used,
    ) = load_xy_from_rich_actions(
        target_col=TARGET_COL,
        data_file=DATA_FILE,
        key_candidates=RICH_ACTIONS_KEYS,
        validation_competitions=VALIDATION_COMPETITIONS,
        test_competitions=TEST_COMPETITIONS,
        train_competitions=TRAIN_COMPETITIONS,
    )

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    train_feature_cols = list(X_train.columns)

    eval_set, eval_set_name = build_eval_set(
        mode=VALIDATION_SET_MODE,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_feature_cols=train_feature_cols,
    )

    model = XGBClassifier(**_drop_none(XGB_MODEL_CONFIG))

    fit_kwargs = _drop_none(XGB_FIT_CONFIG)
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set

    print("Model:", "xgboost")
    print("Target:", TARGET_COL)
    print("Objective:", XGB_MODEL_CONFIG.get("objective"))
    print("Eval metric:", XGB_MODEL_CONFIG.get("eval_metric"))
    print("Train competitions:", len(train_competitions_used), train_competitions_used)
    print("Validation competitions:", len(val_competitions_used), val_competitions_used)
    print("Test competitions:", len(test_competitions_used), test_competitions_used)
    print("Validation mode:", eval_set_name)

    model.fit(X_train, y_train, **fit_kwargs)
    X_test = X_test.reindex(columns=train_feature_cols, fill_value=0)

    X_val_eval = X_val.reindex(columns=train_feature_cols, fill_value=0)

    y_proba_train = get_positive_class_scores(model, X_train)
    y_proba_val = get_positive_class_scores(model, X_val_eval)
    y_proba_test = get_positive_class_scores(model, X_test)

    selected_threshold = float(PRED_THRESHOLD)
    threshold_sweep_df: pd.DataFrame | None = None
    if ENABLE_THRESHOLD_SWEEP:
        threshold_sweep_df, selected_threshold = sweep_thresholds_for_f1(
            y_true=y_val,
            y_score=y_proba_val,
            threshold_min=THRESHOLD_MIN,
            threshold_max=THRESHOLD_MAX,
            threshold_steps=THRESHOLD_STEPS,
        )
        threshold_sweep_df.to_csv(
            RESULTS_PATH / f"threshold_sweep_xgboost_{TARGET_COL}.csv",
            index=False,
        )
        print(f"\nSelected threshold from validation F1 sweep: {selected_threshold:.4f}")

    train_metrics = evaluate_binary(y_proba_train, y_train, threshold=selected_threshold)
    val_metrics = evaluate_binary(y_proba_val, y_val, threshold=selected_threshold)
    test_metrics = evaluate_binary(y_proba_test, y_test, threshold=selected_threshold)

    _print_metrics("TRAIN", train_metrics)
    _print_metrics("VALIDATION", val_metrics)
    _print_metrics("TEST", test_metrics)

    results_df = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "league_season": [
                "major_leagues_group",
                "major_leagues_group",
                "european_competitions_group",
            ],
            **{k: [train_metrics[k], val_metrics[k], test_metrics[k]] for k in train_metrics},
        }
    )
    results_df.to_csv(RESULTS_PATH / f"metrics_xgboost_positive-focus_{TARGET_COL}.csv", index=False)

    # Compute confusion matrices
    train_cm = confusion_matrix(y_train, (y_proba_train >= selected_threshold).astype(int))
    val_cm = confusion_matrix(y_val, (y_proba_val >= selected_threshold).astype(int))
    test_cm = confusion_matrix(y_test, (y_proba_test >= selected_threshold).astype(int))

    plot_confusion_matrix(train_cm, "train")
    plot_confusion_matrix(val_cm, "validation")
    plot_confusion_matrix(test_cm, "test")

    config_df = pd.DataFrame(
        {
            "model": ["xgboost"],
            "target": [TARGET_COL],
            "objective": [XGB_MODEL_CONFIG.get("objective")],
            "eval_metric": [str(XGB_MODEL_CONFIG.get("eval_metric"))],
            "train_dataset_group": [str(train_competitions_used)],
            "validation_dataset_group": [str(val_competitions_used)],
            "test_dataset_group": [str(test_competitions_used)],
            "data_file": [str(DATA_FILE)],
            "h5_key_candidates": [str(RICH_ACTIONS_KEYS)],
            "validation_set_mode": [eval_set_name],
            "random_state": [RANDOM_STATE],
            "pred_threshold": [PRED_THRESHOLD],
            "selected_threshold": [selected_threshold],
            "threshold_sweep_enabled": [ENABLE_THRESHOLD_SWEEP],
            "threshold_grid": [f"{THRESHOLD_MIN}:{THRESHOLD_MAX}:{THRESHOLD_STEPS}"],
            "xgb_model_config": [str(_drop_none(XGB_MODEL_CONFIG))],
            "xgb_fit_config": [str(_drop_none(XGB_FIT_CONFIG))],
        }
    )
    config_df.to_csv(RESULTS_PATH / f"config_xgboost_{TARGET_COL}.csv", index=False)

    # FOR INTERACTIVE DEBUGGING/RESULTS EXPLORATION
    # Filter to rows where model predicted goal (probability = 1)
    # goals_mask = y_proba_test >= selected_threshold
    # X_test_goals = X_test[goals_mask]
    
    # # Keep only columns with variance (exclude constant columns)
    # relevant_cols = [col for col in X_test_goals.columns if X_test_goals[col].nunique() > 1]
    # X_test_goals[relevant_cols].to_csv(RESULTS_PATH / f"predicted_goals_{TARGET_COL}.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
