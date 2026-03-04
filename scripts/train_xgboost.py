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

from football_ai.utils import list_available_dataset_keys


# =========================
# Global config
# =========================
DATA_DIR = Path("data/spadl_data")
TARGET_COL = "scores"  # or "concedes"

RESULTS_PATH = Path("results/debug_train_xgboost")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

TRAIN_COMPETITION_PREFIXES = [
    "1_bundesliga",
    "la_liga",
    "ligue_1",
    "premier_league",
    "serie_a",
]
TEST_COMPETITION_PREFIXES = [
    "champions_league",
    "uefa_europa_league",
]

# Optional season filters by suffix on dataset key, e.g. ["2015_2016"].
# Keep None to include all available seasons for selected competitions.
TRAIN_SEASON_SUFFIXES: list[str] | None = None
TEST_SEASON_SUFFIXES: list[str] | None = None

# Validation split inside TRAIN_LEAGUE/TRAIN_SEASON (game_id-level split)
VAL_PCT = 0.20
RANDOM_STATE = 20260304

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
# - "train_val_split" -> use X_val/y_val from VAL_PCT split on train league/season
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
THRESHOLD_STEPS = 20

# Keep only non-None params when building estimator / fit kwargs.
XGB_MODEL_CONFIG: dict[str, Any] = {
    # General
    "objective": OBJECTIVE,
    "booster": "gbtree",  # gbtree, gblinear, dart
    "tree_method": "hist",  # auto, exact, approx, hist
    "device": "cuda",  # cpu, cuda
    "n_estimators": 500,
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


def _read_dataset(dataset_key: str, data_dir: Path) -> pd.DataFrame:
    features_path = data_dir / f"features_{dataset_key}.h5"
    labels_path = data_dir / f"labels_{dataset_key}.h5"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    df_features = pd.read_hdf(features_path, key="features")
    df_labels = pd.read_hdf(labels_path, key="labels")
    return df_features.merge(df_labels, on=["game_id", "action_id"], how="inner")


def resolve_dataset_keys(
    available_dataset_keys: list[str],
    competition_prefixes: list[str],
    season_suffixes: list[str] | None = None,
) -> list[str]:
    keys = [
        k
        for k in available_dataset_keys
        if any(k.startswith(prefix + "_") for prefix in competition_prefixes)
    ]
    if season_suffixes:
        keys = [k for k in keys if any(k.endswith(suffix) for suffix in season_suffixes)]
    return sorted(keys)


def load_xy(
    dataset_keys: list[str],
    target_col: str,
    data_dir: Path,
    val_pct: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load and split multiple datasets into train/val based on unique game identifiers.
    This prevents leakage of actions from the same game across splits.
    """
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be 'scores' or 'concedes'")
    if not (0 <= val_pct < 1):
        raise ValueError("val_pct must be in [0, 1)")
    if not dataset_keys:
        raise ValueError("dataset_keys is empty")

    frames: list[pd.DataFrame] = []
    for dataset_key in dataset_keys:
        df_part = _read_dataset(dataset_key=dataset_key, data_dir=data_dir).copy()
        df_part["__dataset_key"] = dataset_key
        frames.append(df_part)
    df = pd.concat(frames, ignore_index=True)

    feature_cols = [
        col
        for col in df.columns
        if col not in {"game_id", "action_id", "scores", "concedes", "__dataset_key"}
    ]

    split_game_key = df["__dataset_key"].astype(str) + "::" + df["game_id"].astype(str)
    game_ids = np.array(sorted(split_game_key.unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(game_ids)

    n_val_games = int(round(len(game_ids) * val_pct))
    val_game_ids = set(game_ids[:n_val_games])
    print(f"Total games: {len(game_ids)}, Validation games: {n_val_games}")

    is_val = split_game_key.isin(val_game_ids)
    df_train = df.loc[~is_val].copy()
    df_val = df.loc[is_val].copy()

    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = df_train[target_col].astype(int)

    X_val = df_val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = df_val[target_col].astype(int)

    return X_train, y_train, X_val, y_val


def load_xy_all(
    dataset_keys: list[str],
    target_col: str,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    if not dataset_keys:
        raise ValueError("dataset_keys is empty")
    frames: list[pd.DataFrame] = []
    for dataset_key in dataset_keys:
        frames.append(_read_dataset(dataset_key=dataset_key, data_dir=data_dir))
    df = pd.concat(frames, ignore_index=True)
    feature_cols = [
        col
        for col in df.columns
        if col not in {"game_id", "action_id", "scores", "concedes"}
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].astype(int)
    return X, y


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
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR.resolve()}")

    available_dataset_keys = list_available_dataset_keys(DATA_DIR)
    train_keys = resolve_dataset_keys(
        available_dataset_keys=available_dataset_keys,
        competition_prefixes=TRAIN_COMPETITION_PREFIXES,
        season_suffixes=TRAIN_SEASON_SUFFIXES,
    )
    test_keys = resolve_dataset_keys(
        available_dataset_keys=available_dataset_keys,
        competition_prefixes=TEST_COMPETITION_PREFIXES,
        season_suffixes=TEST_SEASON_SUFFIXES,
    )
    if not train_keys:
        raise ValueError("No train datasets resolved from TRAIN_COMPETITION_PREFIXES/SEASON_SUFFIXES")
    if not test_keys:
        raise ValueError("No test datasets resolved from TEST_COMPETITION_PREFIXES/SEASON_SUFFIXES")

    X_train, y_train, X_val, y_val = load_xy(
        dataset_keys=train_keys,
        target_col=TARGET_COL,
        data_dir=DATA_DIR,
        val_pct=VAL_PCT,
        random_state=RANDOM_STATE,
    )

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)

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
    print("Train datasets:", len(train_keys), train_keys[:10], "..." if len(train_keys) > 10 else "")
    print("Validation mode:", eval_set_name)
    print("Test datasets:", len(test_keys), test_keys[:10], "..." if len(test_keys) > 10 else "")
    print("Validation pct (game split):", VAL_PCT)

    model.fit(X_train, y_train, **fit_kwargs)

    X_test, y_test = load_xy_all(
        dataset_keys=test_keys,
        target_col=TARGET_COL,
        data_dir=DATA_DIR,
    )
    X_test = X_test.astype(np.float32).reindex(columns=train_feature_cols, fill_value=0)
    y_test = y_test.astype(np.uint8)

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
            "train_dataset_group": [str(TRAIN_COMPETITION_PREFIXES)],
            "test_dataset_group": [str(TEST_COMPETITION_PREFIXES)],
            "train_dataset_keys": [str(train_keys)],
            "test_dataset_keys": [str(test_keys)],
            "validation_set_mode": [eval_set_name],
            "val_pct": [VAL_PCT],
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
    goals_mask = y_proba_test >= selected_threshold
    X_test_goals = X_test[goals_mask]
    
    # Keep only columns with variance (exclude constant columns)
    relevant_cols = [col for col in X_test_goals.columns if X_test_goals[col].nunique() > 1]
    X_test_goals[relevant_cols].to_csv(RESULTS_PATH / f"predicted_goals_{TARGET_COL}.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
