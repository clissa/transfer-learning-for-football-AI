"""Train an XGBoost classifier on merged SPADL features (source/calib/target split).

All core logic lives in ``football_ai.training``.  This script is a thin
CLI wrapper that reads config from module-level constants (or a YAML file)
and calls into the library.

Data split strategy:
- Source: La Liga source data (split into train 80% / validation 20% by matches, stratified by season)
- Model selection: source validation only
- Calib / test / target: reporting-only splits, materialized after fit

Usage examples
--------------
# Using YAML config
python -m scripts.train_xgboost --config configs/train_xgboost.yaml

# Override target column from CLI
python -m scripts.train_xgboost --config configs/train_xgboost.yaml --target-col concedes

# Legacy: runs with hardcoded module-level defaults (no --config needed)
python -m scripts.train_xgboost
"""
from __future__ import annotations

import argparse
import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from football_ai.config import load_config, resolve_random_state, setup_logging
from football_ai.evaluation import (
    evaluate_binary_with_baselines,
    get_positive_class_scores,
    plot_confusion_matrix,
    print_metrics,
    sweep_thresholds_for_f1,
)
from football_ai.training import (
    build_xgb_eval_set,
    drop_none_params,
    load_xy_competition_season_split,
    resolve_xgb_eval_metrics,
    save_model,
)

logger = logging.getLogger(__name__)

# =========================
# Global config
# =========================
DATA_FILE = Path("data/feat_engineered_vaep_data/major_leagues_vaep.h5")
DATA_KEY = ["feat_engineered_vaep_data", "vaep_data"]
TARGET_COL = "scores"  # or "concedes"

RESULTS_PATH = Path("results/train_xgboost_vaep")
MODELS_PATH = Path("models")

# Default competition-season split. The current objective is explicit:
# source is La Liga, and model selection optimizes in-domain source validation.
DEFAULT_SPLIT_CONFIG: dict[str, Any] = {
    "source": {
        "competitions": ["La Liga"],
        "exclude_seasons": ["2019-20", "2020-21"],
    },
    "calib": {
        "competitions": ["Champions League", "UEFA Europa League"],
    },
    "test": {
        "competitions": [
            "La Liga",
            "Premier League",
            "Serie A",
            "1. Bundesliga",
            "Ligue 1",
            "Champions League",
            "UEFA Europa League",
        ],
        "year_shift": {
            "seasons": ["2019-20", "2020-21"],
        },
        "league_season_shift": {
            "explicit": {},
        },
    },
}

# Fraction of source matches to use for validation (stratified by season_id)
VALIDATION_FRAC = 0.2
RANDOM_STATE: int | None = None

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
# - "train_val_split" -> use the in-domain held-out La Liga validation split
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
    "device": "cpu",  # cpu, cuda
    "n_estimators": 4000,
    "learning_rate": 0.03,
    "verbosity": 1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,

    # Tree complexity / regularization
    "max_depth": 6,
    # "max_leaves": 0,
    "max_bin": 256,
    "grow_policy": "depthwise",  # depthwise, lossguide
    "gamma": 0.0,
    "min_child_weight": 1.0,
    "max_delta_step": 1.0,

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
    "scale_pos_weight": 80,

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
    # Compact categorical features are deterministically encoded to int32 in
    # shared preprocessing, so native categorical mode stays off by default.
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


def _normalize_key_candidates(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        return [raw_value]
    return [str(item) for item in raw_value]


def _log_resolved_split(logger_obj: logging.Logger, split_name: str, rows: int, games: int, coverage: pd.DataFrame) -> None:
    competitions = sorted(coverage["competition_name"].astype(str).unique().tolist()) if not coverage.empty else []
    logger_obj.info("%s: rows=%d games=%d competition-seasons=%d competitions=%s", split_name, rows, games, len(coverage), competitions)
    if not coverage.empty:
        logger_obj.info("%s coverage:\n%s", split_name, coverage.to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier on merged SPADL features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file (e.g. configs/train_xgboost.yaml)",
    )
    parser.add_argument("--target-col", type=str, default=None, help="Target column: scores or concedes")
    parser.add_argument("--data-file", type=str, default=None, help="Path to HDF5 data file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for result CSVs")
    parser.add_argument("--seed", type=int, default=None, help="Random state")
    parser.add_argument("--device", type=str, default=None, help="XGBoost device: cpu or cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("train_xgboost")

    # Build effective config: module defaults -> YAML -> CLI overrides
    # We only override the few keys that make sense from CLI; the full model
    # config is best changed via YAML.
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_config(args.config)

    # --- Resolve data settings ---
    data_file = Path(
        args.data_file
        or cfg.get("data", {}).get("file")
        or str(DATA_FILE)
    )
    key_candidates = _normalize_key_candidates(cfg.get("data", {}).get("key_candidates", DATA_KEY))
    target_col: str = args.target_col or cfg.get("data", {}).get("target_col", TARGET_COL)

    # --- Resolve split settings ---
    split_cfg = copy.deepcopy(cfg.get("split", DEFAULT_SPLIT_CONFIG))
    validation_frac: float = float(split_cfg.get("validation_frac", VALIDATION_FRAC))

    # --- Resolve model config ---
    model_cfg_from_yaml = cfg.get("model", {})
    random_state: int = resolve_random_state(
        args.seed, model_cfg_from_yaml.get("random_state"), RANDOM_STATE,
    )

    # Start from module-level defaults, overlay YAML model section, then CLI overrides
    # Keys that are ours (not XGBoost constructor params) must be excluded.
    _NON_XGB_KEYS = {"early_stopping_metric"}
    effective_model_config = dict(XGB_MODEL_CONFIG)
    for k, v in model_cfg_from_yaml.items():
        if k in _NON_XGB_KEYS:
            continue
        if k == "eval_metric" and isinstance(v, str):
            effective_model_config[k] = v
        elif v is not None:
            effective_model_config[k] = v
    effective_model_config["random_state"] = random_state
    if args.device is not None:
        effective_model_config["device"] = args.device

    # --- Resolve fit config ---
    fit_cfg_from_yaml = cfg.get("fit", {})
    effective_fit_config = dict(XGB_FIT_CONFIG)
    effective_fit_config.update({k: v for k, v in fit_cfg_from_yaml.items() if v is not None})

    # --- Resolve validation settings ---
    val_cfg = cfg.get("validation", {})
    validation_set_mode: str = val_cfg.get("mode", VALIDATION_SET_MODE)
    include_train_in_eval_set: bool = val_cfg.get("include_train_in_eval_set", INCLUDE_TRAIN_IN_EVAL_SET)

    # --- Resolve threshold settings ---
    thr_cfg = cfg.get("threshold", {})
    pred_threshold: float = float(thr_cfg.get("pred_threshold", PRED_THRESHOLD))
    enable_threshold_sweep: bool = bool(thr_cfg.get("enabled", ENABLE_THRESHOLD_SWEEP))
    threshold_min: float = float(thr_cfg.get("min", THRESHOLD_MIN))
    threshold_max: float = float(thr_cfg.get("max", THRESHOLD_MAX))
    threshold_steps: int = int(thr_cfg.get("steps", THRESHOLD_STEPS))

    # --- Resolve output ---
    results_path = Path(
        args.output_dir
        or cfg.get("output", {}).get("dir")
        or str(RESULTS_PATH)
    )
    results_path.mkdir(parents=True, exist_ok=True)

    models_path = Path(cfg.get("output", {}).get("models_dir", str(MODELS_PATH)))
    models_path.mkdir(parents=True, exist_ok=True)

    # Unique run identifier (timestamp) to avoid overwriting previous artifacts
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Run ----
    resolved_split = load_xy_competition_season_split(
        target_col=target_col,
        data_file=data_file,
        key_candidates=key_candidates,
        split_config=split_cfg,
        validation_frac=validation_frac,
        random_state=random_state,
    )

    X_train = resolved_split.source_train.X
    y_train = resolved_split.source_train.y
    X_val = resolved_split.source_val.X
    y_val = resolved_split.source_val.y
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)

    train_feature_cols = list(X_train.columns)

    eval_set, eval_set_name = build_xgb_eval_set(
        mode=validation_set_mode,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_feature_cols=train_feature_cols,
        include_train=include_train_in_eval_set,
    )

    # Resolve custom eval metrics (e.g. "recall", "precision", "f1") to callables.
    # XGBoost only allows one custom callable; when multiple are requested the
    # resolver returns a composite callable + a companion callback.
    # It also handles early-stopping direction and metric selection: if the
    # user specified early_stopping_metric, that metric is monitored regardless
    # of eval_metric order.
    early_stopping_metric: str | None = model_cfg_from_yaml.get("early_stopping_metric")
    resolved_metrics, extra_callbacks, es_rounds = resolve_xgb_eval_metrics(
        effective_model_config.get("eval_metric"),
        early_stopping_rounds=effective_model_config.get("early_stopping_rounds"),
        early_stopping_metric=early_stopping_metric,
    )
    effective_model_config["eval_metric"] = resolved_metrics
    effective_model_config["early_stopping_rounds"] = es_rounds  # None if handled by callback
    if extra_callbacks:
        existing_cbs = effective_model_config.get("callbacks") or []
        effective_model_config["callbacks"] = extra_callbacks + list(existing_cbs)

    model = XGBClassifier(**drop_none_params(effective_model_config))

    fit_kwargs = drop_none_params(effective_fit_config)
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set

    logger.info("Model: xgboost")
    logger.info("Target: %s", target_col)
    logger.info("Objective: %s", effective_model_config.get("objective"))
    logger.info("Eval metric: %s", effective_model_config.get("eval_metric"))
    logger.info("Selection objective: in-domain source validation only")
    logger.info("Validation fraction: %s", validation_frac)
    logger.info("Validation mode: %s", eval_set_name)
    logger.info("Train samples: %d | Validation samples: %d", len(X_train), len(X_val))
    for split_name, split_frame in resolved_split.named_splits().items():
        _log_resolved_split(
            logger,
            split_name,
            rows=len(split_frame.X),
            games=split_frame.n_games,
            coverage=split_frame.competition_seasons,
        )
        split_frame.competition_seasons.to_csv(
            results_path / f"coverage_{split_name}_{run_id}.csv",
            index=False,
        )

    model.fit(X_train, y_train, **fit_kwargs)
    y_proba_val = get_positive_class_scores(
        model,
        X_val.reindex(columns=train_feature_cols),
    )

    # Save trained model — filename encodes run_id, seed, and best ES metric value
    es_metric_name = early_stopping_metric or "best"
    es_metric_value = getattr(model, "best_score", None)
    metric_tag = f"_{es_metric_name}={es_metric_value:.4f}" if es_metric_value is not None else ""
    model_filename = f"xgboost_{target_col}_{run_id}_seed{random_state}{metric_tag}.pkl"
    model_path = models_path / model_filename
    save_model(model, model_path)
    logger.info("Model saved to: %s", model_path)
    
    evaluation_frames: list[tuple[str, str, pd.DataFrame, pd.Series]] = [
        ("train", "in-sample", X_train.reindex(columns=train_feature_cols), y_train),
        ("validation", "in-sample", X_val.reindex(columns=train_feature_cols), y_val),
    ]
    for test_name, split_frame in resolved_split.iter_test_splits():
        evaluation_frames.append(
            (
                f"test_{test_name}",
                test_name.replace("_", "-"),
                split_frame.X.reindex(columns=train_feature_cols),
                split_frame.y.astype(np.uint8),
            )
        )
    target_split = resolved_split.target
    if not target_split.X.empty:
        evaluation_frames.append(
            (
                "target",
                "residual-target",
                target_split.X.reindex(columns=train_feature_cols),
                target_split.y.astype(np.uint8),
            )
        )

    for split_name, split_frame in resolved_split.named_splits(include_lazy=True).items():
        if split_name in {"source_train", "source_val"}:
            continue
        _log_resolved_split(
            logger,
            split_name,
            rows=len(split_frame.X),
            games=split_frame.n_games,
            coverage=split_frame.competition_seasons,
        )
        split_frame.competition_seasons.to_csv(
            results_path / f"coverage_{split_name}_{run_id}.csv",
            index=False,
        )

    selected_threshold = pred_threshold
    threshold_sweep_df: pd.DataFrame | None = None
    if enable_threshold_sweep:
        threshold_sweep_df, selected_threshold = sweep_thresholds_for_f1(
            y_true=y_val,
            y_score=y_proba_val,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_steps=threshold_steps,
        )
        threshold_sweep_df.to_csv(
            results_path / f"threshold_sweep_xgboost_{target_col}_{run_id}.csv",
            index=False,
        )
        logger.info("Selected threshold from validation F1 sweep: %.4f", selected_threshold)

    results_rows: list[dict[str, Any]] = []
    for split_name, description, X_eval, y_eval in evaluation_frames:
        y_proba = get_positive_class_scores(model, X_eval)
        metrics = evaluate_binary_with_baselines(y_proba, y_eval, threshold=selected_threshold)
        print_metrics(f"{split_name.upper()} ({description})", metrics)
        results_rows.append({
            "split": split_name,
            "description": description,
            "league_season": split_name,
            **metrics,
        })
        cm = confusion_matrix(y_eval, (y_proba >= selected_threshold).astype(int))
        plot_confusion_matrix(cm, split_name, results_path / f"confusion_matrix_{split_name}_{target_col}_{run_id}.png")

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_path / f"metrics_xgboost_positive-focus_{target_col}_{run_id}.csv", index=False)

    config_df = pd.DataFrame(
        {
            "model": ["xgboost"],
            "target": [target_col],
            "objective": [effective_model_config.get("objective")],
            "eval_metric": [str(effective_model_config.get("eval_metric"))],
            "selection_objective": ["in_domain_source_validation_only"],
            "source_competitions": [str(resolved_split.source_train.competitions)],
            "calib_competitions": [str(resolved_split.calib.competitions)],
            "test_competitions": [str(sorted({comp for _, split in resolved_split.iter_test_splits() for comp in split.competitions}))],
            "target_competitions": [str(target_split.competitions)],
            "validation_frac": [validation_frac],
            "data_file": [str(data_file)],
            "h5_key_candidates": [str(key_candidates)],
            "validation_set_mode": [eval_set_name],
            "random_state": [random_state],
            "pred_threshold": [pred_threshold],
            "selected_threshold": [selected_threshold],
            "threshold_sweep_enabled": [enable_threshold_sweep],
            "threshold_grid": [f"{threshold_min}:{threshold_max}:{threshold_steps}"],
            "model_path": [str(model_path)],
            "split_config": [str(split_cfg)],
            "xgb_model_config": [str(drop_none_params(effective_model_config))],
            "xgb_fit_config": [str(drop_none_params(effective_fit_config))],
        }
    )
    config_df.to_csv(results_path / f"config_xgboost_{target_col}_{run_id}.csv", index=False)

    # FOR INTERACTIVE DEBUGGING/RESULTS EXPLORATION
    # Filter to rows where model predicted goal (probability = 1)
    # goals_mask = y_proba_target >= selected_threshold
    # X_target_goals = X_target[goals_mask]
    
    # # Keep only columns with variance (exclude constant columns)
    # relevant_cols = [col for col in X_target_goals.columns if X_target_goals[col].nunique() > 1]
    # X_target_goals[relevant_cols].to_csv(results_path / f"predicted_goals_{target_col}.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
