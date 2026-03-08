from __future__ import annotations

"""Bayesian tuning for XGBoost on tabular football VAEP-style features.

Changes vs previous script:
- Loads a single HDF5 file (DATA_FILE / DATA_KEY)
- Train/Val/Test split by *competition*:
    * VAL competitions: Serie A + Ligue 1
    * TEST competitions: Champions League + UEFA Europa League
    * TRAIN competitions: all remaining competitions in the file
- Optuna objective: validation PR-AUC (Average Precision) using probabilistic outputs
- Keeps other metrics for comparison (ROC-AUC, logloss, Brier, plus thresholded metrics)
- Removes binary:hinge / reg:logistic: uses binary:logistic only
- Uses large n_estimators with early stopping (n_estimators fixed high)
- Adds max_delta_step search (helpful for extreme imbalance)
- Keeps NaNs (no fillna(0)); replaces +/-inf with NaN

Usage examples
--------------
# Using YAML config
python -m scripts.tune_xgboost_bayes_v2 --config configs/tune_xgboost.yaml

# Override number of trials from CLI
python -m scripts.tune_xgboost_bayes_v2 --config configs/tune_xgboost.yaml --n-trials 50

# Legacy: runs with hardcoded module-level defaults (no --config needed)
python -m scripts.tune_xgboost_bayes_v2
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from football_ai.config import load_config, resolve_random_state
from football_ai.data import read_h5_table
from football_ai.evaluation import (
    evaluate_binary,
    get_positive_class_scores,
    sweep_thresholds_for_f1,
)
from football_ai.training import drop_none_params

try:
    import optuna
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "optuna is required for Bayesian tuning. Install with: pip install optuna"
    ) from exc


# =========================
# Global config
# =========================
DATA_FILE = Path("data/spadl_data_rich/major_leagues.h5")
DATA_KEY_CANDIDATES = ["rich_action", "rich_actions"]

TARGET_COL = "scores"  # or "concedes"

RESULTS_BASE_PATH = Path("results/xgboost_bayes_tuning")

# Competition split (names are matched after normalization)
VAL_COMPETITION_NAMES = ["Serie A", "Ligue 1"]
TEST_COMPETITION_NAMES = ["Champions League", "UEFA Europa League"]

# Optional: keep a fixed threshold for comparison only.
# The model's operating threshold will be chosen on validation by maximizing F1.
FIXED_THRESHOLD: float | None = 0.9026

RANDOM_STATE: int | None = 20260305

# Bayesian search budget
N_TRIALS = 300
TIMEOUT_SECONDS: int | None = None

# scale_pos_weight candidates (you can convert this to a log-uniform float later)
SCALE_POS_WEIGHT_CHOICES: list[float | None] = [None, 10, 50, 75, 100, 200]

# Fixed training params
N_ESTIMATORS = 20000
EARLY_STOPPING_ROUNDS = 200

BASE_XGB_PARAMS: dict[str, Any] = {
    "booster": "gbtree",
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "validate_parameters": True,
    "importance_type": "gain",
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": N_ESTIMATORS,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}

# Requested feature space for rich data preprocessing.
NUMERIC_FEATURES = [
    "time_seconds",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "home_score",
    "away_score",
    "minutes_played",
]
BOOLEAN_FEATURES = ["is_starter"]
CATEGORICAL_FEATURES = [
    "player_id",
    "type_id",
    "result_id",
    "bodypart_id",
    "competition_stage",
    "game_day",
    "venue",
    "referee",
    "starting_position_id",
]
DERIVED_BOOLEAN_FEATURES = ["is_home_team"]
# =========================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bayesian tuning for XGBoost on VAEP-style features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file (e.g. configs/tune_xgboost.yaml)",
    )
    parser.add_argument("--target-col", type=str, default=None, help="Target column: scores or concedes")
    parser.add_argument("--data-file", type=str, default=None, help="Path to HDF5 data file")
    parser.add_argument("--output-dir", type=str, default=None, help="Base directory for run outputs")
    parser.add_argument("--n-trials", type=int, default=None, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random state")
    parser.add_argument("--device", type=str, default=None, help="XGBoost device: cpu or cuda")
    return parser.parse_args()


@dataclass
class DataBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_competitions: list[str]
    val_competitions: list[str]
    test_competitions: list[str]
    competition_col: str
    season_col: str | None
    info_train: pd.DataFrame
    info_val: pd.DataFrame
    info_test: pd.DataFrame
    feature_cols: list[str]
    categorical_feature_cols: list[str]
    dropped_info_cols: list[str]
    data_key_used: str





def _normalize_name(s: str) -> str:
    # Keep only [a-z0-9] in lowercase to make matching robust.
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _first_existing(colnames: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(colnames)
    for c in candidates:
        if c in cols:
            return c
    return None





def _detect_competition_and_season_cols(df: pd.DataFrame) -> tuple[str, str | None]:
    # Try the most common naming conventions.
    comp_col = _first_existing(
        df.columns,
        [
            "competition_name",
            # "competition",
            # "comp_name",
            # "league",
            # "league_name",
        ],
    )
    if comp_col is None:
        raise ValueError(
            "Could not find a competition column. Expected one of: "
            "competition_name/competition/league/...\n"
            f"Available columns: {list(df.columns)[:60]}{'...' if len(df.columns) > 60 else ''}"
        )

    season_col = _first_existing(
        df.columns,
        [
            "season_name",
            # "season",
            # "season_id",
            # "year",
            # "season_year",
        ],
    )
    # season_col is optional for splitting; used only for logging.
    return comp_col, season_col


def _prepare_features(
    df: pd.DataFrame,
    target_col: str,
    competition_col: str,
    season_col: str | None,
    numeric_features: list[str] | None = None,
    boolean_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    derived_boolean_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], list[str], list[str]]:
    num_feats = numeric_features or NUMERIC_FEATURES
    bool_feats = boolean_features or BOOLEAN_FEATURES
    cat_feats = categorical_features or CATEGORICAL_FEATURES
    derived_bool_feats = derived_boolean_features or DERIVED_BOOLEAN_FEATURES

    required_cols = {
        "team_id",
        "home_team_id",
        "away_team_id",
        "game_id",
        "action_id",
        competition_col,
        "scores",
        "concedes",
        *num_feats,
        *bool_feats,
        *cat_feats,
    }
    missing = sorted(c for c in required_cols if c not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns in rich dataset: {missing}")

    work = df.copy()
    work["is_home_team"] = (work["team_id"] == work["home_team_id"]).astype(np.uint8)

    feature_cols = num_feats + bool_feats + derived_bool_feats + cat_feats
    X = work[feature_cols].copy()

    for c in num_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
    for c in bool_feats + derived_bool_feats:
        X[c] = X[c].fillna(0).astype(np.uint8)
    # Encode categorical variables as integer codes for compatibility with
    # XGBoost builds that don't reliably propagate enable_categorical.
    for c in cat_feats:
        cat = X[c].astype("string").fillna("__MISSING__").astype("category")
        X[c] = cat.cat.codes.astype(np.int32)

    y = work[target_col].astype(np.uint8)

    keep_for_training_or_targets = set(feature_cols) | {"scores", "concedes"}
    dropped_info_cols = [c for c in work.columns if c not in keep_for_training_or_targets]

    info_base = [
        "game_id",
        "action_id",
        competition_col,
        "team_id",
        "home_team_id",
        "away_team_id",
        "scores",
        "concedes",
    ]
    if season_col:
        info_base.append(season_col)
    info_cols = list(dict.fromkeys([c for c in info_base + dropped_info_cols if c in work.columns]))
    info = work[info_cols].copy()

    return X, y, info, feature_cols, cat_feats.copy(), dropped_info_cols





def load_data_bundle(
    data_file: Path | None = None,
    data_key_candidates: list[str] | None = None,
    target_col: str | None = None,
    val_competition_names: list[str] | None = None,
    test_competition_names: list[str] | None = None,
    numeric_features: list[str] | None = None,
    boolean_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    derived_boolean_features: list[str] | None = None,
) -> DataBundle:
    eff_data_file = data_file or DATA_FILE
    eff_key_candidates = data_key_candidates or DATA_KEY_CANDIDATES
    eff_target_col = target_col or TARGET_COL
    eff_val_names = val_competition_names or VAL_COMPETITION_NAMES
    eff_test_names = test_competition_names or TEST_COMPETITION_NAMES
    eff_numeric = numeric_features or NUMERIC_FEATURES
    eff_boolean = boolean_features or BOOLEAN_FEATURES
    eff_categorical = categorical_features or CATEGORICAL_FEATURES
    eff_derived_boolean = derived_boolean_features or DERIVED_BOOLEAN_FEATURES

    if not eff_data_file.exists():
        raise FileNotFoundError(f"DATA_FILE does not exist: {eff_data_file.resolve()}")

    df = read_h5_table(eff_data_file, eff_key_candidates)
    # Probe which key was actually used (for logging/metadata only).
    with pd.HDFStore(str(eff_data_file), mode="r") as store:
        available = set(store.keys())
        selected_data_key = next(
            (k for k in eff_key_candidates if f"/{k}" in available), eff_key_candidates[0]
        )
    if eff_target_col not in df.columns:
        raise ValueError(
            f"TARGET_COL='{eff_target_col}' not found in data. "
            f"Available columns include: {list(df.columns)[:40]}{'...' if len(df.columns) > 40 else ''}"
        )

    comp_col, season_col = _detect_competition_and_season_cols(df)

    # Normalize competition names for robust matching
    df["__comp_norm"] = df[comp_col].map(_normalize_name)
    val_norm = {_normalize_name(x) for x in eff_val_names}
    test_norm = {_normalize_name(x) for x in eff_test_names}

    unique_comp_raw = sorted(df[comp_col].dropna().unique().tolist())
    unique_comp_norm = sorted(df["__comp_norm"].dropna().unique().tolist())

    # Split masks
    is_val = df["__comp_norm"].isin(val_norm)
    is_test = df["__comp_norm"].isin(test_norm)
    is_train = ~(is_val | is_test)

    # Sanity checks
    if is_train.sum() == 0:
        raise ValueError(
            "Train split is empty. Check VAL_COMPETITION_NAMES / TEST_COMPETITION_NAMES "
            "and competition column detection.\n"
            f"Detected competition column: {comp_col}.\n"
            f"Unique competitions (raw): {unique_comp_raw}"  # usually short
        )
    if is_val.sum() == 0:
        raise ValueError(
            "Validation split is empty (Serie A + Ligue 1 not found).\n"
            f"Detected competition column: {comp_col}.\n"
            f"Unique competitions (raw): {unique_comp_raw}"
        )
    if is_test.sum() == 0:
        raise ValueError(
            "Test split is empty (Champions League / Europa League not found).\n"
            f"Detected competition column: {comp_col}.\n"
            f"Unique competitions (raw): {unique_comp_raw}"
        )

    X_all, y_all, info_all, feature_cols, categorical_feature_cols, dropped_info_cols = _prepare_features(
        df=df,
        target_col=eff_target_col,
        competition_col=comp_col,
        season_col=season_col,
        numeric_features=eff_numeric,
        boolean_features=eff_boolean,
        categorical_features=eff_categorical,
        derived_boolean_features=eff_derived_boolean,
    )

    df_train = df.loc[is_train].copy()
    df_val = df.loc[is_val].copy()
    df_test = df.loc[is_test].copy()

    X_train = X_all.loc[is_train].copy()
    y_train = y_all.loc[is_train].copy()
    X_val = X_all.loc[is_val].copy()
    y_val = y_all.loc[is_val].copy()
    X_test = X_all.loc[is_test].copy()
    y_test = y_all.loc[is_test].copy()
    info_train = info_all.loc[is_train].copy()
    info_val = info_all.loc[is_val].copy()
    info_test = info_all.loc[is_test].copy()

    # Keep original competition names for logging
    train_comps = sorted(df_train[comp_col].dropna().unique().tolist())
    val_comps = sorted(df_val[comp_col].dropna().unique().tolist())
    test_comps = sorted(df_test[comp_col].dropna().unique().tolist())

    # Clean up helper column
    for d in (df_train, df_val, df_test):
        if "__comp_norm" in d.columns:
            d.drop(columns=["__comp_norm"], inplace=True, errors="ignore")

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_competitions=train_comps,
        val_competitions=val_comps,
        test_competitions=test_comps,
        competition_col=comp_col,
        season_col=season_col,
        info_train=info_train,
        info_val=info_val,
        info_test=info_test,
        feature_cols=feature_cols,
        categorical_feature_cols=categorical_feature_cols,
        dropped_info_cols=dropped_info_cols,
        data_key_used=selected_data_key,
    )


def _save_exploration(
    run_dir: Path,
    split_name: str,
    info_df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> None:
    out = info_df.reset_index(drop=True).copy()
    out[f"target_{TARGET_COL}"] = pd.Series(y_true, dtype=np.uint8)
    out["pred_score"] = pd.Series(y_score, dtype=np.float32)
    out["pred_label"] = (out["pred_score"].to_numpy() >= threshold).astype(np.uint8)
    out.to_csv(run_dir / f"exploration_{split_name}.csv.gz", index=False)





def sample_xgb_params(
    trial: optuna.Trial,
    base_params: dict[str, Any] | None = None,
    scale_pos_weight_choices: list[float | None] | None = None,
) -> dict[str, Any]:
    eff_base = base_params or BASE_XGB_PARAMS
    eff_spw_choices = scale_pos_weight_choices or SCALE_POS_WEIGHT_CHOICES
    scale_pos_weight = trial.suggest_categorical("scale_pos_weight", eff_spw_choices)

    params: dict[str, Any] = {
        **eff_base,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "scale_pos_weight": scale_pos_weight,
    }
    return drop_none_params(params)


def main() -> int:
    args = parse_args()

    # Build effective config: module defaults -> YAML -> CLI overrides
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_config(args.config)

    # --- Resolve settings ---
    data_file = Path(args.data_file or cfg.get("data", {}).get("file") or str(DATA_FILE))
    data_key_candidates: list[str] = cfg.get("data", {}).get("key_candidates", DATA_KEY_CANDIDATES)
    target_col: str = args.target_col or cfg.get("data", {}).get("target_col", TARGET_COL)

    split_cfg = cfg.get("split", {})
    val_competition_names: list[str] = split_cfg.get("validation_competitions", VAL_COMPETITION_NAMES)
    test_competition_names: list[str] = split_cfg.get("test_competitions", TEST_COMPETITION_NAMES)

    optuna_cfg = cfg.get("optuna", {})
    n_trials: int = int(args.n_trials or optuna_cfg.get("n_trials", N_TRIALS))
    timeout_seconds: int | None = args.timeout or optuna_cfg.get("timeout_seconds", TIMEOUT_SECONDS)
    random_state: int = resolve_random_state(
        args.seed, optuna_cfg.get("random_state"), RANDOM_STATE,
    )

    training_cfg = cfg.get("training", {})
    n_estimators: int = int(training_cfg.get("n_estimators", N_ESTIMATORS))
    early_stopping_rounds: int = int(training_cfg.get("early_stopping_rounds", EARLY_STOPPING_ROUNDS))

    # Build effective base XGB params
    base_xgb_params = dict(BASE_XGB_PARAMS)
    yaml_base = training_cfg.get("base_params", {})
    base_xgb_params.update({k: v for k, v in yaml_base.items() if v is not None})
    base_xgb_params["random_state"] = random_state
    base_xgb_params["n_estimators"] = n_estimators
    base_xgb_params["early_stopping_rounds"] = early_stopping_rounds
    if args.device is not None:
        base_xgb_params["device"] = args.device

    scale_pos_weight_choices: list[float | None] = training_cfg.get(
        "scale_pos_weight_choices", SCALE_POS_WEIGHT_CHOICES
    )

    fixed_threshold: float | None = cfg.get("threshold", {}).get("fixed", FIXED_THRESHOLD)

    # Resolve feature lists from YAML (fallback to module-level constants)
    feat_cfg = cfg.get("features", {})
    numeric_features = feat_cfg.get("numeric", None)
    boolean_features = feat_cfg.get("boolean", None)
    categorical_features = feat_cfg.get("categorical", None)
    derived_boolean_features = feat_cfg.get("derived_boolean", None)

    results_base_path = Path(args.output_dir or cfg.get("output", {}).get("dir") or str(RESULTS_BASE_PATH))

    # ---- Run ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_base_path / f"run_{run_id}_{target_col}_val_leagues"
    run_dir.mkdir(parents=True, exist_ok=False)

    data = load_data_bundle(
        data_file=data_file,
        data_key_candidates=data_key_candidates,
        target_col=target_col,
        val_competition_names=val_competition_names,
        test_competition_names=test_competition_names,
        numeric_features=numeric_features,
        boolean_features=boolean_features,
        categorical_features=categorical_features,
        derived_boolean_features=derived_boolean_features,
    )

    print(
        f"Train rows: {len(data.X_train)} | Val rows: {len(data.X_val)} | Test rows: {len(data.X_test)}"
    )
    print(f"Train competitions ({len(data.train_competitions)}): {data.train_competitions}")
    print(f"Val competitions ({len(data.val_competitions)}): {data.val_competitions}")
    print(f"Test competitions ({len(data.test_competitions)}): {data.test_competitions}")
    print(f"Using data key: {data.data_key_used}")
    print(f"Training features ({len(data.feature_cols)}): {data.feature_cols}")
    print(f"Target: {target_col} | Optuna objective: val PR-AUC")
    if fixed_threshold is not None:
        print(f"Fixed threshold (comparison only): {fixed_threshold}")
    print(f"Saving run outputs to: {run_dir.resolve()}")

    def objective(trial: optuna.Trial) -> float:
        params = sample_xgb_params(trial, base_params=base_xgb_params, scale_pos_weight_choices=scale_pos_weight_choices)
        model = XGBClassifier(**params)
        model.fit(
            data.X_train,
            data.y_train,
            eval_set=[(data.X_val, data.y_val)],
            verbose=False,
        )
        y_val_score = get_positive_class_scores(model, data.X_val)

        rank = evaluate_binary(y_val_score, data.y_val.values)
        sweep_df, best_thr = sweep_thresholds_for_f1(data.y_val, y_val_score)
        best_row = sweep_df.loc[sweep_df["threshold"] == best_thr].iloc[0]

        trial.set_user_attr("val_pr_auc", rank["pr_auc"])
        trial.set_user_attr("val_roc_auc", rank["roc_auc"])
        trial.set_user_attr("val_logloss", rank["logloss"])
        trial.set_user_attr("val_brier", rank["brier"])
        trial.set_user_attr("val_best_f1", float(best_row["f1"]))
        trial.set_user_attr("val_best_threshold", best_thr)
        trial.set_user_attr("val_best_precision", float(best_row["precision"]))
        trial.set_user_attr("val_best_recall", float(best_row["recall"]))
        trial.set_user_attr("val_pred_pos_rate_at_best", float(best_row["pred_positive_rate"]))

        if fixed_threshold is not None:
            at_fixed = evaluate_binary(y_val_score, data.y_val.values, threshold=fixed_threshold)
            trial.set_user_attr("val_f1_fixed", at_fixed["f1"])
            trial.set_user_attr("val_precision_fixed", at_fixed["precision"])
            trial.set_user_attr("val_recall_fixed", at_fixed["recall"])
            trial.set_user_attr("val_pred_pos_rate_fixed", float(np.mean(y_val_score >= fixed_threshold)))

        # Optimize for validation PR-AUC (Average Precision)
        return float(rank["pr_auc"])

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        study_name=f"xgb_bayes_{target_col}_{run_id}",
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, n_jobs=1)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_df = trials_df.sort_values("value", ascending=False).reset_index(drop=True)
    trials_df.to_csv(run_dir / "trials.csv", index=False)

    best_params = drop_none_params({**base_xgb_params, **study.best_trial.params})
    best_model = XGBClassifier(**best_params)
    best_model.fit(
        data.X_train,
        data.y_train,
        eval_set=[(data.X_val, data.y_val)],
        verbose=False,
    )

    y_train_score = get_positive_class_scores(best_model, data.X_train)
    y_val_score = get_positive_class_scores(best_model, data.X_val)
    y_test_score = get_positive_class_scores(best_model, data.X_test)

    # Choose operating threshold on validation (maximize F1)
    sweep_df, val_best_thr = sweep_thresholds_for_f1(data.y_val, y_val_score)
    val_best_row = sweep_df.loc[sweep_df["threshold"] == val_best_thr].iloc[0]
    val_best_f1 = float(val_best_row["f1"])

    _RANKING_KEYS = ("rows", "positive_rate", "pr_auc", "brier", "roc_auc", "logloss")

    def pack_metrics(split_name: str, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
        metrics = evaluate_binary(y_score, y_true)
        out: dict[str, Any] = {"split": split_name, **{k: metrics[k] for k in _RANKING_KEYS}}

        at_best = evaluate_binary(y_score, y_true, threshold=val_best_thr)
        out["op_threshold"] = val_best_thr
        out["op_pred_positive_rate"] = float(np.mean(y_score >= val_best_thr))
        out["op_precision"] = at_best["precision"]
        out["op_recall"] = at_best["recall"]
        out["op_f1"] = at_best["f1"]
        out["op_threshold_source"] = "val_max_f1"
        out["val_best_f1"] = float(val_best_f1)

        if fixed_threshold is not None:
            at_fixed = evaluate_binary(y_score, y_true, threshold=fixed_threshold)
            out["fixed_threshold"] = fixed_threshold
            out["fixed_pred_positive_rate"] = float(np.mean(y_score >= fixed_threshold))
            out["fixed_precision"] = at_fixed["precision"]
            out["fixed_recall"] = at_fixed["recall"]
            out["fixed_f1"] = at_fixed["f1"]
        return out

    metrics_rows = [
        pack_metrics("train", data.y_train.values, y_train_score),
        pack_metrics("validation", data.y_val.values, y_val_score),
        pack_metrics("test", data.y_test.values, y_test_score),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "best_metrics.csv", index=False)

    with (run_dir / "best_params.json").open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2, sort_keys=True)

    # _save_exploration(run_dir, "train", data.info_train, data.y_train.values, y_train_score, val_best_thr)
    # _save_exploration(run_dir, "validation", data.info_val, data.y_val.values, y_val_score, val_best_thr)
    # _save_exploration(run_dir, "test", data.info_test, data.y_test.values, y_test_score, val_best_thr)

    run_config = {
        "data_file": str(data_file),
        "data_key_candidates": data_key_candidates,
        "data_key_used": data.data_key_used,
        "target_col": target_col,
        "optuna_objective": "val_pr_auc",
        "fixed_threshold": fixed_threshold,
        "n_trials": n_trials,
        "timeout_seconds": timeout_seconds,
        "random_state": random_state,
        "train_competition_names": None,  # derived as "all others"
        "val_competition_names": val_competition_names,
        "test_competition_names": test_competition_names,
        "resolved_train_competitions": data.train_competitions,
        "resolved_val_competitions": data.val_competitions,
        "resolved_test_competitions": data.test_competitions,
        "competition_col": data.competition_col,
        "season_col": data.season_col,
        "scale_pos_weight_choices": scale_pos_weight_choices,
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stopping_rounds,
        "base_xgb_params": base_xgb_params,
        "feature_cols": data.feature_cols,
        "categorical_feature_cols": data.categorical_feature_cols,
        "dropped_info_cols": data.dropped_info_cols,
        "best_trial_number": study.best_trial.number,
        "best_val_pr_auc": float(study.best_value),
        "best_val_threshold_max_f1": float(val_best_thr),
        "best_val_f1_max_f1": float(val_best_f1),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as fp:
        json.dump(run_config, fp, indent=2, sort_keys=True)

    print("\nBest trial:", study.best_trial.number)
    print("Best validation PR-AUC:", float(study.best_value))
    print("Operating threshold (val max-F1):", float(val_best_thr))
    print("Validation max-F1:", float(val_best_f1))
    print("Results saved in:", run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
