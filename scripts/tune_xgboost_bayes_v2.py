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
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

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

RANDOM_STATE = 20260305

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


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _normalize_name(s: str) -> str:
    # Keep only [a-z0-9] in lowercase to make matching robust.
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _first_existing(colnames: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(colnames)
    for c in candidates:
        if c in cols:
            return c
    return None


def _read_rich_actions(data_file: Path, key_candidates: list[str]) -> tuple[pd.DataFrame, str]:
    with pd.HDFStore(str(data_file), mode="r") as store:
        available_keys = set(store.keys())
        selected_key = None
        for key in key_candidates:
            if f"/{key}" in available_keys:
                selected_key = key
                break
        if selected_key is None:
            raise ValueError(
                f"None of keys {key_candidates} found in {data_file}. "
                f"Available keys: {sorted(available_keys)}"
            )
        return store.get(f"/{selected_key}"), selected_key


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
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], list[str], list[str]]:
    required_cols = {
        "team_id",
        "home_team_id",
        "away_team_id",
        "game_id",
        "action_id",
        competition_col,
        "scores",
        "concedes",
        *NUMERIC_FEATURES,
        *BOOLEAN_FEATURES,
        *CATEGORICAL_FEATURES,
    }
    missing = sorted(c for c in required_cols if c not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns in rich dataset: {missing}")

    work = df.copy()
    work["is_home_team"] = (work["team_id"] == work["home_team_id"]).astype(np.uint8)

    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + DERIVED_BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    X = work[feature_cols].copy()

    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
    for c in BOOLEAN_FEATURES + DERIVED_BOOLEAN_FEATURES:
        X[c] = X[c].fillna(0).astype(np.uint8)
    # Encode categorical variables as integer codes for compatibility with
    # XGBoost builds that don't reliably propagate enable_categorical.
    for c in CATEGORICAL_FEATURES:
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

    return X, y, info, feature_cols, CATEGORICAL_FEATURES.copy(), dropped_info_cols


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float, float]:
    """Return (best_f1, best_threshold, precision_at_best, recall_at_best)."""
    p, r, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-12)
    if len(thr) == 0:
        # Degenerate case: all scores identical
        return 0.0, 0.5, float(p[-1]), float(r[-1])
    # thr has length len(p)-1
    best_i = int(np.nanargmax(f1[:-1]))
    return float(f1[best_i]), float(thr[best_i]), float(p[best_i]), float(r[best_i])


def _eval_ranking(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {
        "rows": float(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["logloss"] = float(log_loss(y_true, y_score, labels=[0, 1]))
    else:
        out["roc_auc"] = float("nan")
        out["logloss"] = float("nan")
    return out


def _eval_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict[str, float]:
    y_pred = (y_score >= thr).astype(int)
    return {
        "threshold": float(thr),
        "pred_positive_rate": float(np.mean(y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def load_data_bundle() -> DataBundle:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"DATA_FILE does not exist: {DATA_FILE.resolve()}")

    df, selected_data_key = _read_rich_actions(DATA_FILE, DATA_KEY_CANDIDATES)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TARGET_COL='{TARGET_COL}' not found in data. "
            f"Available columns include: {list(df.columns)[:40]}{'...' if len(df.columns) > 40 else ''}"
        )

    comp_col, season_col = _detect_competition_and_season_cols(df)

    # Normalize competition names for robust matching
    df["__comp_norm"] = df[comp_col].map(_normalize_name)
    val_norm = {_normalize_name(x) for x in VAL_COMPETITION_NAMES}
    test_norm = {_normalize_name(x) for x in TEST_COMPETITION_NAMES}

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
        target_col=TARGET_COL,
        competition_col=comp_col,
        season_col=season_col,
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


def get_scores(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    raise RuntimeError("predict_proba did not return a (n,2) array; check objective.")


def sample_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    scale_pos_weight = trial.suggest_categorical("scale_pos_weight", SCALE_POS_WEIGHT_CHOICES)

    params: dict[str, Any] = {
        **BASE_XGB_PARAMS,
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
    return _drop_none(params)


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE_PATH / f"run_{run_id}_{TARGET_COL}_val_leagues"
    run_dir.mkdir(parents=True, exist_ok=False)

    data = load_data_bundle()

    print(
        f"Train rows: {len(data.X_train)} | Val rows: {len(data.X_val)} | Test rows: {len(data.X_test)}"
    )
    print(f"Train competitions ({len(data.train_competitions)}): {data.train_competitions}")
    print(f"Val competitions ({len(data.val_competitions)}): {data.val_competitions}")
    print(f"Test competitions ({len(data.test_competitions)}): {data.test_competitions}")
    print(f"Using data key: {data.data_key_used}")
    print(f"Training features ({len(data.feature_cols)}): {data.feature_cols}")
    print(f"Target: {TARGET_COL} | Optuna objective: val PR-AUC")
    if FIXED_THRESHOLD is not None:
        print(f"Fixed threshold (comparison only): {FIXED_THRESHOLD}")
    print(f"Saving run outputs to: {run_dir.resolve()}")

    def objective(trial: optuna.Trial) -> float:
        params = sample_xgb_params(trial)
        model = XGBClassifier(**params)
        model.fit(
            data.X_train,
            data.y_train,
            eval_set=[(data.X_val, data.y_val)],
            verbose=False,
        )
        y_val_score = get_scores(model, data.X_val)

        rank = _eval_ranking(data.y_val.values, y_val_score)
        best_f1, best_thr, best_p, best_r = _best_f1_threshold(data.y_val.values, y_val_score)
        at_best = _eval_at_threshold(data.y_val.values, y_val_score, best_thr)

        trial.set_user_attr("val_pr_auc", rank["pr_auc"])
        trial.set_user_attr("val_roc_auc", rank["roc_auc"])
        trial.set_user_attr("val_logloss", rank["logloss"])
        trial.set_user_attr("val_brier", rank["brier"])
        trial.set_user_attr("val_best_f1", best_f1)
        trial.set_user_attr("val_best_threshold", best_thr)
        trial.set_user_attr("val_best_precision", best_p)
        trial.set_user_attr("val_best_recall", best_r)
        trial.set_user_attr("val_pred_pos_rate_at_best", at_best["pred_positive_rate"])

        if FIXED_THRESHOLD is not None:
            at_fixed = _eval_at_threshold(data.y_val.values, y_val_score, FIXED_THRESHOLD)
            trial.set_user_attr("val_f1_fixed", at_fixed["f1"])
            trial.set_user_attr("val_precision_fixed", at_fixed["precision"])
            trial.set_user_attr("val_recall_fixed", at_fixed["recall"])
            trial.set_user_attr("val_pred_pos_rate_fixed", at_fixed["pred_positive_rate"])

        # Optimize for validation PR-AUC (Average Precision)
        return float(rank["pr_auc"])

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        study_name=f"xgb_bayes_{TARGET_COL}_{run_id}",
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, n_jobs=1)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_df = trials_df.sort_values("value", ascending=False).reset_index(drop=True)
    trials_df.to_csv(run_dir / "trials.csv", index=False)

    best_params = _drop_none({**BASE_XGB_PARAMS, **study.best_trial.params})
    best_model = XGBClassifier(**best_params)
    best_model.fit(
        data.X_train,
        data.y_train,
        eval_set=[(data.X_val, data.y_val)],
        verbose=False,
    )

    y_train_score = get_scores(best_model, data.X_train)
    y_val_score = get_scores(best_model, data.X_val)
    y_test_score = get_scores(best_model, data.X_test)

    # Choose operating threshold on validation (maximize F1)
    val_best_f1, val_best_thr, _, _ = _best_f1_threshold(data.y_val.values, y_val_score)

    def pack_metrics(split_name: str, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
        rank = _eval_ranking(y_true, y_score)
        out: dict[str, Any] = {"split": split_name, **rank}

        at_best = _eval_at_threshold(y_true, y_score, val_best_thr)
        out.update({f"op_{k}": v for k, v in at_best.items()})
        out["op_threshold_source"] = "val_max_f1"
        out["val_best_f1"] = float(val_best_f1)

        if FIXED_THRESHOLD is not None:
            at_fixed = _eval_at_threshold(y_true, y_score, FIXED_THRESHOLD)
            out.update({f"fixed_{k}": v for k, v in at_fixed.items()})
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
        "data_file": str(DATA_FILE),
        "data_key_candidates": DATA_KEY_CANDIDATES,
        "data_key_used": data.data_key_used,
        "target_col": TARGET_COL,
        "optuna_objective": "val_pr_auc",
        "fixed_threshold": FIXED_THRESHOLD,
        "n_trials": N_TRIALS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "random_state": RANDOM_STATE,
        "train_competition_names": None,  # derived as "all others"
        "val_competition_names": VAL_COMPETITION_NAMES,
        "test_competition_names": TEST_COMPETITION_NAMES,
        "resolved_train_competitions": data.train_competitions,
        "resolved_val_competitions": data.val_competitions,
        "resolved_test_competitions": data.test_competitions,
        "competition_col": data.competition_col,
        "season_col": data.season_col,
        "scale_pos_weight_choices": SCALE_POS_WEIGHT_CHOICES,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "base_xgb_params": BASE_XGB_PARAMS,
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
