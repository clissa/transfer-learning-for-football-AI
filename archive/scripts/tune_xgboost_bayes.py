from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from football_ai.utils import list_available_dataset_keys

try:
    import optuna
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "optuna is required for Bayesian tuning. Install with: pip install optuna"
    ) from exc


# =========================
# Global config
# =========================
DATA_DIR = Path("data/spadl_data")
TARGET_COL = "scores"  # or "concedes"

RESULTS_BASE_PATH = Path("results/xgboost_bayes_tuning")

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

TRAIN_SEASON_SUFFIXES: list[str] | None = None
TEST_SEASON_SUFFIXES: list[str] | None = None

VAL_PCT = 0.20
RANDOM_STATE = 20260304

# Fixed threshold requested by user
FIXED_THRESHOLD = 0.9026

# Bayesian search budget
N_TRIALS = 100
TIMEOUT_SECONDS: int | None = None

# Search objectives (loss)
OBJECTIVE_CHOICES = [
    "binary:logistic",
    "binary:hinge",
    "reg:logistic",
]

# Eval metrics to pass to XGBoost
EVAL_METRIC_CHOICES = [
    "aucpr",
    "logloss",
    "auc",
]

# scale_pos_weight candidates requested by user
SCALE_POS_WEIGHT_CHOICES: list[float | None] = [None, 10, 50, 75, 100, 200]

# Static model params (not tuned in this script)
BASE_XGB_PARAMS: dict[str, Any] = {
    "booster": "gbtree",
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "validate_parameters": True,
    "importance_type": "gain",
    "early_stopping_rounds": 80,
}
# =========================


@dataclass
class DataBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_keys: list[str]
    test_keys: list[str]


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


def load_xy_split(
    dataset_keys: list[str],
    target_col: str,
    data_dir: Path,
    val_pct: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be 'scores' or 'concedes'")
    if not dataset_keys:
        raise ValueError("dataset_keys is empty")
    if not (0 <= val_pct < 1):
        raise ValueError("val_pct must be in [0, 1)")

    frames: list[pd.DataFrame] = []
    for dataset_key in dataset_keys:
        part = _read_dataset(dataset_key, data_dir).copy()
        part["__dataset_key"] = dataset_key
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    feature_cols = [
        c
        for c in df.columns
        if c not in {"game_id", "action_id", "scores", "concedes", "__dataset_key"}
    ]

    split_game_key = df["__dataset_key"].astype(str) + "::" + df["game_id"].astype(str)
    unique_games = np.array(sorted(split_game_key.unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_games)

    n_val = int(round(len(unique_games) * val_pct))
    val_games = set(unique_games[:n_val])
    is_val = split_game_key.isin(val_games)

    df_train = df.loc[~is_val]
    df_val = df.loc[is_val]

    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y_train = df_train[target_col].astype(np.uint8)
    X_val = df_val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y_val = df_val[target_col].astype(np.uint8)
    return X_train, y_train, X_val, y_val


def load_xy_all(
    dataset_keys: list[str],
    target_col: str,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    if not dataset_keys:
        raise ValueError("dataset_keys is empty")
    frames = [_read_dataset(k, data_dir) for k in dataset_keys]
    df = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in df.columns if c not in {"game_id", "action_id", "scores", "concedes"}]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = df[target_col].astype(np.uint8)
    return X, y


def get_scores(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Return scores for thresholding.
    For objectives with proba output use predict_proba[:,1], otherwise fallback to predict.
    """
    try:
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    except Exception:
        pass
    return model.predict(X).astype(float)


def evaluate_at_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "rows": float(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def sample_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    objective = trial.suggest_categorical("objective", OBJECTIVE_CHOICES)
    eval_metric = trial.suggest_categorical("eval_metric", EVAL_METRIC_CHOICES)
    scale_pos_weight = trial.suggest_categorical("scale_pos_weight", SCALE_POS_WEIGHT_CHOICES)

    params = {
        **BASE_XGB_PARAMS,
        "objective": objective,
        "eval_metric": eval_metric,
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "scale_pos_weight": scale_pos_weight,
    }
    return _drop_none(params)


def build_data_bundle() -> DataBundle:
    available = list_available_dataset_keys(DATA_DIR)
    train_keys = resolve_dataset_keys(available, TRAIN_COMPETITION_PREFIXES, TRAIN_SEASON_SUFFIXES)
    test_keys = resolve_dataset_keys(available, TEST_COMPETITION_PREFIXES, TEST_SEASON_SUFFIXES)
    if not train_keys:
        raise ValueError("No train datasets resolved")
    if not test_keys:
        raise ValueError("No test datasets resolved")

    X_train, y_train, X_val, y_val = load_xy_split(
        dataset_keys=train_keys,
        target_col=TARGET_COL,
        data_dir=DATA_DIR,
        val_pct=VAL_PCT,
        random_state=RANDOM_STATE,
    )
    X_test, y_test = load_xy_all(
        dataset_keys=test_keys,
        target_col=TARGET_COL,
        data_dir=DATA_DIR,
    )
    X_test = X_test.reindex(columns=list(X_train.columns), fill_value=0).astype(np.float32)
    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val.reindex(columns=list(X_train.columns), fill_value=0).astype(np.float32),
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_keys=train_keys,
        test_keys=test_keys,
    )


def main() -> int:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR.resolve()}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE_PATH / f"run_{run_id}_{TARGET_COL}"
    run_dir.mkdir(parents=True, exist_ok=False)

    data = build_data_bundle()
    print(f"Train rows: {len(data.X_train)} | Val rows: {len(data.X_val)} | Test rows: {len(data.X_test)}")
    print(f"Train keys ({len(data.train_keys)}): {data.train_keys}")
    print(f"Test keys ({len(data.test_keys)}): {data.test_keys}")
    print(f"Fixed threshold: {FIXED_THRESHOLD}")
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
        val_metrics = evaluate_at_threshold(data.y_val, y_val_score, FIXED_THRESHOLD)

        trial.set_user_attr("val_precision", val_metrics["precision"])
        trial.set_user_attr("val_recall", val_metrics["recall"])
        trial.set_user_attr("val_pr_auc", val_metrics["pr_auc"])
        trial.set_user_attr("val_roc_auc", val_metrics["roc_auc"])
        return float(val_metrics["f1"])

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

    train_metrics = evaluate_at_threshold(data.y_train, y_train_score, FIXED_THRESHOLD)
    val_metrics = evaluate_at_threshold(data.y_val, y_val_score, FIXED_THRESHOLD)
    test_metrics = evaluate_at_threshold(data.y_test, y_test_score, FIXED_THRESHOLD)

    metrics_df = pd.DataFrame(
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
    metrics_df.to_csv(run_dir / "best_metrics.csv", index=False)

    with (run_dir / "best_params.json").open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2, sort_keys=True)

    run_config = {
        "target_col": TARGET_COL,
        "fixed_threshold": FIXED_THRESHOLD,
        "n_trials": N_TRIALS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "val_pct": VAL_PCT,
        "random_state": RANDOM_STATE,
        "train_competition_prefixes": TRAIN_COMPETITION_PREFIXES,
        "test_competition_prefixes": TEST_COMPETITION_PREFIXES,
        "train_season_suffixes": TRAIN_SEASON_SUFFIXES,
        "test_season_suffixes": TEST_SEASON_SUFFIXES,
        "scale_pos_weight_choices": SCALE_POS_WEIGHT_CHOICES,
        "objective_choices": OBJECTIVE_CHOICES,
        "eval_metric_choices": EVAL_METRIC_CHOICES,
        "train_dataset_keys": data.train_keys,
        "test_dataset_keys": data.test_keys,
        "best_trial_number": study.best_trial.number,
        "best_val_f1": study.best_value,
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as fp:
        json.dump(run_config, fp, indent=2, sort_keys=True)

    print("\nBest trial:", study.best_trial.number)
    print("Best validation F1:", study.best_value)
    print("Results saved in:", run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
