from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_ai.utils import make_dataset_key

# =========================
# Global config
# =========================
DATA_DIR = Path("data/spadl_data")
TARGET_COL = "scores"  # or "concedes"

RESULTS_PATH = Path("results/debug_train")
RESULTS_PATH.mkdir(exist_ok=True)

TRAIN_LEAGUE = "La Liga"
TRAIN_SEASON = "2015/2016"

TEST_LEAGUE = "Champions League"
TEST_SEASON = "2015/2016"

VAL_PCT = 0.20
RANDOM_STATE = 20260304

# Model choices: "rf", "logreg", "mlp"
# MODEL_NAME = "rf"
MODEL_NAME = "logreg"
# MODEL_NAME = "mlp"
MODEL_CONFIG: dict = {
    "rf": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
        "criterion": "log_loss",
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
        # "loss": "log_loss",
        "random_state": RANDOM_STATE,
    },
}
# =========================


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


def load_xy(
    dataset_key: str,
    target_col: str,
    data_dir: Path,
    val_pct: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load and split one league-season into train/val based on unique game_id.
    This prevents leakage of actions from the same game across splits.
    """
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be 'scores' or 'concedes'")
    if not (0 <= val_pct < 1):
        raise ValueError("val_pct must be in [0, 1)")

    df = _read_dataset(dataset_key=dataset_key, data_dir=data_dir)

    feature_cols = [
        col
        for col in df.columns
        if col not in {"game_id", "action_id", "scores", "concedes"}
    ]

    game_ids = np.array(sorted(df["game_id"].unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(game_ids)

    n_val_games = int(round(len(game_ids) * val_pct))
    print(f"Total games: {len(game_ids)}, Validation games: {n_val_games}")
    val_game_ids = set(game_ids[:n_val_games])

    is_val = df["game_id"].isin(val_game_ids)
    df_train = df.loc[~is_val].copy()
    df_val = df.loc[is_val].copy()

    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = df_train[target_col].astype(int)

    X_val = df_val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = df_val[target_col].astype(int)

    return X_train, y_train, X_val, y_val


def load_xy_all(
    dataset_key: str,
    target_col: str,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load all actions from one dataset key without train/val split."""
    df = _read_dataset(dataset_key=dataset_key, data_dir=data_dir)
    feature_cols = [
        col
        for col in df.columns
        if col not in {"game_id", "action_id", "scores", "concedes"}
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].astype(int)
    return X, y


def build_model(model_name: str):
    """
    Default is Random Forest because VAEP-style tabular features are nonlinear
    and class imbalance is common for scoring labels.
    """
    if model_name == "rf":
        return RandomForestClassifier(**MODEL_CONFIG["rf"])
    if model_name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**MODEL_CONFIG["logreg"])),
            ]
        )
    if model_name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(**MODEL_CONFIG["mlp"])),
            ]
        )
    raise ValueError(f"Unknown MODEL_NAME={model_name!r}. Use one of: rf, logreg, mlp")


def evaluate_binary(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "rows": float(len(y)),
        "positive_rate": float(y.mean()),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)) if y.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "brier": float(brier_score_loss(y, y_proba)),
    }


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:>14}: {v:.6f}")


def main() -> int:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR.resolve()}")

    train_key = make_dataset_key(TRAIN_LEAGUE, TRAIN_SEASON)
    test_key = make_dataset_key(TEST_LEAGUE, TEST_SEASON)

    X_train, y_train, X_val, y_val = load_xy(
        dataset_key=train_key,
        target_col=TARGET_COL,
        data_dir=DATA_DIR,
        val_pct=VAL_PCT,
        random_state=RANDOM_STATE,
    )

    X_test, y_test = load_xy_all(
        dataset_key=test_key,
        target_col=TARGET_COL,
        data_dir=DATA_DIR,
    )

    model = build_model(MODEL_NAME)
    model.fit(X_train, y_train)

    feature_cols = list(X_train.columns)
    X_val = X_val.reindex(columns=feature_cols, fill_value=0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    train_metrics = evaluate_binary(model, X_train, y_train)
    val_metrics = evaluate_binary(model, X_val, y_val)
    test_metrics = evaluate_binary(model, X_test, y_test)

    print("Model:", MODEL_NAME)
    print("Target:", TARGET_COL)
    print("Train dataset:", train_key)
    print("Test dataset:", test_key)
    print("Validation pct (game split):", VAL_PCT)

    _print_metrics("TRAIN", train_metrics)
    _print_metrics("VALIDATION", val_metrics)
    _print_metrics("TEST", test_metrics)

    results_df = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "league_season": [f"{TRAIN_LEAGUE}_{TRAIN_SEASON}", f"{TRAIN_LEAGUE}_{TRAIN_SEASON}", f"{TEST_LEAGUE}_{TEST_SEASON}"],
            **{k: [train_metrics[k], val_metrics[k], test_metrics[k]] for k in train_metrics},
        }
    )
    results_df.to_csv(RESULTS_PATH / f"metrics_{MODEL_NAME}_{TARGET_COL}.csv", index=False)
    config_df = pd.DataFrame(
        {
            "model": [MODEL_NAME],
            "target": [TARGET_COL],
            "train_league_season": [f"{TRAIN_LEAGUE}_{TRAIN_SEASON}"],
            "test_league_season": [f"{TEST_LEAGUE}_{TEST_SEASON}"],
            "val_pct": [VAL_PCT],
            "random_state": [RANDOM_STATE],
            "model_config": [str(MODEL_CONFIG[MODEL_NAME])],
        }
    )
    config_df.to_csv(RESULTS_PATH / f"config_{MODEL_NAME}_{TARGET_COL}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
