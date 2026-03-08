"""Core training utilities for football AI models.

Centralises data loading, model building, evaluation, and training helpers.
Scripts in ``scripts/`` are thin CLI wrappers that call into these functions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from .data import load_dataset_tables, load_xy, read_h5_table, split_dataset_key

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# VAEP feature whitelist
# ──────────────────────────────────────────────
# Curated set of VAEP features for training.  Raw ID columns (type_id,
# result_id, bodypart_id, …) are intentionally excluded because they are
# redundant with the one-hot encoded versions listed here.

VAEP_FEATURE_COLS: tuple[str, ...] = (
    # ── action-type one-hots (a0, a1, a2) ──
    "actiontype_pass_a0", "actiontype_cross_a0", "actiontype_throw_in_a0",
    "actiontype_freekick_crossed_a0", "actiontype_freekick_short_a0",
    "actiontype_corner_crossed_a0", "actiontype_corner_short_a0",
    "actiontype_take_on_a0", "actiontype_foul_a0", "actiontype_tackle_a0",
    "actiontype_interception_a0", "actiontype_shot_a0",
    "actiontype_shot_penalty_a0", "actiontype_shot_freekick_a0",
    "actiontype_keeper_save_a0", "actiontype_keeper_claim_a0",
    "actiontype_keeper_punch_a0", "actiontype_keeper_pick_up_a0",
    "actiontype_clearance_a0", "actiontype_bad_touch_a0",
    "actiontype_non_action_a0", "actiontype_dribble_a0",
    "actiontype_goalkick_a0",
    "actiontype_pass_a1", "actiontype_cross_a1", "actiontype_throw_in_a1",
    "actiontype_freekick_crossed_a1", "actiontype_freekick_short_a1",
    "actiontype_corner_crossed_a1", "actiontype_corner_short_a1",
    "actiontype_take_on_a1", "actiontype_foul_a1", "actiontype_tackle_a1",
    "actiontype_interception_a1", "actiontype_shot_a1",
    "actiontype_shot_penalty_a1", "actiontype_shot_freekick_a1",
    "actiontype_keeper_save_a1", "actiontype_keeper_claim_a1",
    "actiontype_keeper_punch_a1", "actiontype_keeper_pick_up_a1",
    "actiontype_clearance_a1", "actiontype_bad_touch_a1",
    "actiontype_non_action_a1", "actiontype_dribble_a1",
    "actiontype_goalkick_a1",
    "actiontype_pass_a2", "actiontype_cross_a2", "actiontype_throw_in_a2",
    "actiontype_freekick_crossed_a2", "actiontype_freekick_short_a2",
    "actiontype_corner_crossed_a2", "actiontype_corner_short_a2",
    "actiontype_take_on_a2", "actiontype_foul_a2", "actiontype_tackle_a2",
    "actiontype_interception_a2", "actiontype_shot_a2",
    "actiontype_shot_penalty_a2", "actiontype_shot_freekick_a2",
    "actiontype_keeper_save_a2", "actiontype_keeper_claim_a2",
    "actiontype_keeper_punch_a2", "actiontype_keeper_pick_up_a2",
    "actiontype_clearance_a2", "actiontype_bad_touch_a2",
    "actiontype_non_action_a2", "actiontype_dribble_a2",
    "actiontype_goalkick_a2",
    # ── result one-hots (a0, a1, a2) ──
    "result_fail_a0", "result_success_a0", "result_offside_a0",
    "result_owngoal_a0", "result_yellow_card_a0", "result_red_card_a0",
    "result_fail_a1", "result_success_a1", "result_offside_a1",
    "result_owngoal_a1", "result_yellow_card_a1", "result_red_card_a1",
    "result_fail_a2", "result_success_a2", "result_offside_a2",
    "result_owngoal_a2", "result_yellow_card_a2", "result_red_card_a2",
    # ── bodypart one-hots (a0, a1, a2) ──
    "bodypart_foot_a0", "bodypart_head_a0", "bodypart_other_a0",
    "bodypart_head/other_a0",
    "bodypart_foot_a1", "bodypart_head_a1", "bodypart_other_a1",
    "bodypart_head/other_a1",
    "bodypart_foot_a2", "bodypart_head_a2", "bodypart_other_a2",
    "bodypart_head/other_a2",
    # ── spatial features ──
    "start_x_a0", "start_y_a0", "start_x_a1", "start_y_a1",
    "start_x_a2", "start_y_a2",
    "end_x_a0", "end_y_a0", "end_x_a1", "end_y_a1",
    "end_x_a2", "end_y_a2",
    # ── movement / displacement ──
    "dx_a0", "dy_a0", "movement_a0",
    "dx_a1", "dy_a1", "movement_a1",
    "dx_a2", "dy_a2", "movement_a2",
    "dx_a01", "dy_a01", "mov_a01",
    "dx_a02", "dy_a02", "mov_a02",
    # ── game-state ──
    "goalscore_team", "goalscore_opponent", "goalscore_diff",
    # ── timing ──
    "period_id_a0", "time_seconds_a0", "time_seconds_overall_a0",
    "period_id_a1", "time_seconds_a1", "time_seconds_overall_a1",
    "period_id_a2", "time_seconds_a2", "time_seconds_overall_a2",
)

_VAEP_FEATURE_SET: frozenset[str] = frozenset(VAEP_FEATURE_COLS)


def select_vaep_feature_cols(available_cols: Sequence[str]) -> list[str]:
    """Return the subset of *available_cols* that belong to the VAEP whitelist.

    Preserves the canonical order defined in :data:`VAEP_FEATURE_COLS`.
    Logs a warning for any whitelisted column missing from *available_cols*.
    """
    available = set(available_cols)
    missing = _VAEP_FEATURE_SET - available
    if missing:
        logger.warning(
            "%d VAEP feature(s) not found in data and will be skipped: %s",
            len(missing), sorted(missing),
        )
    selected = [c for c in VAEP_FEATURE_COLS if c in available]
    if not selected:
        raise ValueError(
            "No VAEP feature columns found in the dataset. "
            "Check that VAEP features have been generated."
        )
    logger.info("Selected %d / %d VAEP features.", len(selected), len(VAEP_FEATURE_COLS))
    return selected


# ──────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────


def drop_none_params(d: dict[str, Any]) -> dict[str, Any]:
    """Remove keys whose value is ``None`` (handy for sklearn / xgboost param dicts)."""
    return {k: v for k, v in d.items() if v is not None}

# ──────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────


def build_preprocessor(
    X_train: pd.DataFrame,
    num_feats: list[str],
    cat_feats: list[str],
    min_frequency: int = 10,
) -> tuple[StandardScaler, OneHotEncoder | None, list[str]]:
    """Fit a StandardScaler on numeric features and an optional OneHotEncoder on categoricals.

    Args:
        X_train: Training data containing both numeric and categorical columns.
        num_feats: Column names of numeric features.
        cat_feats: Column names of categorical features (empty list to skip).
        min_frequency: Minimum category frequency for OneHotEncoder.

    Returns:
        ``(scaler, encoder_or_None, all_feature_names)`` where
        *all_feature_names* is the ordered list of column names in the
        preprocessed output.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[num_feats])

    encoder: OneHotEncoder | None = None
    all_feature_names = list(num_feats)

    if cat_feats:
        encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", min_frequency=min_frequency
        )
        encoder.fit(X_train[cat_feats])
        cat_feature_names = encoder.get_feature_names_out(cat_feats).tolist()
        all_feature_names += cat_feature_names

    return scaler, encoder, all_feature_names


def preprocess_split(
    X: pd.DataFrame,
    num_feats: list[str],
    cat_feats: list[str],
    scaler: StandardScaler,
    encoder: OneHotEncoder | None,
) -> np.ndarray:
    """Transform a data split using a pre-fitted scaler and encoder.

    Args:
        X: Feature DataFrame.
        num_feats: Numeric feature column names.
        cat_feats: Categorical feature column names.
        scaler: Fitted StandardScaler.
        encoder: Fitted OneHotEncoder (or ``None`` if no categoricals).

    Returns:
        2-D NumPy array of preprocessed features.
    """
    X_num = scaler.transform(X[num_feats])
    if encoder is not None and cat_feats:
        X_cat = encoder.transform(X[cat_feats])
        return np.hstack([X_num, X_cat])
    return X_num


# ──────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────


def load_xy_game_split(
    dataset_key: str,
    target_col: str,
    data_dir: str | Path,
    val_pct: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load features/labels for one league-season and split by ``game_id``.

    The game-level split prevents action leakage across train/val.

    Args:
        dataset_key: e.g. ``'la_liga_2015_2016'``.
        target_col: ``'scores'`` or ``'concedes'``.
        data_dir: Folder with ``features_*.h5`` / ``labels_*.h5``.
        val_pct: Fraction of games reserved for validation.
        random_state: Seed for the game shuffle.

    Returns:
        ``(X_train, y_train, X_val, y_val)``
    """
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be 'scores' or 'concedes'")
    if not (0 <= val_pct < 1):
        raise ValueError("val_pct must be in [0, 1)")

    features_df, labels_df = load_dataset_tables(dataset_key=dataset_key, data_dir=data_dir)
    df = features_df.merge(labels_df, on=["game_id", "action_id"], how="inner")

    feature_cols = [
        col for col in df.columns
        if col not in {"game_id", "action_id", "scores", "concedes"}
    ]

    game_ids = np.array(sorted(df["game_id"].unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(game_ids)

    n_val_games = int(round(len(game_ids) * val_pct))
    print(f"Total games: {len(game_ids)}, Validation games: {n_val_games}")
    val_game_ids = set(game_ids[:n_val_games])

    is_val = df["game_id"].isin(val_game_ids)
    df_train = df.loc[~is_val]
    df_val = df.loc[is_val]

    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = df_train[target_col].astype(int)
    X_val = df_val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = df_val[target_col].astype(int)

    return X_train, y_train, X_val, y_val


def load_xy_all(
    dataset_key: str,
    target_col: str,
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load all actions from one dataset key without splitting.

    Args:
        dataset_key: e.g. ``'champions_league_2015_2016'``.
        target_col: ``'scores'`` or ``'concedes'``.
        data_dir: Folder with ``features_*.h5`` / ``labels_*.h5``.

    Returns:
        ``(X, y)``
    """
    features_df, labels_df = load_dataset_tables(dataset_key=dataset_key, data_dir=data_dir)
    df = features_df.merge(labels_df, on=["game_id", "action_id"], how="inner")
    feature_cols = [
        col for col in df.columns
        if col not in {"game_id", "action_id", "scores", "concedes"}
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].astype(int)
    return X, y


def load_xy_competition_split(
    target_col: str,
    data_file: str | Path,
    key_candidates: Sequence[str],
    validation_competitions: Sequence[str],
    test_competitions: Sequence[str],
    train_competitions: Sequence[str] | None = None,
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    list[str], list[str], list[str],
]:
    """Load data from a merged H5 file and split by competition name.

    Args:
        target_col: ``'scores'`` or ``'concedes'``.
        data_file: HDF5 file with a merged actions table.
        key_candidates: HDF5 keys to try (first match wins).
        validation_competitions: Competition names for validation.
        test_competitions: Competition names for test.
        train_competitions: Competition names for train (``None`` → all remaining).

    Returns:
        ``(X_train, y_train, X_val, y_val, X_test, y_test,
          train_comps, val_comps, test_comps)``
    """
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be 'scores' or 'concedes'")

    df = read_h5_table(data_file=data_file, key_candidates=key_candidates).copy()
    if "competition_name" not in df.columns:
        raise KeyError("Expected 'competition_name' column in dataset")

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
        col for col in df.select_dtypes(include=[np.number, bool]).columns
        if col not in excluded_cols
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns available in dataset")

    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = df_train[target_col].astype(int)
    X_val = df_val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = df_val[target_col].astype(int)
    X_test = df_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = df_test[target_col].astype(int)

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        sorted(train_set), sorted(val_set), sorted(test_set),
    )

# ──────────────────────────────────────────────
# Model persistence
# ──────────────────────────────────────────────


def _strip_xgb_callables(model: Any) -> tuple[Any, Any]:
    """Temporarily replace callable eval_metric entries with their string names.

    XGBoost stores ``eval_metric`` (including Python callables) in the
    estimator's ``__dict__``.  Closures (e.g. the composite metric) are
    not picklable.  This helper swaps callables for their ``__name__``
    strings so the model can be serialised with joblib.

    Returns:
        ``(original_eval_metric, original_callbacks)`` so the caller can
        restore them after saving.
    """
    original_eval_metric = getattr(model, "eval_metric", None)
    original_callbacks = getattr(model, "callbacks", None)

    if original_eval_metric is not None:
        cleaned: list[str] = []
        if isinstance(original_eval_metric, (list, tuple)):
            for m in original_eval_metric:
                cleaned.append(m if isinstance(m, str) else getattr(m, "__name__", str(m)))
        elif callable(original_eval_metric):
            cleaned.append(getattr(original_eval_metric, "__name__", str(original_eval_metric)))
        else:
            cleaned.append(str(original_eval_metric))
        model.eval_metric = cleaned

    # Callbacks may also contain unpicklable objects (closures, etc.).
    # They are not needed at inference time, so we drop them.
    if original_callbacks is not None:
        model.callbacks = None

    return original_eval_metric, original_callbacks


def save_model(model: Any, filepath: str | Path) -> None:
    """Save a trained model to disk using joblib.

    For XGBoost models that contain callable ``eval_metric`` entries
    (which are not picklable), the callables are temporarily replaced
    with their string names before serialisation.

    Args:
        model: Trained sklearn/xgboost model or pipeline.
        filepath: Path where to save the model (e.g., 'models/xgb_scores.pkl').
    """
    import joblib
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Strip unpicklable callables if present (XGBoost models).
    orig_eval_metric = orig_callbacks = None
    is_xgb = hasattr(model, "eval_metric")
    if is_xgb:
        orig_eval_metric, orig_callbacks = _strip_xgb_callables(model)

    try:
        joblib.dump(model, filepath)
    finally:
        # Restore callables so the in-memory model stays usable.
        if is_xgb:
            if orig_eval_metric is not None:
                model.eval_metric = orig_eval_metric
            if orig_callbacks is not None:
                model.callbacks = orig_callbacks

    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str | Path) -> Any:
    """Load a trained model from disk using joblib.

    Args:
        filepath: Path to the saved model file.

    Returns:
        The loaded model.
    """
    import joblib
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


# ──────────────────────────────────────────────
# Model building
# ──────────────────────────────────────────────


def build_sklearn_model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
    random_state: int = 42,
) -> ClassifierMixin | Pipeline:
    """Build a single sklearn estimator by name.

    Args:
        model_name: One of ``'rf'``, ``'logreg'``, ``'mlp'``.
        model_config: Hyperparameters for the chosen model.
            ``random_state`` is injected automatically if absent.
        random_state: Seed for reproducibility (used when not in *model_config*).

    Returns:
        Configured sklearn estimator or Pipeline.
    """
    cfg = dict(model_config or {})
    cfg.setdefault("random_state", random_state)

    if model_name == "rf":
        cfg.setdefault("n_estimators", 500)
        cfg.setdefault("n_jobs", -1)
        return RandomForestClassifier(**cfg)

    if model_name == "logreg":
        cfg.setdefault("max_iter", 5000)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**cfg)),
        ])

    if model_name == "mlp":
        cfg.setdefault("solver", "adam")
        cfg.setdefault("max_iter", 200)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(**cfg)),
        ])

    raise ValueError(f"Unknown model_name={model_name!r}. Use one of: rf, logreg, mlp")


def build_models(random_state: int = 42) -> dict[str, ClassifierMixin | Pipeline]:
    """Create baseline estimators used in the training benchmark."""
    return {
        "GLM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "Random Forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            n_estimators=100,
            criterion="gini",
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                max_iter=300, random_state=random_state, solver="adam",
            )),
        ]),
    }


def build_param_grids() -> dict[str, dict]:
    """Return default hyperparameter grids for each model.

    Returns:
        dict[str, dict]: Grids for ``"GLM"``, ``"Random Forest"``, ``"MLP"``.
    """
    return {
        "GLM": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
            "clf__solver": ["lbfgs"],
        },
        "Random Forest": {
            "n_estimators": [200, 400],
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 5],
            "class_weight": [None, "balanced_subsample"],
        },
        "MLP": {
            "clf__hidden_layer_sizes": [(64,), (128, 64)],
            "clf__alpha": [0.0001, 0.001],
            "clf__learning_rate_init": [0.001, 0.01],
        },
    }


# ──────────────────────────────────────────────
# XGBoost helpers
# ──────────────────────────────────────────────


def build_xgb_eval_set(
    mode: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    train_feature_cols: list[str],
    include_train: bool = False,
) -> tuple[list[tuple[pd.DataFrame, pd.Series]] | None, str]:
    """Build ``eval_set`` kwarg for XGBoost ``.fit()``.

    Args:
        mode: ``'train_val_split'`` or ``'none'``.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        train_feature_cols: Column order to align validation to.
        include_train: If True, prepend training pair to eval_set.

    Returns:
        ``(eval_set_or_None, mode_name)``
    """
    if mode == "none":
        return None, "none"

    if mode == "train_val_split":
        eval_set: list[tuple[pd.DataFrame, pd.Series]] = []
        if include_train:
            eval_set.append((X_train, y_train))
        eval_set.append((
            X_val.reindex(columns=train_feature_cols, fill_value=0),
            y_val,
        ))
        return eval_set, "train_val_split"

    raise ValueError("mode must be one of: 'train_val_split', 'none'")


# ──────────────────────────────────────────────
# Grid-search orchestration (existing API)
# ──────────────────────────────────────────────


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Mapping[str, ClassifierMixin | Pipeline],
    param_grids: Mapping[str, dict],
    scoring: str = "f1",
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
    backend: str | None = "threading",
) -> tuple[dict[str, ClassifierMixin | Pipeline], pd.DataFrame]:
    """Tune all models with GridSearchCV and return best estimators + summary."""

    def normalize_param_grid(
        estimator: ClassifierMixin | Pipeline, grid: dict
    ) -> dict:
        available = estimator.get_params(deep=True)
        normalized: dict = {}
        for key, value in grid.items():
            if key in available:
                normalized[key] = value
                continue
            prefixed_key = f"clf__{key}"
            if prefixed_key in available:
                normalized[prefixed_key] = value
                continue
            normalized[key] = value
        return normalized

    trained_models: dict[str, ClassifierMixin | Pipeline] = {}
    tuning_rows: list[dict] = []

    for model_name, estimator in tqdm(models.items(), desc="Tuning models"):
        grid = normalize_param_grid(estimator, param_grids[model_name])
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        search.fit(X_train, y_train)
        trained_models[model_name] = search.best_estimator_
        tuning_rows.append({
            "model": model_name,
            "best_f1_cv": search.best_score_,
            "best_params": str(search.best_params_),
        })

    tuning_table = pd.DataFrame(tuning_rows)
    return trained_models, tuning_table


def evaluate_models_on_datasets(
    trained_models: Mapping[str, ClassifierMixin | Pipeline],
    test_dataset_keys: list[str],
    train_feature_cols: list[str],
    target_col: str,
    data_dir: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Evaluate trained models on a list of test dataset keys."""
    results_tables_by_model: dict[str, pd.DataFrame] = {}

    for model_name, model in tqdm(trained_models.items(), desc="Evaluating models"):
        rows: list[dict] = []
        for dataset_key in test_dataset_keys:
            league, season = split_dataset_key(dataset_key)
            X_test, y_test = load_xy(
                dataset_key=dataset_key,
                target_col=target_col,
                data_dir=data_dir,
            )
            X_test_aligned = X_test.reindex(columns=train_feature_cols, fill_value=0)
            y_pred = model.predict(X_test_aligned)
            rows.append({
                "model": model_name,
                "league": league,
                "season": season,
                "tested_league_year": dataset_key,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            })

        table = pd.DataFrame(rows).sort_values(["league", "season"]).reset_index(drop=True)
        results_tables_by_model[model_name] = table

    comparison_table = pd.concat(results_tables_by_model.values(), ignore_index=True)
    comparison_table = comparison_table[
        ["model", "league", "season", "accuracy", "precision", "recall", "f1", "tested_league_year"]
    ]
    comparison_table = comparison_table.sort_values(
        ["model", "league", "season"]
    ).reset_index(drop=True)

    return results_tables_by_model, comparison_table
