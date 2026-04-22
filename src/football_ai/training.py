"""Core training utilities for football AI models.

Centralises data loading, model building, evaluation, and training helpers.
Scripts in ``scripts/`` are thin CLI wrappers that call into these functions.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

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

from .data import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    load_dataset_tables,
    load_xy,
    parse_season_sort_key,
    read_h5_table,
    split_dataset_key,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedSplitFrame:
    """Materialized features, labels, and metadata for one named split."""

    name: str
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    competition_seasons: pd.DataFrame

    @property
    def competitions(self) -> list[str]:
        if self.competition_seasons.empty:
            return []
        return sorted(self.competition_seasons["competition_name"].astype(str).unique().tolist())

    @property
    def season_names(self) -> list[str]:
        if self.competition_seasons.empty:
            return []
        seasons = self.competition_seasons["season_name"].astype(str).unique().tolist()
        return sorted(seasons, key=parse_season_sort_key)

    @property
    def n_games(self) -> int:
        if self.df.empty or "game_id" not in self.df.columns:
            return 0
        return int(self.df["game_id"].nunique())


@dataclass
class ResolvedCompetitionSeasonSplit:
    """Structured train/calib/test/target view for competition-season experiments."""

    source_train: ResolvedSplitFrame
    source_val: ResolvedSplitFrame
    feature_cols: list[str]
    split_config: dict[str, Any]
    _lazy_split_loaders: dict[str, Callable[[], ResolvedSplitFrame]] = field(
        default_factory=dict,
        repr=False,
    )
    _materialized_splits: dict[str, ResolvedSplitFrame] = field(
        default_factory=dict,
        repr=False,
    )

    def _get_lazy_split(self, name: str) -> ResolvedSplitFrame:
        if name not in self._lazy_split_loaders:
            raise KeyError(f"Unknown split {name!r}")
        if name not in self._materialized_splits:
            self._materialized_splits[name] = self._lazy_split_loaders[name]()
        return self._materialized_splits[name]

    @property
    def calib(self) -> ResolvedSplitFrame:
        return self._get_lazy_split("calib")

    @property
    def target(self) -> ResolvedSplitFrame:
        return self._get_lazy_split("target")

    @property
    def test_names(self) -> list[str]:
        return sorted(
            name.removeprefix("test_")
            for name in self._lazy_split_loaders
            if name.startswith("test_")
        )

    def get_test_split(self, name: str) -> ResolvedSplitFrame:
        return self._get_lazy_split(f"test_{name}")

    def iter_test_splits(self) -> list[tuple[str, ResolvedSplitFrame]]:
        return [(name, self.get_test_split(name)) for name in self.test_names]

    def is_materialized(self, name: str) -> bool:
        if name in {"source_train", "source_val"}:
            return True
        return name in self._materialized_splits

    def named_splits(self, include_lazy: bool = False) -> dict[str, ResolvedSplitFrame]:
        splits = {
            "source_train": self.source_train,
            "source_val": self.source_val,
        }
        if include_lazy:
            splits["calib"] = self.calib
            splits["target"] = self.target
            for name in self.test_names:
                splits[f"test_{name}"] = self.get_test_split(name)
        return splits


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
_VAEP_CATEGORICAL_PREFIXES: tuple[str, ...] = ("actiontype_", "result_", "bodypart_")
_VAEP_ONEHOT_FEATURE_COLS: tuple[str, ...] = tuple(
    col for col in VAEP_FEATURE_COLS if col.startswith(_VAEP_CATEGORICAL_PREFIXES)
)
_VAEP_BASE_NUMERIC_FEATURE_COLS: tuple[str, ...] = tuple(
    col for col in VAEP_FEATURE_COLS if col not in _VAEP_ONEHOT_FEATURE_COLS
)
_SCORES_CATEGORICAL_FEATURE_COLS: tuple[str, ...] = (
    "actiontype_a0", "actiontype_a1", "actiontype_a2",
    "result_a0", "result_a1", "result_a2",
    "bodypart_a0", "bodypart_a1", "bodypart_a2",
)
_SCORES_BINARY_FEATURE_COLS: tuple[str, ...] = (
    "in_final_third",
    "start_in_box",
    "end_in_box",
    "same_team_a01",
    "same_team_a12",
    "same_team_a02",
    "turnover_a01",
    "turnover_a12",
)
_SCORES_DERIVED_FEATURE_COLS: tuple[str, ...] = (
    "start_dist_to_goal",
    "end_dist_to_goal",
    "start_angle_to_goal",
    "end_angle_to_goal",
    "in_final_third",
    "start_in_box",
    "end_in_box",
    "dist_to_goal_delta",
    "same_team_a01",
    "same_team_a12",
    "same_team_a02",
    "turnover_a01",
    "turnover_a12",
    "possession_chain_len",
)
_SCORES_ENGINEERED_FEATURE_COLS: tuple[str, ...] = (
    *_VAEP_BASE_NUMERIC_FEATURE_COLS,
    *_SCORES_DERIVED_FEATURE_COLS,
    *_SCORES_CATEGORICAL_FEATURE_COLS,
)
_SCORES_METADATA_COLS: tuple[str, ...] = (
    "game_id",
    "action_id",
    "competition_name",
    "season_name",
    "season_id",
    "team_id",
)


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


def _extract_onehot_categories(
    available_cols: Sequence[str],
    family_prefix: str,
    slot: str,
) -> list[tuple[str, str]]:
    slot_suffix = f"_{slot}"
    matches: list[tuple[str, str]] = []
    for col in available_cols:
        if not col.startswith(family_prefix) or not col.endswith(slot_suffix):
            continue
        label = col[len(family_prefix):-len(slot_suffix)]
        matches.append((col, label))
    return matches


def _decode_onehot_slot_feature(
    df: pd.DataFrame,
    family_prefix: str,
    slot: str,
    output_col: str,
) -> pd.Series:
    matches = _extract_onehot_categories(df.columns, family_prefix=family_prefix, slot=slot)
    if not matches:
        raise ValueError(
            f"Could not decode {output_col!r}: no one-hot columns found for "
            f"prefix={family_prefix!r}, slot={slot!r}."
        )

    block_cols = [col for col, _ in matches]
    labels = np.asarray([label for _, label in matches], dtype=object)
    block = (
        df.loc[:, block_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    has_signal = block.gt(0).any(axis=1).to_numpy()
    max_idx = block.to_numpy().argmax(axis=1)

    decoded = pd.Series(pd.NA, index=df.index, dtype="object", name=output_col)
    if has_signal.any():
        decoded.loc[has_signal] = labels[max_idx[has_signal]]
    return decoded.astype("category")


def _distance_to_goal(x: pd.Series, y: pd.Series) -> pd.Series:
    goal_x = float(FIELD_LENGTH)
    goal_y = float(FIELD_WIDTH) / 2.0
    return np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)


def _angle_to_goal(x: pd.Series, y: pd.Series) -> pd.Series:
    goal_x = float(FIELD_LENGTH)
    goal_y = float(FIELD_WIDTH) / 2.0
    dx = (goal_x - x).clip(lower=1e-6)
    dy = (goal_y - y).abs()
    return np.arctan2(dy, dx)


def _in_box(x: pd.Series, y: pd.Series) -> pd.Series:
    goal_y = float(FIELD_WIDTH) / 2.0
    return (
        (x >= float(FIELD_LENGTH) - 16.5)
        & ((y - goal_y).abs() <= 20.16)
    )


def normalize_xgb_feature_frame(
    X: pd.DataFrame,
    *,
    binary_cols: Sequence[str] = (),
    categorical_cols: Sequence[str] = (),
    categorical_levels: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Cast XGBoost feature columns to compact, deterministic dtypes."""
    X_norm = X.copy()
    binary_set = {col for col in binary_cols if col in X_norm.columns}
    categorical_set = {col for col in categorical_cols if col in X_norm.columns}
    numeric_cols = [
        col for col in X_norm.columns if col not in binary_set and col not in categorical_set
    ]

    if numeric_cols:
        numeric_frame = (
            X_norm.loc[:, numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        for col in numeric_cols:
            X_norm[col] = numeric_frame[col].astype(np.float32)

    for col in binary_set:
        X_norm[col] = (
            pd.to_numeric(X_norm[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=0, upper=1)
            .astype(np.uint8)
        )

    for col in categorical_set:
        if pd.api.types.is_numeric_dtype(X_norm[col]) and not pd.api.types.is_bool_dtype(X_norm[col]):
            X_norm[col] = (
                pd.to_numeric(X_norm[col], errors="coerce")
                .fillna(-1)
                .astype(np.int32)
            )
            continue

        levels = None
        if categorical_levels is not None and col in categorical_levels:
            levels = list(dict.fromkeys(str(level) for level in categorical_levels[col]))
            levels.append("__MISSING__")

        normalized = X_norm[col].astype("string").fillna("__MISSING__")
        if levels is None:
            X_norm[col] = normalized.astype("category").cat.codes.astype(np.int32)
        else:
            X_norm[col] = pd.Categorical(normalized, categories=levels).codes.astype(np.int32)

    return X_norm


def normalize_xgb_labels(y: pd.Series) -> pd.Series:
    """Cast binary labels to uint8."""
    values = (
        pd.to_numeric(y, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(np.uint8)
    )
    return pd.Series(values, index=y.index, name=y.name)


def is_scores_engineered_dataset(df: pd.DataFrame) -> bool:
    """Return True when df already contains the persisted scores feature schema."""
    return set(_SCORES_ENGINEERED_FEATURE_COLS).issubset(df.columns)


def build_scores_xgb_feature_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the training-time XGBoost feature frame for the ``scores`` target.

    This keeps the persisted VAEP dataset unchanged and reshapes features only
    at training time by:
    - decoding one-hot categorical blocks into single categorical columns
    - adding geometry features from the oriented ``a0`` coordinates
    - adding possession continuity features from per-game action history

    Args:
        df: Raw merged VAEP dataset loaded from HDF5.

    Returns:
        ``(X, feature_cols, categorical_cols)`` where ``X`` preserves the
        original row index and includes mixed numeric / categorical dtypes.
    """
    total_start = perf_counter()
    categorical_cols = list(_SCORES_CATEGORICAL_FEATURE_COLS)

    if is_scores_engineered_dataset(df):
        feature_cols = list(_SCORES_ENGINEERED_FEATURE_COLS)
        X = normalize_xgb_feature_frame(
            df.loc[:, feature_cols].copy(),
            binary_cols=_SCORES_BINARY_FEATURE_COLS,
            categorical_cols=categorical_cols,
        )
        logger.info(
            "scores feature prep: reused persisted engineered schema rows=%d features=%d in %.2fs",
            len(X),
            len(feature_cols),
            perf_counter() - total_start,
        )
        return X, feature_cols, categorical_cols

    required_cols = {
        "game_id",
        "action_id",
        "team_id",
        "start_x_a0",
        "start_y_a0",
        "end_x_a0",
        "end_y_a0",
    }
    missing_required = sorted(required_cols - set(df.columns))
    if missing_required:
        raise ValueError(
            "Missing required columns for training-time scores feature prep: "
            f"{missing_required}"
        )

    available = set(df.columns)
    base_numeric_cols = [col for col in _VAEP_BASE_NUMERIC_FEATURE_COLS if col in available]
    logger.info(
        "scores feature prep: start rows=%d cols=%d base_numeric=%d",
        len(df),
        len(df.columns),
        len(base_numeric_cols),
    )
    X = df.loc[:, base_numeric_cols].copy()
    categorical_levels: dict[str, list[str]] = {}

    decode_start = perf_counter()
    for feature_name, family_prefix in (
        ("actiontype", "actiontype_"),
        ("result", "result_"),
        ("bodypart", "bodypart_"),
    ):
        family_start = perf_counter()
        for slot in ("a0", "a1", "a2"):
            output_col = f"{feature_name}_{slot}"
            slot_start = perf_counter()
            matches = _extract_onehot_categories(
                df.columns,
                family_prefix=family_prefix,
                slot=slot,
            )
            categorical_levels[output_col] = [label for _, label in matches]
            X[output_col] = _decode_onehot_slot_feature(
                df=df,
                family_prefix=family_prefix,
                slot=slot,
                output_col=output_col,
            )
            logger.info(
                "scores feature prep: decoded %s in %.2fs",
                output_col,
                perf_counter() - slot_start,
            )
        logger.info(
            "scores feature prep: decoded %s family in %.2fs",
            feature_name,
            perf_counter() - family_start,
        )
    logger.info(
        "scores feature prep: all categorical decoding finished in %.2fs",
        perf_counter() - decode_start,
    )

    geometry_start = perf_counter()
    start_x = pd.to_numeric(df["start_x_a0"], errors="coerce")
    start_y = pd.to_numeric(df["start_y_a0"], errors="coerce")
    end_x = pd.to_numeric(df["end_x_a0"], errors="coerce")
    end_y = pd.to_numeric(df["end_y_a0"], errors="coerce")

    start_dist = _distance_to_goal(start_x, start_y)
    end_dist = _distance_to_goal(end_x, end_y)
    X["start_dist_to_goal"] = start_dist
    X["end_dist_to_goal"] = end_dist
    X["start_angle_to_goal"] = _angle_to_goal(start_x, start_y)
    X["end_angle_to_goal"] = _angle_to_goal(end_x, end_y)
    X["in_final_third"] = start_x >= (2.0 * float(FIELD_LENGTH) / 3.0)
    X["start_in_box"] = _in_box(start_x, start_y)
    X["end_in_box"] = _in_box(end_x, end_y)
    X["dist_to_goal_delta"] = start_dist - end_dist
    logger.info(
        "scores feature prep: geometry features finished in %.2fs",
        perf_counter() - geometry_start,
    )

    possession_start_time = perf_counter()
    history = (
        df.loc[:, ["game_id", "action_id", "team_id"]]
        .copy()
        .sort_values(["game_id", "action_id"])
    )
    team_id = history["team_id"]
    prev_team = history.groupby("game_id")["team_id"].shift(1)
    prev_prev_team = history.groupby("game_id")["team_id"].shift(2)
    possession_start = team_id.ne(prev_team).fillna(True)
    possession_group = possession_start.groupby(history["game_id"]).cumsum()

    history["same_team_a01"] = team_id.eq(prev_team).fillna(False)
    history["same_team_a12"] = prev_team.eq(prev_prev_team).fillna(False)
    history["same_team_a02"] = team_id.eq(prev_prev_team).fillna(False)
    history["turnover_a01"] = prev_team.notna() & team_id.ne(prev_team)
    history["turnover_a12"] = prev_prev_team.notna() & prev_team.ne(prev_prev_team)
    history["possession_chain_len"] = history.groupby(
        ["game_id", possession_group]
    ).cumcount() + 1

    for col in (
        "same_team_a01",
        "same_team_a12",
        "same_team_a02",
        "turnover_a01",
        "turnover_a12",
        "possession_chain_len",
    ):
        X[col] = history[col].reindex(df.index)
    logger.info(
        "scores feature prep: possession features finished in %.2fs",
        perf_counter() - possession_start_time,
    )

    feature_cols = list(_SCORES_ENGINEERED_FEATURE_COLS)
    X = normalize_xgb_feature_frame(
        X.loc[:, feature_cols],
        binary_cols=_SCORES_BINARY_FEATURE_COLS,
        categorical_cols=categorical_cols,
        categorical_levels=categorical_levels,
    )
    logger.info(
        "scores feature prep: complete rows=%d features=%d in %.2fs",
        len(X),
        len(feature_cols),
        perf_counter() - total_start,
    )
    return X, feature_cols, categorical_cols


def prepare_vaep_xgb_features(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare an XGBoost-ready feature frame from the raw VAEP table.

    Args:
        df: Loaded raw VAEP dataset.
        target_col: ``"scores"`` or ``"concedes"``.

    Returns:
        ``(X, feature_cols, categorical_cols)``.
    """
    if target_col == "scores":
        return build_scores_xgb_feature_frame(df)
    else:
        feature_cols = select_vaep_feature_cols(df.columns)
        categorical_cols = []
        X = normalize_xgb_feature_frame(
            df.loc[:, feature_cols].copy(),
            binary_cols=[col for col in _VAEP_ONEHOT_FEATURE_COLS if col in feature_cols],
        )
        return X, feature_cols, categorical_cols


def build_scores_engineered_dataset(
    df: pd.DataFrame,
    metadata_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the persisted engineered dataset for the scores XGBoost workflow."""
    X, feature_cols, categorical_cols = build_scores_xgb_feature_frame(df)
    selected_metadata_cols = [
        col for col in (metadata_cols or _SCORES_METADATA_COLS) if col in df.columns
    ]
    engineered = df.loc[:, selected_metadata_cols].copy()
    for col in ("competition_name", "season_name"):
        if col in engineered.columns:
            engineered[col] = engineered[col].astype(str)
    for label_col in ("scores", "concedes"):
        if label_col in df.columns:
            engineered[label_col] = normalize_xgb_labels(df[label_col])
    engineered = pd.concat([engineered, X], axis=1)
    return engineered, feature_cols, categorical_cols


# ──────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────


def drop_none_params(d: dict[str, Any]) -> dict[str, Any]:
    """Remove keys whose value is ``None`` (handy for sklearn / xgboost param dicts)."""
    return {k: v for k, v in d.items() if v is not None}


# ──────────────────────────────────────────────
# Custom XGBoost eval metrics
# ──────────────────────────────────────────────
# XGBoost's C++ engine supports only a fixed set of built-in metrics.
# Metrics like "recall" must be provided as Python callables.
# For the sklearn API (XGBClassifier), the callable signature is:
#     (y_true: np.ndarray, y_score: np.ndarray) -> float
# and the metric name is taken from ``func.__name__``.
#
# IMPORTANT: XGBoost ≤ 3.2 only supports **one** custom callable in
# ``eval_metric``.  When the user requests multiple custom metrics,
# we bundle them into a single "composite" callable (which returns the
# last custom metric's value for early-stopping) plus a companion
# ``TrainingCallback`` that injects the remaining metric values into
# ``evals_log`` so they appear in ``model.evals_result()``.

from xgboost.callback import TrainingCallback

# Built-in metric names recognised by XGBoost's C++ engine.
_XGB_BUILTIN_METRICS: frozenset[str] = frozenset({
    "rmse", "rmsle", "mae", "mape", "mphe", "logloss", "error",
    "merror", "mlogloss", "auc", "aucpr", "ndcg", "map",
    "pre", "gamma-nloglik", "gamma-deviance", "poisson-nloglik",
    "tweedie-nloglik", "tweedie-nloglik@1.5", "cox-nloglik",
    "aft-nloglik", "interval-regression-accuracy",
})


# ── Individual custom metric functions ──────────────────────────────────────

def _xgb_recall(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Recall = TP / (TP + FN).  Threshold at 0.5 on predicted probabilities."""
    y_hat = (np.asarray(y_score) > 0.5).astype(int)
    y_t = np.asarray(y_true).astype(int)
    tp = int(((y_hat == 1) & (y_t == 1)).sum())
    p = int((y_t == 1).sum())
    return float(tp / p) if p > 0 else 0.0


def _xgb_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision = TP / (TP + FP).  Threshold at 0.5 on predicted probabilities."""
    y_hat = (np.asarray(y_score) > 0.5).astype(int)
    y_t = np.asarray(y_true).astype(int)
    tp = int(((y_hat == 1) & (y_t == 1)).sum())
    pp = int((y_hat == 1).sum())
    return float(tp / pp) if pp > 0 else 0.0


def _xgb_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """F1 = 2 * precision * recall / (precision + recall).  Threshold at 0.5."""
    y_hat = (np.asarray(y_score) > 0.5).astype(int)
    y_t = np.asarray(y_true).astype(int)
    tp = int(((y_hat == 1) & (y_t == 1)).sum())
    pp = int((y_hat == 1).sum())  # predicted positives
    p = int((y_t == 1).sum())     # actual positives
    prec = tp / pp if pp > 0 else 0.0
    rec = tp / p if p > 0 else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# Give them clean names that XGBoost will log in eval output.
_xgb_recall.__name__ = "recall"
_xgb_precision.__name__ = "precision"
_xgb_f1.__name__ = "f1"

# Registry: maps YAML-friendly string names to callables.
_CUSTOM_XGB_METRICS: dict[str, Any] = {
    "recall": _xgb_recall,
    "precision": _xgb_precision,
    "f1": _xgb_f1,
}

# Custom metrics that should be *maximized* for early stopping.
# (All current custom metrics are higher-is-better.)
_CUSTOM_METRICS_MAXIMIZE: frozenset[str] = frozenset({
    "recall", "precision", "f1",
})

# Built-in XGBoost metrics that should be maximized.
_BUILTIN_METRICS_MAXIMIZE: frozenset[str] = frozenset({
    "auc", "aucpr", "ndcg", "map", "pre",
})


# ── Composite callable + callback for multiple custom metrics ───────────────

def _make_composite_xgb_metric(
    custom_names: list[str],
) -> tuple[Any, TrainingCallback]:
    """Bundle several custom metrics into one callable + companion callback.

    XGBoost only accepts **one** custom callable in ``eval_metric``.
    This function creates:

    * A *composite callable* that computes **all** requested custom metrics,
      stores them in a shared buffer, and **returns** the value of the
      **last** metric (which drives early-stopping).
    * A lightweight :class:`TrainingCallback` that, after each boosting
      round, injects the "extra" metric values into ``evals_log`` so they
      appear in ``model.evals_result()``.

    Args:
        custom_names: Ordered list of custom metric names (e.g.
            ``["precision", "recall", "f1"]``).

    Returns:
        ``(composite_callable, companion_callback)``
    """
    fns = [(name, _CUSTOM_XGB_METRICS[name]) for name in custom_names]
    early_stop_name = custom_names[-1]
    extra_names = custom_names[:-1]

    # Shared buffer: list of per-eval-set dicts within the current round.
    # The callable appends one dict per call; the callback reads & resets.
    _round_buffer: list[dict[str, float]] = []

    def composite(y_true: np.ndarray, y_score: np.ndarray) -> float:
        values: dict[str, float] = {}
        for name, fn in fns:
            values[name] = fn(y_true, y_score)
        _round_buffer.append(values)
        return values[early_stop_name]

    composite.__name__ = early_stop_name

    class _ExtraMetricsInjector(TrainingCallback):
        """Inject extra custom metric values into *evals_log*.

        Must be placed **before** ``EarlyStopping`` in the callback list
        (the default when passed via ``XGBClassifier(callbacks=[...])``)
        so that early-stopping can also see the injected metrics.
        """

        def after_iteration(
            self,
            model: Any,
            epoch: int,
            evals_log: dict[str, dict[str, list[float]]],
        ) -> bool:
            eval_keys = list(evals_log.keys())
            for i, eval_key in enumerate(eval_keys):
                if i < len(_round_buffer):
                    vals = _round_buffer[i]
                    for name in extra_names:
                        evals_log[eval_key].setdefault(name, []).append(
                            vals.get(name, 0.0)
                        )
            _round_buffer.clear()
            return False  # never request stopping

    return composite, _ExtraMetricsInjector()


# ── Public resolver ─────────────────────────────────────────────────────────

def resolve_xgb_eval_metrics(
    eval_metric: str | list[str] | None,
    early_stopping_rounds: int | None = None,
    early_stopping_metric: str | None = None,
) -> tuple[list[str | Any] | None, list[Any], int | None]:
    """Replace non-built-in metric names with callable implementations.

    Because XGBoost (≤ 3.2) only allows **one** custom callable in
    ``eval_metric``, this function bundles multiple custom metrics into a
    single composite callable and returns a companion callback list that
    must be prepended to the model's ``callbacks`` parameter.

    When the early-stopping metric requires maximisation (or is not the
    default last-in-list metric), this function injects an explicit
    ``xgboost.callback.EarlyStopping`` callback and returns
    ``early_stopping_rounds_out = None`` so that the caller **removes**
    the ``early_stopping_rounds`` constructor parameter (which would
    create a *second*, conflicting ``EarlyStopping`` callback).

    Args:
        eval_metric: The ``eval_metric`` value from config (string, list,
            or *None*).
        early_stopping_rounds: The early-stopping patience from config.
            Needed to build the explicit ``EarlyStopping`` callback when
            the last metric must be maximised.
        early_stopping_metric: Name of the metric to use for early
            stopping.  If *None*, XGBoost's default (last metric in
            ``eval_metric``) is used.  When set, an explicit
            ``EarlyStopping(metric_name=...)`` callback is always
            injected so the stopping metric is independent of the
            ``eval_metric`` order.

    Returns:
        ``(resolved_metrics, extra_callbacks, early_stopping_rounds_out)``
        where:

        * *resolved_metrics* – list to pass as ``eval_metric``.
        * *extra_callbacks* – (possibly empty) list of callbacks to
          **prepend** to the model's ``callbacks``.
        * *early_stopping_rounds_out* – the value to set on the model's
          ``early_stopping_rounds`` parameter.  ``None`` means *"do not
          pass it; the explicit callback handles stopping"*.

    Raises:
        ValueError: If an unknown custom metric name is requested.
    """
    if eval_metric is None:
        return None, [], early_stopping_rounds

    if isinstance(eval_metric, str):
        eval_metric = [eval_metric]

    # Separate built-in from custom, preserving order.
    builtin: list[str] = []
    custom_names: list[str] = []
    for m in eval_metric:
        if not isinstance(m, str):
            raise TypeError(f"eval_metric entries must be strings, got {type(m)}")
        if m in _XGB_BUILTIN_METRICS:
            builtin.append(m)
        elif m in _CUSTOM_XGB_METRICS:
            custom_names.append(m)
        else:
            raise ValueError(
                f"Unknown eval_metric {m!r}. Built-in XGBoost metrics: "
                f"{sorted(_XGB_BUILTIN_METRICS)}. "
                f"Custom metrics available: {sorted(_CUSTOM_XGB_METRICS)}."
            )

    extra_callbacks: list[Any] = []

    if len(custom_names) == 0:
        resolved = builtin
    elif len(custom_names) == 1:
        # Single custom metric – pass directly as callable (no callback needed).
        fn = _CUSTOM_XGB_METRICS[custom_names[0]]
        resolved = builtin + [fn]
        logger.info("Resolved custom eval metric %r → callable", custom_names[0])
    else:
        # Multiple custom metrics – composite callable + injection callback.
        composite, cb = _make_composite_xgb_metric(custom_names)
        resolved = builtin + [composite]
        extra_callbacks.append(cb)
        logger.info(
            "Resolved %d custom eval metrics %s → composite callable "
            "(early-stopping on %r) + callback",
            len(custom_names),
            custom_names,
            custom_names[-1],
        )

    # ── Determine early-stopping metric and direction ───────────────────
    # By default XGBoost monitors the *last* eval_metric.  When the user
    # specifies ``early_stopping_metric`` we inject an explicit
    # ``EarlyStopping`` callback with ``metric_name`` so the stopping
    # criterion is decoupled from the eval_metric order.
    # We also need an explicit callback whenever the monitored metric
    # must be maximised (custom metrics default to minimize).
    es_rounds_out: int | None = early_stopping_rounds

    # Which metric drives early stopping?
    es_metric: str = early_stopping_metric or eval_metric[-1]

    if early_stopping_rounds is not None:
        # Decide direction (maximize vs minimize)
        needs_maximize = (
            es_metric in _CUSTOM_METRICS_MAXIMIZE
            or es_metric in _BUILTIN_METRICS_MAXIMIZE
        )

        # We need an explicit callback when:
        #  1. The user specified early_stopping_metric (need metric_name), OR
        #  2. The metric needs maximize (XGBoost default is minimize for custom)
        need_explicit_cb = (
            early_stopping_metric is not None
            or es_metric in _CUSTOM_METRICS_MAXIMIZE
            or es_metric in (_CUSTOM_XGB_METRICS.keys() - _CUSTOM_METRICS_MAXIMIZE)
        )

        if need_explicit_cb:
            from xgboost.callback import EarlyStopping as _EarlyStopping

            # For custom metrics, XGBoost logs them under the callable's
            # __name__; for built-in metrics use the string name directly.
            es_cb = _EarlyStopping(
                rounds=early_stopping_rounds,
                metric_name=es_metric,
                maximize=needs_maximize,
                save_best=True,
            )
            extra_callbacks.append(es_cb)
            es_rounds_out = None  # caller must NOT set early_stopping_rounds
            logger.info(
                "Early stopping: explicit EarlyStopping(rounds=%d, "
                "metric_name=%r, maximize=%s)",
                early_stopping_rounds,
                es_metric,
                needs_maximize,
            )
        else:
            logger.info(
                "Early stopping: using default direction for metric %r",
                es_metric,
            )

    return resolved, extra_callbacks, es_rounds_out


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


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _canonicalize_season_tokens(season_value: str) -> set[str]:
    text = str(season_value).strip()
    lowered = text.lower().replace(" ", "")
    tokens = {lowered}
    match = re.search(r"(?P<start>\d{4})\s*[-/]\s*(?P<end>\d{2,4})", lowered)
    if not match:
        return tokens

    start_year = int(match.group("start"))
    end_raw = match.group("end")
    end_year = int(end_raw) if len(end_raw) == 4 else (start_year // 100) * 100 + int(end_raw)
    tokens.update({
        f"{start_year}/{end_year:04d}",
        f"{start_year}-{end_year:04d}",
        f"{start_year}/{end_year % 100:02d}",
        f"{start_year}-{end_year % 100:02d}",
    })
    return tokens


def resolve_season_aliases(
    available_seasons: Sequence[str],
    requested_seasons: Sequence[str],
) -> list[str]:
    """Resolve user-friendly season specs against canonical dataset labels."""
    if not requested_seasons:
        return []

    available = [str(season) for season in available_seasons]
    resolved: list[str] = []
    for requested in requested_seasons:
        requested_tokens = _canonicalize_season_tokens(str(requested))
        matches = [
            season for season in available
            if requested_tokens & _canonicalize_season_tokens(season)
        ]
        unique_matches = sorted(set(matches), key=lambda season: str(season))
        if not unique_matches:
            raise ValueError(
                f"Could not resolve requested season {requested!r} from available seasons: {sorted(set(available))}"
            )
        if len(unique_matches) > 1:
            raise ValueError(
                f"Season spec {requested!r} is ambiguous; matches={unique_matches}"
            )
        resolved.append(unique_matches[0])

    return sorted(set(resolved), key=parse_season_sort_key)


def _validate_requested_competitions(
    requested_competitions: Sequence[str],
    available_competitions: Sequence[str],
    label: str,
) -> list[str]:
    requested = [str(comp) for comp in requested_competitions]
    available_set = {str(comp) for comp in available_competitions}
    missing = sorted(set(requested) - available_set)
    if missing:
        raise ValueError(f"{label} contains unknown competitions: {missing}")
    return requested


def _build_competition_season_coverage(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["competition_name", "season_name"]
    agg: dict[str, tuple[str, str]] = {}
    if "season_id" in df.columns:
        agg["season_id"] = ("season_id", "first")
    if "competition_id" in df.columns:
        agg["competition_id"] = ("competition_id", "first")
    agg["rows"] = ("game_id", "size")
    agg["games"] = ("game_id", "nunique")

    if df.empty:
        empty_cols = cols + list(agg.keys())
        return pd.DataFrame(columns=empty_cols)

    coverage = (
        df.groupby(cols, dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(["competition_name", "season_name"])
        .reset_index(drop=True)
    )
    return coverage


def _build_split_frame(
    name: str,
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    X: pd.DataFrame | None = None,
) -> ResolvedSplitFrame:
    if X is None:
        X, _, _ = prepare_vaep_xgb_features(frame, target_col=target_col)
    X = X.loc[:, list(feature_cols)].copy()
    y = (
        normalize_xgb_labels(frame[target_col])
        if target_col in frame.columns
        else pd.Series(dtype=np.uint8)
    )
    coverage = _build_competition_season_coverage(frame)
    return ResolvedSplitFrame(name=name, df=frame.copy(), X=X, y=y, competition_seasons=coverage)


def resolve_competition_season_split_spec(
    df: pd.DataFrame,
    split_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve source/calib/test/target membership at competition-season level."""
    required_columns = {"competition_name", "season_name", "season_id", "game_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Expected dataset columns for competition-season split: {sorted(missing_columns)}")

    df_local = df.copy()
    df_local["competition_name"] = df_local["competition_name"].astype(str)
    df_local["season_name"] = df_local["season_name"].astype(str)

    split_cfg = dict(split_config)
    source_cfg = dict(split_cfg.get("source", {}))
    calib_cfg = dict(split_cfg.get("calib", {}))
    test_cfg = dict(split_cfg.get("test", {}))

    if not source_cfg and "source_competitions" in split_cfg:
        source_cfg = {"competitions": split_cfg.get("source_competitions", [])}
    if not calib_cfg and "calib_competitions" in split_cfg:
        calib_cfg = {"competitions": split_cfg.get("calib_competitions", [])}
    if not test_cfg and "target_competitions" in split_cfg:
        test_cfg = {"competitions": split_cfg.get("target_competitions", [])}

    source_competitions = _validate_requested_competitions(
        _as_string_list(source_cfg.get("competitions", [])),
        df_local["competition_name"].dropna().unique().tolist(),
        "source.competitions",
    )
    calib_competitions = _validate_requested_competitions(
        _as_string_list(calib_cfg.get("competitions", split_cfg.get("calib_competitions", []))),
        df_local["competition_name"].dropna().unique().tolist(),
        "calib.competitions",
    )
    test_competitions = _validate_requested_competitions(
        _as_string_list(test_cfg.get("competitions", split_cfg.get("target_competitions", []))),
        df_local["competition_name"].dropna().unique().tolist(),
        "test.competitions",
    )

    exclude_specs = _as_string_list(source_cfg.get("exclude_seasons", []))
    year_shift_cfg = dict(test_cfg.get("year_shift", {}))
    year_shift_specs = _as_string_list(year_shift_cfg.get("seasons", exclude_specs))
    explicit_cfg = dict(test_cfg.get("league_season_shift", {}))
    explicit_map = explicit_cfg.get("explicit", {}) or {}

    available_by_comp = {
        competition: sorted(
            df_local.loc[df_local["competition_name"] == competition, "season_name"].dropna().astype(str).unique().tolist(),
        )
        for competition in sorted(df_local["competition_name"].dropna().astype(str).unique().tolist())
    }

    excluded_keys: set[tuple[str, str]] = set()
    year_shift_keys: set[tuple[str, str]] = set()
    for competition in source_competitions:
        resolved_excluded = resolve_season_aliases(available_by_comp[competition], exclude_specs)
        resolved_year_shift = resolve_season_aliases(available_by_comp[competition], year_shift_specs)
        excluded_keys.update((competition, season) for season in resolved_excluded)
        year_shift_keys.update((competition, season) for season in resolved_year_shift)

    source_keys = {
        (str(row.competition_name), str(row.season_name))
        for row in (
            df_local.loc[df_local["competition_name"].isin(source_competitions), ["competition_name", "season_name"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
    } - excluded_keys - year_shift_keys

    if not source_keys:
        raise ValueError("Resolved source competition-season set is empty")

    source_seasons = {season for _, season in source_keys}
    non_source_test_competitions = sorted(set(test_competitions) - set(source_competitions))
    test_rows = df_local.loc[df_local["competition_name"].isin(non_source_test_competitions), ["competition_name", "season_name"]]
    test_keys = {
        (str(row.competition_name), str(row.season_name))
        for row in test_rows.drop_duplicates().itertuples(index=False)
    }
    league_shift_keys = {key for key in test_keys if key[1] in source_seasons}
    league_season_shift_keys = test_keys - league_shift_keys

    for competition, season_specs in explicit_map.items():
        _validate_requested_competitions([competition], df_local["competition_name"].dropna().unique().tolist(), "test.league_season_shift.explicit")
        resolved_seasons = resolve_season_aliases(available_by_comp[str(competition)], _as_string_list(season_specs))
        league_season_shift_keys.update((str(competition), season_name) for season_name in resolved_seasons)

    overlap_with_source = source_keys & (year_shift_keys | league_shift_keys | league_season_shift_keys)
    if overlap_with_source:
        raise ValueError(f"Resolved source overlap with non-source split(s): {sorted(overlap_with_source)}")

    calib_keys = {
        (str(row.competition_name), str(row.season_name))
        for row in (
            df_local.loc[df_local["competition_name"].isin(calib_competitions), ["competition_name", "season_name"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
    }
    source_calib_overlap = source_keys & calib_keys
    if source_calib_overlap:
        raise ValueError(f"Resolved source overlap with calib split: {sorted(source_calib_overlap)}")

    all_keys = {
        (str(row.competition_name), str(row.season_name))
        for row in df_local[["competition_name", "season_name"]].drop_duplicates().itertuples(index=False)
    }
    claimed_test_keys = year_shift_keys | league_shift_keys | league_season_shift_keys
    residual_target_keys = all_keys - source_keys - claimed_test_keys

    return {
        "source_keys": source_keys,
        "calib_keys": calib_keys,
        "target_keys": residual_target_keys,
        "test_keys": {
            "league_shift": league_shift_keys,
            "year_shift": year_shift_keys,
            "league_season_shift": league_season_shift_keys,
        },
        "source_competitions": sorted({comp for comp, _ in source_keys}),
        "calib_competitions": sorted({comp for comp, _ in calib_keys}),
        "test_competitions": sorted({comp for comp, _ in claimed_test_keys}),
    }


def load_xy_competition_season_split(
    target_col: str,
    data_file: str | Path,
    key_candidates: Sequence[str],
    split_config: Mapping[str, Any],
    validation_frac: float = 0.2,
    random_state: int = 42,
) -> ResolvedCompetitionSeasonSplit:
    """Load data using competition-season split rules and stratify source by season."""
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be 'scores' or 'concedes'")

    total_start = perf_counter()
    logger.info(
        "load_xy_competition_season_split: loading data target=%s file=%s keys=%s",
        target_col,
        data_file,
        list(key_candidates),
    )
    load_start = perf_counter()
    df = read_h5_table(data_file=data_file, key_candidates=key_candidates).copy()
    logger.info(
        "load_xy_competition_season_split: loaded rows=%d cols=%d in %.2fs",
        len(df),
        len(df.columns),
        perf_counter() - load_start,
    )

    resolve_start = perf_counter()
    resolved = resolve_competition_season_split_spec(df=df, split_config=split_config)
    logger.info(
        "load_xy_competition_season_split: resolved split spec in %.2fs",
        perf_counter() - resolve_start,
    )

    key_membership = pd.Series(
        list(zip(df["competition_name"].astype(str), df["season_name"].astype(str))),
        index=df.index,
    )

    def _subset_frame(keys: set[tuple[str, str]]) -> pd.DataFrame:
        if not keys:
            empty_mask = pd.Series(False, index=df.index)
            return df.loc[empty_mask].copy()
        mask = key_membership.isin(keys)
        return df.loc[mask].copy()

    subset_start = perf_counter()
    df_source = _subset_frame(resolved["source_keys"])
    logger.info(
        "load_xy_competition_season_split: isolated source membership in %.2fs "
        "(source_rows=%d source_games=%d)",
        perf_counter() - subset_start,
        len(df_source),
        df_source["game_id"].nunique() if not df_source.empty else 0,
    )

    if df_source.empty:
        raise ValueError("Source split is empty after competition-season filtering")

    stratify_start = perf_counter()
    source_matches = (
        df_source[["game_id", "season_id"]]
        .drop_duplicates(subset=["game_id"])
        .reset_index(drop=True)
    )
    season_counts = source_matches["season_id"].value_counts()
    if (season_counts < 2).any():
        too_small = sorted(season_counts[season_counts < 2].index.tolist())
        raise ValueError(f"Cannot stratify source split by season_id; too few games in seasons: {too_small}")

    train_matches, val_matches = train_test_split(
        source_matches,
        test_size=validation_frac,
        stratify=source_matches["season_id"],
        random_state=random_state,
    )
    train_game_ids = set(train_matches["game_id"].tolist())
    val_game_ids = set(val_matches["game_id"].tolist())

    df_source_train = df_source[df_source["game_id"].isin(train_game_ids)].copy()
    df_source_val = df_source[df_source["game_id"].isin(val_game_ids)].copy()
    if df_source_train.empty:
        raise ValueError("Source train split is empty after season-stratified game sampling")
    if df_source_val.empty:
        raise ValueError("Source validation split is empty after season-stratified game sampling")
    logger.info(
        "load_xy_competition_season_split: stratified source train/val in %.2fs "
        "(train_rows=%d val_rows=%d train_games=%d val_games=%d)",
        perf_counter() - stratify_start,
        len(df_source_train),
        len(df_source_val),
        df_source_train["game_id"].nunique(),
        df_source_val["game_id"].nunique(),
    )

    frame_start = perf_counter()
    X_source_train, feature_cols, _ = prepare_vaep_xgb_features(
        df_source_train,
        target_col=target_col,
    )
    result = ResolvedCompetitionSeasonSplit(
        source_train=_build_split_frame(
            "source_train",
            df_source_train,
            target_col,
            feature_cols,
            X=X_source_train,
        ),
        source_val=_build_split_frame("source_val", df_source_val, target_col, feature_cols),
        feature_cols=list(feature_cols),
        split_config=dict(split_config),
        _lazy_split_loaders={
            "calib": lambda: _build_split_frame(
                "calib",
                _subset_frame(resolved["calib_keys"]),
                target_col,
                feature_cols,
            ),
            "target": lambda: _build_split_frame(
                "target",
                _subset_frame(resolved["target_keys"]),
                target_col,
                feature_cols,
            ),
            **{
                f"test_{name}": (
                    lambda split_name=name, split_keys=keys: _build_split_frame(
                        f"test_{split_name}",
                        _subset_frame(split_keys),
                        target_col,
                        feature_cols,
                    )
                )
                for name, keys in resolved["test_keys"].items()
            },
        },
    )
    logger.info(
        "load_xy_competition_season_split: materialized source frames in %.2fs; "
        "registered %d lazy non-source splits; total %.2fs",
        perf_counter() - frame_start,
        len(result._lazy_split_loaders),
        perf_counter() - total_start,
    )

    return result


def load_xy_source_calib_target_split(
    target_col: str,
    data_file: str | Path,
    key_candidates: Sequence[str],
    source_competitions: Sequence[str],
    calib_competitions: Sequence[str],
    target_competitions: Sequence[str],
    validation_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    list[str], list[str], list[str],
]:
    """Load data and split source into train/validation by match (stratified by league).

    This function:
    1. Loads source (train), calib (val), target (test) data by competition splits
    2. Splits source into train (1-validation_frac) and validation (validation_frac)
       by sampling matches, stratified by competition (league)

    Args:
        target_col: ``'scores'`` or ``'concedes'``.
        data_file: HDF5 file with a merged actions table.
        key_candidates: HDF5 keys to try (first match wins).
        source_competitions: Competition names for source (to be split into train/val).
        calib_competitions: Competition names for calibration (unused in training).
        target_competitions: Competition names for target (0-shot evaluation).
        validation_frac: Fraction of source matches to use for validation (default 0.2).
        random_state: Random seed for stratified match sampling.

    Returns:
        ``(X_train, y_train, X_val, y_val, X_calib, y_calib, X_target, y_target,
          source_comps, calib_comps, target_comps)``
    """
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be 'scores' or 'concedes'")

    df = read_h5_table(data_file=data_file, key_candidates=key_candidates).copy()
    X_all, feature_cols, _ = prepare_vaep_xgb_features(df, target_col=target_col)
    if "competition_name" not in df.columns:
        raise KeyError("Expected 'competition_name' column in dataset")
    if "game_id" not in df.columns:
        raise KeyError("Expected 'game_id' column in dataset")

    df["competition_name"] = df["competition_name"].astype(str)

    source_set = set(source_competitions)
    calib_set = set(calib_competitions)
    target_set = set(target_competitions)

    overlap = (source_set & calib_set) | (source_set & target_set) | (calib_set & target_set)
    if overlap:
        raise ValueError(f"Source/calib/target competition sets overlap: {sorted(overlap)}")

    # Extract splits
    df_source = df[df["competition_name"].isin(source_set)].copy()
    df_calib = df[df["competition_name"].isin(calib_set)].copy()
    df_target = df[df["competition_name"].isin(target_set)].copy()

    if df_source.empty:
        raise ValueError("Source split is empty after applying competition filters")
    if df_calib.empty:
        raise ValueError("Calib split is empty after applying competition filters")
    if df_target.empty:
        raise ValueError("Target split is empty after applying competition filters")

    # Split source by matches (game_id), stratified by competition_name
    # Build a match-level DataFrame with competition as stratification variable
    source_matches = (
        df_source[["game_id", "competition_name"]]
        .drop_duplicates(subset=["game_id"])
        .reset_index(drop=True)
    )

    # Perform stratified split at the match level
    train_matches, val_matches = train_test_split(
        source_matches,
        test_size=validation_frac,
        stratify=source_matches["competition_name"],
        random_state=random_state,
    )

    train_game_ids = set(train_matches["game_id"])
    val_game_ids = set(val_matches["game_id"])

    df_train = df_source[df_source["game_id"].isin(train_game_ids)].copy()
    df_val = df_source[df_source["game_id"].isin(val_game_ids)].copy()

    if df_train.empty:
        raise ValueError("Train split is empty after match sampling")
    if df_val.empty:
        raise ValueError("Validation split is empty after match sampling")
    X_train = X_all.loc[df_train.index, feature_cols].copy()
    y_train = normalize_xgb_labels(df_train[target_col])
    X_val = X_all.loc[df_val.index, feature_cols].copy()
    y_val = normalize_xgb_labels(df_val[target_col])
    X_calib = X_all.loc[df_calib.index, feature_cols].copy()
    y_calib = normalize_xgb_labels(df_calib[target_col])
    X_target = X_all.loc[df_target.index, feature_cols].copy()
    y_target = normalize_xgb_labels(df_target[target_col])

    return (
        X_train, y_train,
        X_val, y_val,
        X_calib, y_calib,
        X_target, y_target,
        sorted(source_set), sorted(calib_set), sorted(target_set),
    )


# ──────────────────────────────────────────────
# Few-shot / transfer-learning helpers
# ──────────────────────────────────────────────


def load_fewshot_splits(
    target_col: str,
    data_file: str | Path,
    key_candidates: Sequence[str],
    source_competitions: Sequence[str],
    target_competitions: Sequence[str],
    validation_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.Series,
    list[str],
]:
    """Load data for few-shot transfer experiments.

    Splits by competition into source and target, then splits source into
    train/val by ``game_id`` (stratified by league).  Returns ``game_id``
    alongside the target arrays so the caller can sub-sample games without
    leakage.  ``game_id`` is **never** included in feature columns.

    Args:
        target_col: ``'scores'`` or ``'concedes'``.
        data_file: HDF5 file with a merged actions table.
        key_candidates: HDF5 keys to try (first match wins).
        source_competitions: Competition names used for source training.
        target_competitions: Competition names for the target domain.
        validation_frac: Fraction of source matches held out for validation.
        random_state: Random seed for the source train/val split.

    Returns:
        ``(X_source_train, y_source_train,
          X_source_val, y_source_val,
          X_target, y_target,
          target_game_ids,    # pd.Series aligned with X_target rows
          feature_cols)``     # list[str] of feature column names
    """
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be 'scores' or 'concedes'")

    df = read_h5_table(data_file=data_file, key_candidates=key_candidates).copy()
    X_all, feature_cols, _ = prepare_vaep_xgb_features(df, target_col=target_col)
    for col_name in ("competition_name", "game_id"):
        if col_name not in df.columns:
            raise KeyError(f"Expected '{col_name}' column in dataset")
    df["competition_name"] = df["competition_name"].astype(str)

    source_set = set(source_competitions)
    target_set = set(target_competitions)
    overlap = source_set & target_set
    if overlap:
        raise ValueError(f"Source/target competition sets overlap: {sorted(overlap)}")

    df_source = df[df["competition_name"].isin(source_set)].copy()
    df_target = df[df["competition_name"].isin(target_set)].copy()

    if df_source.empty:
        raise ValueError("Source split is empty after applying competition filters")
    if df_target.empty:
        raise ValueError("Target split is empty after applying competition filters")

    # Split source by match, stratified by league
    source_matches = (
        df_source[["game_id", "competition_name"]]
        .drop_duplicates(subset=["game_id"])
        .reset_index(drop=True)
    )
    train_matches, val_matches = train_test_split(
        source_matches,
        test_size=validation_frac,
        stratify=source_matches["competition_name"],
        random_state=random_state,
    )
    df_train = df_source[df_source["game_id"].isin(set(train_matches["game_id"]))].copy()
    df_val = df_source[df_source["game_id"].isin(set(val_matches["game_id"]))].copy()

    def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = X_all.loc[frame.index, feature_cols].copy()
        y = normalize_xgb_labels(frame[target_col])
        return X, y

    X_train, y_train = _xy(df_train)
    X_val, y_val = _xy(df_val)
    X_target, y_target = _xy(df_target)

    # game_id Series aligned with X_target index
    target_game_ids = df_target.loc[X_target.index, "game_id"]

    return (
        X_train, y_train,
        X_val, y_val,
        X_target, y_target,
        target_game_ids,
        feature_cols,
    )


def sample_target_games(
    X_target: pd.DataFrame,
    y_target: pd.Series,
    target_game_ids: pd.Series,
    frac: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Sample a fraction of target *games* and split into few-shot / holdout.

    Sampling is at the ``game_id`` level (not action level) to prevent
    within-match leakage.

    Args:
        X_target: Feature DataFrame for the full target set.
        y_target: Labels for the full target set.
        target_game_ids: ``game_id`` Series aligned with ``X_target``.
        frac: Fraction of unique games to sample (e.g. 0.01, 0.05, 0.20).
        random_state: Random seed for reproducibility.

    Returns:
        ``(X_few, y_few, X_holdout, y_holdout)``
    """
    unique_games = target_game_ids.unique()
    n_sample = max(1, int(round(len(unique_games) * frac)))
    rng = np.random.RandomState(random_state)
    sampled_games = set(rng.choice(unique_games, size=n_sample, replace=False))

    few_mask = target_game_ids.isin(sampled_games)
    holdout_mask = ~few_mask

    return (
        X_target.loc[few_mask].copy(),
        y_target.loc[few_mask].copy(),
        X_target.loc[holdout_mask].copy(),
        y_target.loc[holdout_mask].copy(),
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
            X_val.reindex(columns=train_feature_cols),
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
