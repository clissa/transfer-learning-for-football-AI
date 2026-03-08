"""VAEP feature extraction from SPADL actions.

Provides utilities to orient actions (play-left-to-right), compute VAEP
game-state features using socceraction's feature functions, and build a
merged dataset that retains all original metadata columns alongside the
new VAEP features.

Typical usage::

    from football_ai.data import read_h5_table
    from football_ai.features import build_vaep_dataset, save_vaep_dataset

    df = read_h5_table("data/spadl_full_data/major_leagues.h5", "full_data")
    df_vaep = build_vaep_dataset(df, nb_prev_actions=3)
    save_vaep_dataset(df_vaep, "data/vaep_data/major_leagues_vaep.h5")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from tqdm import tqdm

import socceraction.spadl as spadl
import socceraction.spadl.config as spadlcfg
import socceraction.vaep.features as vaep_features

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

SPADL_ACTION_COLS: list[str] = [
    "game_id",
    "action_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "type_id",
    "result_id",
    "bodypart_id",
]
"""The 13 SPADL columns required to compute VAEP game-state features."""

DEFAULT_FEATURE_FUNCTIONS: list[Callable] = [
    vaep_features.actiontype_onehot,
    vaep_features.result_onehot,
    vaep_features.bodypart_onehot,
    vaep_features.startlocation,
    vaep_features.endlocation,
    vaep_features.movement,
    vaep_features.space_delta,
    vaep_features.goalscore,
    vaep_features.time,
]
"""All 9 socceraction VAEP feature functions (full set)."""

LABEL_COLS: list[str] = ["scores", "concedes"]
"""VAEP label column names."""

ACTION_TYPE_MAP: dict[int, str] = dict(enumerate(spadlcfg.actiontypes))
"""SPADL action-type id → name mapping (used by orient_actions)."""


# ──────────────────────────────────────────────
# orient_actions
# ──────────────────────────────────────────────


def orient_actions(
    df_actions: pd.DataFrame,
    home_team_id_col: str = "home_team_id",
) -> pd.DataFrame:
    """Orient actions so the home team always attacks left-to-right.

    Centralises the ``play_left_to_right`` pattern that was previously
    duplicated in ``data.py`` (``build_vaep_dataset_for_competition_season``
    and ``build_labels``).

    Steps:

    1. Sort by ``(game_id, period_id, time_seconds, action_id)``.
    2. Add ``type_name`` from :data:`ACTION_TYPE_MAP` if missing (needed by
       ``goalscore`` in some socceraction versions).
    3. Apply :func:`socceraction.spadl.play_left_to_right` per game.

    Parameters
    ----------
    df_actions : pd.DataFrame
        Actions DataFrame.  Must contain at least :data:`SPADL_ACTION_COLS`
        **plus** *home_team_id_col*.
    home_team_id_col : str
        Column holding the home-team id for each action row.

    Returns
    -------
    pd.DataFrame
        Actions with coordinates flipped so the home team attacks left→right.
        The *home_team_id_col* column is **dropped** from the output.
    """
    actions = df_actions.copy()

    # Ensure action_id exists
    if "action_id" not in actions.columns:
        actions = actions.sort_values(
            ["game_id", "period_id", "time_seconds"]
        ).copy()
        actions["action_id"] = (
            actions.groupby("game_id").cumcount().astype(int)
        )

    # Ensure type_name exists (required by goalscore in some versions)
    if "type_name" not in actions.columns and "type_id" in actions.columns:
        actions["type_name"] = (
            actions["type_id"].map(ACTION_TYPE_MAP).astype(str)
        )

    # Sort deterministically
    actions = actions.sort_values(
        ["game_id", "period_id", "time_seconds", "action_id"]
    ).reset_index(drop=True)

    # Flip coordinates per game
    actions = (
        actions.groupby("game_id", group_keys=False)
        .apply(
            lambda ga: spadl.play_left_to_right(
                ga.drop(columns=[home_team_id_col]),
                home_team_id=int(ga[home_team_id_col].iloc[0]),
            )
        )
        .reset_index(drop=True)
    )

    return actions


# ──────────────────────────────────────────────
# compute_vaep_features
# ──────────────────────────────────────────────


def compute_vaep_features(
    actions: pd.DataFrame,
    nb_prev_actions: int = 3,
    feature_fns: Sequence[Callable] | None = None,
) -> pd.DataFrame:
    """Compute VAEP features for every action across all games.

    Parameters
    ----------
    actions : pd.DataFrame
        Oriented actions (output of :func:`orient_actions`).
        Must contain ``game_id`` and ``action_id``.
    nb_prev_actions : int
        Number of previous actions used to build game states (default 3).
    feature_fns : sequence of callables, optional
        Feature functions to apply to game states.  Each callable takes a
        list of DataFrames (game states) and returns a DataFrame.
        Defaults to :data:`DEFAULT_FEATURE_FUNCTIONS`.

    Returns
    -------
    pd.DataFrame
        One row per action with ``game_id``, ``action_id`` as the first two
        columns, followed by all computed feature columns (suffixed
        ``_a0``, ``_a1``, …, ``_a{nb_prev_actions-1}``).
    """
    if feature_fns is None:
        feature_fns = DEFAULT_FEATURE_FUNCTIONS

    game_ids = actions["game_id"].unique()
    logger.info(
        "Computing VAEP features for %d games (nb_prev_actions=%d, %d feature fns)",
        len(game_ids),
        nb_prev_actions,
        len(feature_fns),
    )

    features_parts: list[pd.DataFrame] = []

    for game_id in tqdm(game_ids, desc="VAEP features"):
        game_actions = actions[actions.game_id == game_id].reset_index(drop=True)
        if game_actions.empty:
            continue

        game_states = vaep_features.gamestates(game_actions, nb_prev_actions)
        features_game = pd.concat(
            [fn(game_states) for fn in feature_fns], axis=1
        )
        features_game.insert(0, "game_id", game_id)
        features_game.insert(
            1, "action_id", game_actions["action_id"].astype(int).values
        )
        features_parts.append(features_game)

    if not features_parts:
        logger.warning("No features computed (empty actions?)")
        return pd.DataFrame()

    return pd.concat(features_parts, ignore_index=True)


# ──────────────────────────────────────────────
# build_vaep_dataset  (end-to-end orchestrator)
# ──────────────────────────────────────────────


def build_vaep_dataset(
    df_full_data: pd.DataFrame,
    nb_prev_actions: int = 3,
    feature_fns: Sequence[Callable] | None = None,
) -> pd.DataFrame:
    """Build a merged VAEP-feature dataset from the ``full_data`` table.

    End-to-end orchestrator:

    1. Extract SPADL action columns + ``home_team_id`` from *df_full_data*.
    2. Orient actions via :func:`orient_actions`.
    3. Compute VAEP features via :func:`compute_vaep_features`.
    4. Join VAEP features back to **all** original *df_full_data* columns
       on ``(game_id, action_id)``.

    The output contains every original metadata column alongside the new
    VAEP feature columns (with suffixes ``_a0``, ``_a1``, …).

    Parameters
    ----------
    df_full_data : pd.DataFrame
        The ``full_data`` table loaded from an HDF5 SPADL dataset.  Must
        contain :data:`SPADL_ACTION_COLS`, ``home_team_id``, and the label
        columns ``scores`` / ``concedes``.
    nb_prev_actions : int
        Number of previous actions for context (default 3).
    feature_fns : sequence of callables, optional
        Feature functions (defaults to :data:`DEFAULT_FEATURE_FUNCTIONS`).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame: all original columns + VAEP feature columns.

    Notes
    -----
    - The raw coordinate columns in *df_full_data* (``start_x``, ``end_x``,
      etc.) are the **un-flipped** originals.  VAEP features (``start_x_a0``,
      ``type_id_a0``, …) are derived from the **flipped** (oriented) actions.
    - Labels ``scores`` and ``concedes`` are **reused** from *df_full_data*
      (already correctly computed by ``build_labels``).
    - The first ``nb_prev_actions - 1`` actions of each game will have NaN
      in past-action columns (``_a1``, ``_a2``); this is standard VAEP
      behaviour and is handled at training time.
    """
    logger.info(
        "build_vaep_dataset — input shape: %s, games: %d",
        df_full_data.shape,
        df_full_data["game_id"].nunique(),
    )

    # 1. Extract SPADL columns + home_team_id for orientation
    needed_cols = SPADL_ACTION_COLS + ["home_team_id"]
    missing = [c for c in needed_cols if c not in df_full_data.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    actions_subset = df_full_data[needed_cols].copy()

    # 2. Orient actions (play left-to-right)
    actions_oriented = orient_actions(actions_subset, home_team_id_col="home_team_id")

    # 3. Compute VAEP features
    vaep_feats = compute_vaep_features(
        actions_oriented, nb_prev_actions=nb_prev_actions, feature_fns=feature_fns
    )

    # 4. Join VAEP features to original full_data on (game_id, action_id)
    df_merged = df_full_data.merge(
        vaep_feats, on=["game_id", "action_id"], how="inner"
    )

    n_vaep_cols = len(vaep_feats.columns) - 2  # exclude game_id, action_id
    logger.info(
        "build_vaep_dataset — %d VAEP feature columns, output shape: %s",
        n_vaep_cols,
        df_merged.shape,
    )

    return df_merged


# ──────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────


def save_vaep_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
    key: str = "vaep_data",
) -> None:
    """Save a VAEP dataset to HDF5.

    Parameters
    ----------
    df : pd.DataFrame
        The merged VAEP dataset to persist.
    output_path : str | Path
        Destination HDF5 file path.
    key : str
        HDF5 key under which to store the table (default ``"vaep_data"``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    with pd.HDFStore(str(output_path), mode="w") as store:
        store.put(key, df, format="table")

    logger.info("Saved VAEP dataset → %s  (key=%r, shape=%s)", output_path, key, df.shape)
