"""Unified data module for football AI.

Contains ALL data logic: StatsBomb loading, SPADL conversion, VAEP features/labels,
DataFrame merge, HDF5 I/O, and generic H5 reading utilities.

This module absorbs the former ``utils.py`` and reusable logic previously inlined
in ``scripts/create_datasets.py``.
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import socceraction.spadl.config as spadlcfg
import socceraction.spadl.statsbomb as sb_spadl
import socceraction.vaep.features as vaep_features
import socceraction.vaep.labels as vaep_labels

warnings.simplefilter(action="ignore", category=FutureWarning)


# ──────────────────────────────────────────────
# SPADL lookup constants
# ──────────────────────────────────────────────

ACTION_TYPE_MAP: dict[int, str] = dict(enumerate(spadlcfg.actiontypes))
"""SPADL action-type id → name mapping."""

RESULT_MAP: dict[int, str] = dict(enumerate(spadlcfg.results))
"""SPADL result id → name mapping."""

BODYPART_MAP: dict[int, str] = dict(enumerate(spadlcfg.bodyparts))
"""SPADL body-part id → name mapping."""

FIELD_LENGTH: float = float(spadlcfg.field_length)
"""SPADL pitch length in metres."""

FIELD_WIDTH: float = float(spadlcfg.field_width)
"""SPADL pitch width in metres."""

STYLE_CATEGORICAL_COLS: list[str] = ["action_type", "result", "bodypart"]
"""Categorical columns added by :func:`build_styles_dataframe`."""

STYLE_DISPLACEMENT_COLS: list[str] = ["dx", "dy", "distance"]
"""Displacement columns added by :func:`build_styles_dataframe`."""

REQUIRED_STYLE_COLS: list[str] = [
    "game_id",
    "competition_name",
    "competition_id",
    "season_name",
    "season_id",
    "type_id",
    "result_id",
    "bodypart_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "scores",
    "concedes",
]
"""Columns required by :func:`build_styles_dataframe`."""


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Default columns included in the merged output table (``build_merged_output``).
REQUESTED_COLUMNS: list[str] = [
    "game_id",
    "original_event_id",
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
    "action_id",
    "season_id",
    "competition_id",
    "competition_stage",
    "game_day",
    "game_date",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "venue",
    "referee",
    "team_name",
    "player_name",
    "nickname",
    "jersey_number",
    "is_starter",
    "starting_position_id",
    "starting_position_name",
    "minutes_played",
    "competition_name",
    "country_name",
    "competition_gender",
    "season_name",
    "scores",
    "concedes",
]


# ──────────────────────────────────────────────
# Slug / dataset-key helpers
# ──────────────────────────────────────────────


def slug(text: str) -> str:
    """Return a filesystem-safe slug from free text."""
    text = str(text).strip().lower()
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def make_dataset_key(league: str, season: str) -> str:
    """Build a dataset key (e.g., 'serie_a_2015_2016') from league/season labels."""
    return f"{slug(league)}_{slug(season)}"


def split_dataset_key(dataset_key: str) -> tuple[str, str]:
    """Split dataset key into normalized (league, season_display) values."""
    m2 = re.match(r"^(.*)_(\d{4}_\d{4})$", dataset_key)
    m1 = re.match(r"^(.*)_(\d{4})$", dataset_key)
    if m2:
        league_slug, season_slug = m2.group(1), m2.group(2)
    elif m1:
        league_slug, season_slug = m1.group(1), m1.group(2)
    else:
        league_slug, season_slug = dataset_key, "unknown"
    return league_slug.replace("_", " "), season_slug.replace("_", "/")


# ──────────────────────────────────────────────
# Generic HDF5 I/O
# ──────────────────────────────────────────────


def read_h5_table(
    data_file: str | Path,
    key_candidates: str | Sequence[str],
) -> pd.DataFrame:
    """Read a table from an HDF5 file, trying *key_candidates* in order.

    Args:
        data_file: Path to the HDF5 file.
        key_candidates: HDF5 key(s) to try (first match wins).
            A single string is accepted for convenience.

    Returns:
        DataFrame loaded from the first matching key.

    Raises:
        FileNotFoundError: If *data_file* does not exist.
        KeyError: If none of the *key_candidates* are found.
    """
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Missing H5 dataset file: {data_file}")

    if isinstance(key_candidates, str):
        key_candidates = [key_candidates]

    with pd.HDFStore(str(data_file), mode="r") as store:
        available_keys = set(store.keys())
        for key in key_candidates:
            key_with_slash = f"/{key}"
            if key_with_slash in available_keys:
                print(f"Using H5 key '{key}' from {data_file}")
                return store.get(key_with_slash)

        raise KeyError(
            f"None of keys {list(key_candidates)} found in {data_file}. "
            f"Available keys: {sorted(available_keys)}"
        )


# ──────────────────────────────────────────────
# Legacy per-league HDF5 I/O (features_*.h5 / labels_*.h5)
# ──────────────────────────────────────────────


def list_available_dataset_keys(
    data_dir: str | os.PathLike[str],
) -> list[str]:
    """Return sorted dataset keys where both features and labels files exist."""
    data_dir = Path(data_dir)
    keys: set[str] = set()
    for feature_path in data_dir.glob("features_*.h5"):
        dataset_key = feature_path.stem[len("features_") :]
        labels_path = data_dir / f"labels_{dataset_key}.h5"
        if labels_path.exists():
            keys.add(dataset_key)
    return sorted(keys)


def load_dataset_tables(
    dataset_key: str,
    data_dir: str | os.PathLike[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and labels tables for one dataset key."""
    data_dir = Path(data_dir)
    features_path = data_dir / f"features_{dataset_key}.h5"
    labels_path = data_dir / f"labels_{dataset_key}.h5"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    features_df = pd.read_hdf(features_path, key="features")
    labels_df = pd.read_hdf(labels_path, key="labels")
    return features_df, labels_df


def load_xy(
    dataset_key: str,
    target_col: str,
    data_dir: str | os.PathLike[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Load X/y for one dataset key from features/labels HDF5 files."""
    if target_col not in {"scores", "concedes"}:
        raise ValueError("target_col must be either 'scores' or 'concedes'")

    df_features, df_labels = load_dataset_tables(dataset_key=dataset_key, data_dir=data_dir)
    df = df_features.merge(df_labels, on=["game_id", "action_id"], how="inner")

    feature_cols = [
        col
        for col in df.columns
        if col not in {"game_id", "action_id", "scores", "concedes"}
    ]
    X = df[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    y = df[target_col].astype(int)
    return X, y


def save_vaep_dataset(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    features_path: str | os.PathLike[str],
    labels_path: str | os.PathLike[str],
) -> tuple[str, str]:
    """Save features/labels tables to HDF5 files, overwriting existing files."""
    features_path = Path(features_path)
    labels_path = Path(labels_path)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    for file_path in (features_path, labels_path):
        if file_path.exists():
            file_path.unlink()

    with pd.HDFStore(str(features_path), mode="w") as store:
        store.put("features", features_df, format="table")

    with pd.HDFStore(str(labels_path), mode="w") as store:
        store.put("labels", labels_df, format="table")

    return str(features_path), str(labels_path)


def output_paths_for_competition_season(
    output_dir: str | os.PathLike[str],
    competition_name: str,
    season_name: str,
) -> tuple[Path, Path]:
    """Return features/labels output paths for one competition-season."""
    league_slug = slug(competition_name)
    season_slug = slug(season_name)
    output_dir = Path(output_dir)
    return (
        output_dir / f"features_{league_slug}_{season_slug}.h5",
        output_dir / f"labels_{league_slug}_{season_slug}.h5",
    )


# ──────────────────────────────────────────────
# StatsBomb loader helpers
# ──────────────────────────────────────────────


def make_statsbomb_loader(data_root: str | os.PathLike[str]) -> StatsBombLoader:
    """Create a StatsBomb local-data loader."""
    return StatsBombLoader(getter="local", root=str(data_root))


def list_competitions(loader: StatsBombLoader) -> pd.DataFrame:
    """Return available competition/season rows."""
    return loader.competitions().copy()


def resolve_competition_season_ids(
    competitions_df: pd.DataFrame,
    competition_name: str,
    season_name: str,
) -> tuple[int, int]:
    """Resolve a unique (competition_id, season_id) by names."""
    match = competitions_df[
        (competitions_df["competition_name"] == competition_name)
        & (competitions_df["season_name"] == season_name)
    ]
    if len(match) == 0:
        raise ValueError(
            f"No match for competition={competition_name!r}, season={season_name!r}"
        )
    if len(match) > 1:
        options = (
            match[
                [
                    "competition_id",
                    "season_id",
                    "competition_name",
                    "season_name",
                ]
            ]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            f"Ambiguous match for competition={competition_name!r}, season={season_name!r}. "
            f"Use selected_id_pairs instead. Options: {options}"
        )

    row = match.iloc[0]
    return int(row.competition_id), int(row.season_id)


def select_competition_seasons(
    competitions_df: pd.DataFrame,
    save_all_available: bool,
    selected_name_pairs: Iterable[tuple[str, str]] | None = None,
    selected_id_pairs: Iterable[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Build list of (competition_id, season_id) pairs from user selection."""
    selected_name_pairs = list(selected_name_pairs or [])
    selected_id_pairs = list(selected_id_pairs or [])

    if save_all_available:
        return [
            (int(row.competition_id), int(row.season_id))
            for row in competitions_df[["competition_id", "season_id"]]
            .drop_duplicates()
            .itertuples(index=False)
        ]

    if selected_name_pairs and selected_id_pairs:
        raise ValueError("Use either selected_name_pairs OR selected_id_pairs (not both).")

    if selected_name_pairs:
        return [
            resolve_competition_season_ids(competitions_df, competition_name, season_name)
            for (competition_name, season_name) in selected_name_pairs
        ]

    if selected_id_pairs:
        return [(int(competition_id), int(season_id)) for competition_id, season_id in selected_id_pairs]

    return []


# ──────────────────────────────────────────────
# SPADL & style helpers
# ──────────────────────────────────────────────


def get_spadl_type_from_id(spadl_map: dict[int, Any], type_id: Any) -> str:
    """Look up a human-readable SPADL label from a numeric id.

    Args:
        spadl_map: One of :data:`ACTION_TYPE_MAP`, :data:`RESULT_MAP`,
            :data:`BODYPART_MAP`.
        type_id: Integer id (or NaN).

    Returns:
        Name string, or ``"missing"``/``"unknown_..."`` for bad ids.
    """
    if pd.isna(type_id):
        return "missing"
    try:
        key = int(type_id)
    except (TypeError, ValueError):
        return f"unknown_{type_id=}"
    return str(spadl_map.get(key, f"unknown_{type_id=}"))


def parse_season_sort_key(season_name: str) -> tuple[int, str]:
    """Return a sortable (year, name) tuple for a season string.

    Extracts the first 4-digit year starting with 19xx or 20xx.
    Seasons without a parseable year sort last.

    >>> parse_season_sort_key("2015/2016")
    (2015, '2015/2016')
    """
    text = str(season_name)
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group(0)), text
    return 10**9, text


def pick_three_seasons(season_names: list[str]) -> list[str]:
    """Select oldest, middle, and newest season from a list.

    Args:
        season_names: Season name strings (may contain duplicates).

    Returns:
        Up to 3 unique season names sorted chronologically.
    """
    if not season_names:
        return []
    unique_sorted = sorted({str(s) for s in season_names}, key=parse_season_sort_key)
    if len(unique_sorted) <= 3:
        return unique_sorted
    oldest = unique_sorted[0]
    middle = unique_sorted[len(unique_sorted) // 2]
    newest = unique_sorted[-1]
    selected: list[str] = []
    for season in [oldest, middle, newest]:
        if season not in selected:
            selected.append(season)
    if len(selected) < 3:
        for season in unique_sorted:
            if season not in selected:
                selected.append(season)
            if len(selected) == 3:
                break
    return selected


def sample_series(series: pd.Series, max_rows: int, seed: int) -> pd.Series:
    """Subsample a Series (after dropping NaN) if it exceeds *max_rows*.

    Args:
        series: Input data.
        max_rows: Maximum number of rows to return.
        seed: Random seed.

    Returns:
        Subsampled (or full) Series with NaN removed.
    """
    cleaned = series.dropna()
    if len(cleaned) > max_rows:
        return cleaned.sample(n=max_rows, random_state=seed)
    return cleaned


def sample_dataframe(
    df: pd.DataFrame,
    frac: float,
    seed: int,
) -> pd.DataFrame:
    """Subsample a DataFrame by fraction.

    Args:
        df: Input DataFrame.
        frac: Fraction in (0, 1].  If >= 1, returns *df* unchanged.
        seed: Random seed.

    Returns:
        Subsampled DataFrame.
    """
    if frac >= 1.0:
        return df
    n = max(1, int(len(df) * frac))
    return df.sample(n=n, random_state=seed)


def build_styles_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich raw SPADL actions with human-readable labels and displacement cols.

    Adds columns: ``action_type``, ``result``, ``bodypart``, ``dx``, ``dy``,
    ``distance``.

    Args:
        df: DataFrame with at least the columns in :data:`REQUIRED_STYLE_COLS`.

    Returns:
        A new DataFrame with the original style columns plus enrichment.

    Raises:
        KeyError: If required columns are missing.
    """
    missing = [col for col in REQUIRED_STYLE_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for style analysis: {missing}")

    styles = df[REQUIRED_STYLE_COLS].copy()
    styles["action_type"] = styles["type_id"].apply(
        lambda x: get_spadl_type_from_id(ACTION_TYPE_MAP, x)
    )
    styles["result"] = styles["result_id"].apply(
        lambda x: get_spadl_type_from_id(RESULT_MAP, x)
    )
    styles["bodypart"] = styles["bodypart_id"].apply(
        lambda x: get_spadl_type_from_id(BODYPART_MAP, x)
    )
    styles["dx"] = styles["end_x"] - styles["start_x"]
    styles["dy"] = styles["end_y"] - styles["start_y"]
    styles["distance"] = np.sqrt(styles["dx"] ** 2 + styles["dy"] ** 2)
    return styles


def build_percentage_distribution(
    df_styles: pd.DataFrame,
    col: str,
) -> pd.DataFrame:
    """Compute value-count percentages for a categorical column.

    Args:
        df_styles: Style DataFrame.
        col: Column name.

    Returns:
        DataFrame with *col*, ``count_actions``, ``pct_actions``.
    """
    dist = (
        df_styles[col]
        .value_counts(dropna=False)
        .rename_axis(col)
        .reset_index(name="count_actions")
    )
    total = max(len(df_styles), 1)
    dist["pct_actions"] = (100 * dist["count_actions"] / total).round(4)
    return dist.sort_values("pct_actions", ascending=False)

