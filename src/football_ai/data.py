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


# ──────────────────────────────────────────────
# EDA data-analysis helpers
# ──────────────────────────────────────────────


def classify_columns(
    df: pd.DataFrame,
    target_col: str,
    known_id_cols: set[str] | None = None,
    known_meta_cols: set[str] | None = None,
    leakage_suspect_cols: set[str] | None = None,
) -> dict[str, list[str]]:
    """Classify DataFrame columns into semantic groups.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        known_id_cols: Column names to assign to the ``"id"`` group.
        known_meta_cols: Column names to assign to the ``"meta"`` group.
        leakage_suspect_cols: Column names to flag as leakage suspects.

    Returns:
        Dict with keys ``"id"``, ``"meta"``, ``"target"``,
        ``"leakage_suspect"``, ``"numeric"``, ``"categorical"``,
        ``"datetime"``, ``"bool"``.
    """
    known_id_cols = known_id_cols or set()
    known_meta_cols = known_meta_cols or set()
    leakage_suspect_cols = leakage_suspect_cols or set()

    groups: dict[str, list[str]] = {
        "id": [],
        "meta": [],
        "target": [],
        "leakage_suspect": [],
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "bool": [],
    }
    for col in df.columns:
        if col == target_col:
            groups["target"].append(col)
        elif col in known_id_cols:
            groups["id"].append(col)
        elif col in known_meta_cols:
            groups["meta"].append(col)
        elif col in leakage_suspect_cols:
            groups["leakage_suspect"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            groups["datetime"].append(col)
        elif pd.api.types.is_bool_dtype(df[col]):
            groups["bool"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            groups["numeric"].append(col)
        else:
            groups["categorical"].append(col)
    return groups


def compute_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-column data-quality audit.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: ``column``, ``dtype``, ``n_missing``,
        ``pct_missing``, ``n_unique``, ``is_constant``, ``example_values``.
    """
    records = []
    for col in df.columns:
        s = df[col]
        n_miss = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        records.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_missing": n_miss,
                "pct_missing": round(100 * n_miss / max(len(df), 1), 2),
                "n_unique": n_unique,
                "is_constant": n_unique <= 1,
                "example_values": str(s.dropna().unique()[:5].tolist()),
            }
        )
    return pd.DataFrame(records).sort_values("pct_missing", ascending=False)


def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-column missing-value percentages.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with ``column`` and ``pct_missing`` for columns with
        any missing values, or an empty DataFrame if none.
    """
    miss = df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return pd.DataFrame(columns=["column", "pct_missing"])
    result = miss.reset_index()
    result.columns = ["column", "pct_missing"]
    result["pct_missing"] = (result["pct_missing"] * 100).round(2)
    return result


def detect_leakage_suspects(
    df: pd.DataFrame,
    target_col: str,
    leakage_suspect_cols: set[str],
    numeric_feature_cols: list[str],
    corr_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect columns suspected of target leakage.

    Checks known suspect columns and any numeric feature with
    |correlation| > *corr_threshold*.

    Args:
        df: Input DataFrame.
        target_col: Target column name.
        leakage_suspect_cols: Column names known to be suspects.
        numeric_feature_cols: Numeric feature column names to scan.
        corr_threshold: Absolute correlation threshold.

    Returns:
        List of dicts with ``column``, ``reason``, ``corr_with_target``.
    """
    y_float = df[target_col].astype(float)
    suspects: list[dict[str, Any]] = []

    for col in leakage_suspect_cols:
        if col == target_col or col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            corr = y_float.corr(s.astype(float))
            suspects.append(
                {"column": col, "reason": "known_suspect", "corr_with_target": round(corr, 4)}
            )

    for col in numeric_feature_cols:
        if col not in df.columns:
            continue
        corr = y_float.corr(df[col].astype(float))
        if abs(corr) > corr_threshold:
            suspects.append(
                {"column": col, "reason": "high_corr_with_target", "corr_with_target": round(corr, 4)}
            )

    return suspects


def build_zone_metrics_for_coordinates(
    df_styles: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> pd.DataFrame:
    """Aggregate action metrics into spatial zones defined by coordinate bins.

    Args:
        df_styles: Style DataFrame (must contain *x_col*, *y_col*,
            ``result``, ``scores``, ``distance``).
        x_col: Column for x-coordinate (e.g. ``"start_x"``).
        y_col: Column for y-coordinate (e.g. ``"start_y"``).
        x_edges: Bin edges along x-axis.
        y_edges: Bin edges along y-axis.

    Returns:
        DataFrame with one row per zone, including ``count_actions``,
        ``success_rate_pct``, ``score_rate_pct``, ``mean_distance``, etc.
    """
    full_idx = pd.MultiIndex.from_product(
        [range(len(y_edges) - 1), range(len(x_edges) - 1)],
        names=["y_zone_idx", "x_zone_idx"],
    )

    if len(df_styles) == 0:
        zone_rows = pd.DataFrame(index=full_idx).reset_index()
        zone_rows["count_actions"] = 0
        zone_rows["n_success"] = 0
        zone_rows["n_scores"] = 0
        zone_rows["success_rate_pct"] = np.nan
        zone_rows["score_rate_pct"] = np.nan
        zone_rows["mean_distance"] = np.nan
        zone_rows["pct_actions"] = 0.0
    else:
        x_upper = np.nextafter(x_edges[-1], x_edges[0])
        y_upper = np.nextafter(y_edges[-1], y_edges[0])

        coords = df_styles[[x_col, y_col, "result", "scores", "distance"]].copy()
        coords[x_col] = pd.to_numeric(coords[x_col], errors="coerce").clip(
            lower=x_edges[0], upper=x_upper
        )
        coords[y_col] = pd.to_numeric(coords[y_col], errors="coerce").clip(
            lower=y_edges[0], upper=y_upper
        )
        coords["is_success"] = coords["result"].astype(str).str.casefold().eq("success")
        coords["is_score"] = (
            pd.to_numeric(coords["scores"], errors="coerce").fillna(0).astype(float) > 0
        )
        coords["x_zone_idx"] = pd.cut(
            coords[x_col], bins=x_edges, labels=False, include_lowest=True, right=False
        )
        coords["y_zone_idx"] = pd.cut(
            coords[y_col], bins=y_edges, labels=False, include_lowest=True, right=False
        )
        valid = coords.dropna(subset=["x_zone_idx", "y_zone_idx"]).copy()
        valid["x_zone_idx"] = valid["x_zone_idx"].astype(int)
        valid["y_zone_idx"] = valid["y_zone_idx"].astype(int)

        grouped = valid.groupby(["y_zone_idx", "x_zone_idx"], dropna=False).agg(
            count_actions=("is_success", "size"),
            n_success=("is_success", "sum"),
            n_scores=("is_score", "sum"),
            success_rate=("is_success", "mean"),
            score_rate=("is_score", "mean"),
            mean_distance=("distance", "mean"),
        )
        zone_rows = grouped.reindex(full_idx).reset_index()
        zone_rows["count_actions"] = zone_rows["count_actions"].fillna(0).astype(int)
        zone_rows["n_success"] = zone_rows["n_success"].fillna(0).astype(int)
        zone_rows["n_scores"] = zone_rows["n_scores"].fillna(0).astype(int)
        zone_rows["success_rate_pct"] = (100 * zone_rows["success_rate"]).round(4)
        zone_rows["score_rate_pct"] = (100 * zone_rows["score_rate"]).round(4)
        zone_rows = zone_rows.drop(columns=["success_rate", "score_rate"])
        zone_rows["pct_actions"] = (
            100 * zone_rows["count_actions"] / max(len(df_styles), 1)
        ).round(4)

    zone_rows["x_min"] = zone_rows["x_zone_idx"].map(lambda idx: float(x_edges[idx]))
    zone_rows["x_max"] = zone_rows["x_zone_idx"].map(lambda idx: float(x_edges[idx + 1]))
    zone_rows["y_min"] = zone_rows["y_zone_idx"].map(lambda idx: float(y_edges[idx]))
    zone_rows["y_max"] = zone_rows["y_zone_idx"].map(lambda idx: float(y_edges[idx + 1]))
    return zone_rows


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _as_series(obj: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """Compatibility shim: socceraction labels may return Series or 1-col DataFrame."""
    if isinstance(obj, pd.DataFrame):
        if name in obj.columns and obj.shape[1] == 1:
            series = obj[name]
        elif obj.shape[1] == 1:
            series = obj.iloc[:, 0]
        else:
            series = obj.iloc[:, 0]
    else:
        series = obj
    return series.rename(name)


def _competition_row(
    competitions_df: pd.DataFrame,
    competition_id: int,
    season_id: int,
) -> pd.Series:
    row = competitions_df[
        (competitions_df.competition_id == competition_id)
        & (competitions_df.season_id == season_id)
    ]
    if len(row) != 1:
        raise ValueError(
            f"Cannot uniquely resolve competition_id={competition_id}, season_id={season_id}"
        )
    return row.iloc[0]


def _ensure_cols_from_index(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Reset index if required columns are missing from columns (but present in index)."""
    if not set(required_cols).issubset(df.columns):
        df = df.reset_index()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}. Available: {list(df.columns)}")
    return df


def _drop_overlaps(
    right: pd.DataFrame, left_cols: pd.Index, join_keys: set[str]
) -> pd.DataFrame:
    """Drop columns from *right* that already exist in *left_cols* (except join keys)."""
    overlaps = (set(right.columns) & set(left_cols)) - join_keys
    if overlaps:
        right = right.drop(columns=sorted(overlaps))
    return right


def _stringify_for_hdf(
    df_games: pd.DataFrame, df_players: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast problematic object columns to str so HDF5 serialisation works."""
    games = df_games.copy()
    players = df_players.copy()

    for col in ["competition_stage", "venue", "referee"]:
        if col in games.columns:
            games[col] = games[col].fillna("").astype(str)

    for col in ["player_name", "nickname", "starting_position_name"]:
        if col in players.columns:
            players[col] = players[col].fillna("").astype(str)

    return games, players


# ──────────────────────────────────────────────
# Legacy per-competition VAEP pipeline (features_*.h5 / labels_*.h5)
# ──────────────────────────────────────────────


def build_vaep_dataset_for_competition_season(
    loader: StatsBombLoader,
    competitions_df: pd.DataFrame,
    competition_id: int,
    season_id: int,
    nb_prev_actions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build VAEP features and labels for one competition-season."""
    competition = _competition_row(competitions_df, competition_id, season_id)
    competition_name = str(competition.competition_name)
    season_name = str(competition.season_name)

    games = loader.games(competition_id=competition_id, season_id=season_id).copy()
    if games.empty:
        raise ValueError("No games found for this competition/season")

    actions_parts: list[pd.DataFrame] = []
    failed_games: list[tuple[int, str]] = []

    for i, game in games.reset_index(drop=True).iterrows():
        game_id = int(game.game_id)
        home_team_id = int(game.home_team_id)
        try:
            events = loader.events(game_id)
            actions = sb_spadl.convert_to_actions(
                events,
                home_team_id=home_team_id,
                xy_fidelity_version=1,
                shot_fidelity_version=1,
            )
            actions_parts.append(actions)
        except Exception as exc:  # noqa: BLE001
            failed_games.append((game_id, str(exc)))

        if (i + 1) % 50 == 0:
            print(
                f"[{competition_name} | {season_name}] converted {i + 1}/{len(games)} games"
            )

    if not actions_parts:
        raise ValueError("No actions were generated (all games failed?)")

    actions = pd.concat(actions_parts, ignore_index=True)

    if "action_id" not in actions.columns:
        actions = actions.sort_values(["game_id", "period_id", "time_seconds"]).copy()
        actions["action_id"] = actions.groupby("game_id").cumcount().astype(int)

    if "type_name" not in actions.columns and "type_id" in actions.columns:
        actiontype_map = dict(enumerate(spadlcfg.actiontypes))
        actions["type_name"] = actions["type_id"].map(actiontype_map).astype(str)

    actions = actions.merge(games[["game_id", "home_team_id"]], on="game_id", how="left")
    actions = actions.sort_values(
        ["game_id", "period_id", "time_seconds", "action_id"]
    ).reset_index(drop=True)
    actions = (
        actions.groupby("game_id", group_keys=False)
        .apply(
            lambda game_actions: spadl.play_left_to_right(
                game_actions.drop(columns=["home_team_id"]),
                home_team_id=int(game_actions.home_team_id.iloc[0]),
            )
        )
        .reset_index(drop=True)
    )

    feature_functions = [
        vaep_features.result_onehot,
        vaep_features.startlocation,
        vaep_features.movement,
        vaep_features.goalscore,
    ]

    features_parts: list[pd.DataFrame] = []
    labels_parts: list[pd.DataFrame] = []

    for game_id in games.game_id.astype(int).tolist():
        game_actions = actions[actions.game_id == game_id].reset_index(drop=True)
        if game_actions.empty:
            continue

        game_states = vaep_features.gamestates(game_actions, nb_prev_actions)
        features_game = pd.concat([fn(game_states) for fn in feature_functions], axis=1)
        features_game.insert(0, "game_id", game_id)
        features_game.insert(1, "action_id", game_actions.action_id.astype(int).values)
        features_parts.append(features_game)

        y_scores = _as_series(vaep_labels.scores(game_actions), "scores")
        y_concedes = _as_series(vaep_labels.concedes(game_actions), "concedes")
        labels_game = pd.concat([y_scores, y_concedes], axis=1)
        labels_game.insert(0, "game_id", game_id)
        labels_game.insert(1, "action_id", game_actions.action_id.astype(int).values)
        labels_parts.append(labels_game)

    features_df = pd.concat(features_parts, ignore_index=True) if features_parts else pd.DataFrame()
    labels_df = pd.concat(labels_parts, ignore_index=True) if labels_parts else pd.DataFrame()

    meta = {
        "competition_id": int(competition_id),
        "season_id": int(season_id),
        "competition_name": competition_name,
        "season_name": season_name,
        "games_total": int(len(games)),
        "games_failed": int(len(failed_games)),
        "failed_games": failed_games,
    }
    return features_df, labels_df, meta


def build_and_save_vaep_for_competition_season(
    loader: StatsBombLoader,
    competitions_df: pd.DataFrame,
    output_dir: str | os.PathLike[str],
    competition_id: int,
    season_id: int,
    nb_prev_actions: int = 3,
) -> tuple[str, str, dict]:
    """Convenience function: build VAEP dataset and persist it for one competition-season."""
    competition = _competition_row(competitions_df, competition_id, season_id)
    competition_name = str(competition.competition_name)
    season_name = str(competition.season_name)

    features_path, labels_path = output_paths_for_competition_season(
        output_dir,
        competition_name,
        season_name,
    )

    print(
        f"\n=== {competition_name} | {season_name} "
        f"(cid={competition_id}, sid={season_id}) ==="
    )
    print(f"-> {features_path}")
    print(f"-> {labels_path}")

    features_df, labels_df, meta = build_vaep_dataset_for_competition_season(
        loader=loader,
        competitions_df=competitions_df,
        competition_id=competition_id,
        season_id=season_id,
        nb_prev_actions=nb_prev_actions,
    )

    save_vaep_dataset(features_df, labels_df, features_path, labels_path)

    print(f"Saved features: {features_df.shape}")
    print(f"Saved labels:   {labels_df.shape}")
    if meta["games_failed"]:
        print(f"Warning: {meta['games_failed']} games failed conversion")

    return str(features_path), str(labels_path), meta


# ──────────────────────────────────────────────
# Multi-league dataset pipeline (merged H5)
# ──────────────────────────────────────────────


def select_competitions(
    competitions_df: pd.DataFrame,
    names: list[str] | None,
) -> pd.DataFrame:
    """Select competitions by name. If *names* is None, return all.

    Args:
        competitions_df: Full competitions table from StatsBombLoader.
        names: Competition names to keep (None -> keep all).

    Returns:
        Filtered and sorted competitions DataFrame.
    """
    if names is None:
        selected = competitions_df.copy()
        print(f"Selected all competitions ({len(selected)})")
    else:
        selected = competitions_df[competitions_df["competition_name"].isin(names)].copy()
        if selected.empty:
            raise ValueError(f"No competitions found for requested names: {names}")

        found = sorted(selected["competition_name"].dropna().unique().tolist())
        missing = sorted(set(names) - set(found))
        print(f"Selected competitions ({len(found)}): {found}")
        if missing:
            print(f"Missing competitions ({len(missing)}): {missing}")

    return selected.sort_values(["competition_name", "season_name"]).reset_index(drop=True)


def load_games(
    loader: StatsBombLoader, selected_competitions: pd.DataFrame
) -> tuple[pd.DataFrame, list[tuple[int, int, str]]]:
    """Load games for all selected competition/season pairs.

    Returns:
        (df_games, failed) where *failed* is a list of (cid, sid, error_msg) tuples.
    """
    all_games: list[pd.DataFrame] = []
    failed: list[tuple[int, int, str]] = []

    for comp in selected_competitions.itertuples(index=False):
        competition_id = int(comp.competition_id)
        season_id = int(comp.season_id)
        try:
            games = loader.games(competition_id=competition_id, season_id=season_id).copy()
            all_games.append(games)
            print(
                f"Loaded {len(games):4d} games for "
                f"{comp.competition_name} | {comp.season_name} "
                f"(cid={competition_id}, sid={season_id})"
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((competition_id, season_id, str(exc)))
            print(f"Could not load games for cid={competition_id}, sid={season_id}: {exc}")

    df_games = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
    return df_games, failed


def convert_games_to_actions(
    loader: StatsBombLoader, df_games: pd.DataFrame
) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Convert raw StatsBomb events to SPADL actions for every game.

    Returns:
        (df_actions, failed_games) where *failed_games* lists (game_id, error_msg).
    """
    all_actions: list[pd.DataFrame] = []
    failed_games: list[tuple[int, str]] = []

    games_iter = df_games.reset_index(drop=True)
    for i, game in games_iter.iterrows():
        game_id = int(game.game_id)
        home_team_id = int(game.home_team_id)
        try:
            events = loader.events(game_id)
            actions = sb_spadl.convert_to_actions(
                events,
                home_team_id=home_team_id,
                xy_fidelity_version=1,
                shot_fidelity_version=1,
            )
            all_actions.append(actions)
        except Exception as exc:  # noqa: BLE001
            failed_games.append((game_id, str(exc)))

        if (i + 1) % 100 == 0:
            print(f"Converted {i + 1}/{len(df_games)} games")

    df_actions = pd.concat(all_actions, ignore_index=True) if all_actions else pd.DataFrame()
    if not df_actions.empty and "action_id" not in df_actions.columns:
        df_actions = df_actions.sort_values(["game_id", "period_id", "time_seconds"]).copy()
        df_actions["action_id"] = df_actions.groupby("game_id").cumcount().astype(int)

    return df_actions, failed_games


def load_teams_players(
    loader: StatsBombLoader, game_ids: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, str]]]:
    """Load teams and players tables for a list of game ids.

    Returns:
        (df_teams, df_players, failed) where *failed* lists (game_id, error_msg).
    """
    all_teams: list[pd.DataFrame] = []
    all_players: list[pd.DataFrame] = []
    failed: list[tuple[int, str]] = []

    for i, game_id in enumerate(game_ids):
        try:
            all_teams.append(loader.teams(game_id))
            all_players.append(loader.players(game_id))
        except Exception as exc:  # noqa: BLE001
            failed.append((game_id, str(exc)))

        if (i + 1) % 50 == 0:
            print(f"Loaded teams/players for {i + 1}/{len(game_ids)} games")

    df_teams = (
        pd.concat(all_teams, ignore_index=True)
        .drop_duplicates(subset=["team_id"])
        .reset_index(drop=True)
        if all_teams
        else pd.DataFrame()
    )
    df_players = (
        pd.concat(all_players, ignore_index=True)
        .drop_duplicates(subset=["player_id"])
        .reset_index(drop=True)
        if all_players
        else pd.DataFrame()
    )
    return df_teams, df_players, failed


def build_labels(df_actions: pd.DataFrame, df_games: pd.DataFrame) -> pd.DataFrame:
    """Compute VAEP labels (scores, concedes) for every action.

    Args:
        df_actions: SPADL actions with columns [game_id, action_id, period_id,
            time_seconds, type_id, ...].
        df_games: Games table with [game_id, home_team_id].

    Returns:
        DataFrame with columns [game_id, action_id, scores, concedes].
    """
    if df_actions.empty:
        return pd.DataFrame(columns=["game_id", "action_id", "scores", "concedes"])

    actions = df_actions.copy()
    games = df_games.copy()

    if "action_id" not in actions.columns:
        actions = actions.sort_values(["game_id", "period_id", "time_seconds"]).copy()
        actions["action_id"] = actions.groupby("game_id").cumcount().astype(int)

    if "type_name" not in actions.columns and "type_id" in actions.columns:
        actiontype_map = dict(enumerate(spadlcfg.actiontypes))
        actions["type_name"] = actions["type_id"].map(actiontype_map).astype(str)

    actions = actions.merge(games[["game_id", "home_team_id"]], on="game_id", how="left")
    actions = actions.sort_values(
        ["game_id", "period_id", "time_seconds", "action_id"]
    ).reset_index(drop=True)
    actions = (
        actions.groupby("game_id", group_keys=False)
        .apply(
            lambda game_actions: spadl.play_left_to_right(
                game_actions.drop(columns=["home_team_id"]),
                home_team_id=int(game_actions.home_team_id.iloc[0]),
            )
        )
        .reset_index(drop=True)
    )

    labels_parts: list[pd.DataFrame] = []
    for game_id in games.game_id.astype(int).tolist():
        game_actions = actions[actions.game_id == game_id].reset_index(drop=True)
        if game_actions.empty:
            continue

        y_scores = _as_series(vaep_labels.scores(game_actions), "scores")
        y_concedes = _as_series(vaep_labels.concedes(game_actions), "concedes")
        labels_game = pd.concat([y_scores, y_concedes], axis=1)
        labels_game.insert(0, "game_id", game_id)
        labels_game.insert(1, "action_id", game_actions.action_id.astype(int).values)
        labels_parts.append(labels_game)

    return pd.concat(labels_parts, ignore_index=True) if labels_parts else pd.DataFrame()


def build_merged_output(
    df_actions: pd.DataFrame,
    df_games: pd.DataFrame,
    df_teams: pd.DataFrame,
    df_players: pd.DataFrame,
    df_competitions: pd.DataFrame,
    df_labels: pd.DataFrame,
    requested_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Merge all tables into one wide actions DataFrame with full metadata.

    Args:
        df_actions: SPADL actions.
        df_games: Games.
        df_teams: Teams.
        df_players: Players.
        df_competitions: Selected competitions.
        df_labels: Labels (scores, concedes).
        requested_columns: Columns to include in the output. If ``None``, uses
            the module-level :data:`REQUESTED_COLUMNS` default.

    Returns:
        DataFrame with the requested columns.
    """
    if requested_columns is None:
        requested_columns = REQUESTED_COLUMNS

    df_actions = _ensure_cols_from_index(df_actions, ["game_id", "action_id"])
    df_games = _ensure_cols_from_index(df_games, ["game_id"])
    df_teams = _ensure_cols_from_index(df_teams, ["team_id"])
    df_players = _ensure_cols_from_index(df_players, ["player_id"])
    df_competitions = _ensure_cols_from_index(df_competitions, ["competition_id", "season_id"])
    df_labels = _ensure_cols_from_index(df_labels, ["game_id", "action_id"])

    merged_df = pd.merge(df_actions, df_games, on="game_id", how="left")

    df_teams_m = _drop_overlaps(df_teams, merged_df.columns, join_keys={"team_id"})
    merged_df = pd.merge(merged_df, df_teams_m, on="team_id", how="left")

    df_players_m = _drop_overlaps(df_players, merged_df.columns, join_keys={"player_id"})
    merged_df = pd.merge(merged_df, df_players_m, on="player_id", how="left")

    df_comp_m = _drop_overlaps(
        df_competitions,
        merged_df.columns,
        join_keys={"competition_id", "season_id"},
    )
    merged_df = pd.merge(merged_df, df_comp_m, on=["competition_id", "season_id"], how="left")

    label_cols = [c for c in df_labels.columns if c in {"game_id", "action_id", "scores", "concedes"}]
    df_labels_m = _drop_overlaps(
        df_labels[label_cols].copy(), merged_df.columns, join_keys={"game_id", "action_id"}
    )
    merged_df = pd.merge(merged_df, df_labels_m, on=["game_id", "action_id"], how="left")

    for col in requested_columns:
        if col not in merged_df.columns:
            merged_df[col] = pd.NA

    return merged_df[requested_columns].copy()


def build_and_save_dataset(
    loader: StatsBombLoader,
    competitions_df: pd.DataFrame,
    league_names: list[str] | None,
    output_file: Path,
    nb_prev_actions: int = 3,
) -> None:
    """Build the full dataset for the given leagues and write it to *output_file*.

    Produces a single HDF5 file with keys: ``actions``, ``games``, ``teams``,
    ``players``, ``competitions``, ``labels``, ``full_data``.

    Args:
        loader: StatsBomb data loader.
        competitions_df: All available competitions.
        league_names: Competition names to include (None -> all available).
        output_file: Path to the output HDF5 file.
        nb_prev_actions: Number of previous actions to store as context (default 3).
    """
    print(f"\n=== Building dataset: {output_file} ===")
    selected_competitions = select_competitions(competitions_df, league_names)

    df_games, failed_competitions = load_games(loader, selected_competitions)
    if df_games.empty:
        raise ValueError(f"No games loaded for output {output_file}")

    df_actions, failed_actions = convert_games_to_actions(loader, df_games)
    df_teams, df_players, failed_lineups = load_teams_players(
        loader, df_games.game_id.astype(int).tolist()
    )
    df_labels = build_labels(df_actions, df_games)
    df_merged = build_merged_output(
        df_actions=df_actions,
        df_games=df_games,
        df_teams=df_teams,
        df_players=df_players,
        df_competitions=selected_competitions,
        df_labels=df_labels,
    )

    df_games_to_save, df_players_to_save = _stringify_for_hdf(df_games, df_players)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    with pd.HDFStore(str(output_file), mode="w") as store:
        store.put("actions", df_actions, format="table")
        store.put("games", df_games_to_save, format="fixed")
        store.put("teams", df_teams, format="fixed")
        store.put("players", df_players_to_save, format="table")
        store.put("competitions", selected_competitions, format="fixed")
        store.put("labels", df_labels, format="table")
        store.put("full_data", df_merged, format="table")

    print(f"\nSaved: {output_file}")
    print(
        f"  actions={df_actions.shape}  games={df_games.shape}  "
        f"teams={df_teams.shape}  players={df_players.shape}"
    )
    print(f"  labels={df_labels.shape}  full_data={df_merged.shape}")
    print(
        f"  failures: competitions={len(failed_competitions)}, "
        f"actions={len(failed_actions)}, lineups={len(failed_lineups)}"
    )
    print(f"  nb_prev_actions={nb_prev_actions} (stored for reference)")
