import os
import re
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import socceraction.spadl.config as spadlcfg
import socceraction.spadl.statsbomb as sb_spadl
import socceraction.vaep.features as vaep_features
import socceraction.vaep.labels as vaep_labels

warnings.simplefilter(action="ignore", category=FutureWarning)


def slug(text: str) -> str:
    """Return a filesystem-safe slug from free text."""
    text = str(text).strip().lower()
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


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
