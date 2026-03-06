from __future__ import annotations

from pathlib import Path

import pandas as pd
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import socceraction.spadl.config as spadlcfg
import socceraction.spadl.statsbomb as sb_spadl
import socceraction.vaep.labels as vaep_labels


DATA_ROOT = Path("../open-data/data")
OUTPUT_DIR = Path("data/spadl_data_rich")

MAJOR_LEAGUES = [
    "La Liga",
    "Serie A",
    "Premier League",
    "1. Bundesliga",
    "Ligue 1",
    "Champions League",
    "UEFA Europa League",
]

WOMEN_LEAGUES = [
    "FA Women's Super League",
    "Women's World Cup",
    "UEFA Women's Euro",
]

REQUESTED_RICH_COLUMNS = [
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


def _as_series(obj: pd.Series | pd.DataFrame, name: str) -> pd.Series:
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


def _ensure_cols_from_index(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    if not set(required_cols).issubset(df.columns):
        df = df.reset_index()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}. Available: {list(df.columns)}")
    return df


def _drop_overlaps(right: pd.DataFrame, left_cols: pd.Index, join_keys: set[str]) -> pd.DataFrame:
    overlaps = (set(right.columns) & set(left_cols)) - join_keys
    if overlaps:
        right = right.drop(columns=sorted(overlaps))
    return right


def _select_competitions(competitions_df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    selected = competitions_df[competitions_df["competition_name"].isin(names)].copy()
    if selected.empty:
        raise ValueError(f"No competitions found for requested names: {names}")

    found = sorted(selected["competition_name"].dropna().unique().tolist())
    missing = sorted(set(names) - set(found))
    print(f"Selected competitions ({len(found)}): {found}")
    if missing:
        print(f"Missing competitions ({len(missing)}): {missing}")

    return selected.sort_values(["competition_name", "season_name"]).reset_index(drop=True)


def _load_games(
    loader: StatsBombLoader, selected_competitions: pd.DataFrame
) -> tuple[pd.DataFrame, list[tuple[int, int, str]]]:
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
            print(
                f"Could not load games for cid={competition_id}, sid={season_id}: {exc}"
            )

    df_games = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
    return df_games, failed


def _convert_games_to_actions(
    loader: StatsBombLoader, df_games: pd.DataFrame
) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
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


def _load_teams_players(
    loader: StatsBombLoader, game_ids: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, str]]]:
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
        pd.concat(all_teams, ignore_index=True).drop_duplicates(subset=["team_id"]).reset_index(drop=True)
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


def _build_labels(df_actions: pd.DataFrame, df_games: pd.DataFrame) -> pd.DataFrame:
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


def _build_rich_output(
    df_actions: pd.DataFrame,
    df_games: pd.DataFrame,
    df_teams: pd.DataFrame,
    df_players: pd.DataFrame,
    df_competitions: pd.DataFrame,
    df_labels: pd.DataFrame,
) -> pd.DataFrame:
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
    df_labels_m = _drop_overlaps(df_labels[label_cols].copy(), merged_df.columns, join_keys={"game_id", "action_id"})
    merged_df = pd.merge(merged_df, df_labels_m, on=["game_id", "action_id"], how="left")

    for col in REQUESTED_RICH_COLUMNS:
        if col not in merged_df.columns:
            merged_df[col] = pd.NA

    return merged_df[REQUESTED_RICH_COLUMNS].copy()


def _stringify_for_hdf(df_games: pd.DataFrame, df_players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = df_games.copy()
    players = df_players.copy()

    for col in ["competition_stage", "venue", "referee"]:
        if col in games.columns:
            games[col] = games[col].fillna("").astype(str)

    for col in ["player_name", "nickname", "starting_position_name"]:
        if col in players.columns:
            players[col] = players[col].fillna("").astype(str)

    return games, players


def build_group_dataset(
    loader: StatsBombLoader,
    competitions_df: pd.DataFrame,
    competition_names: list[str],
    output_file: Path,
) -> None:
    print(f"\n=== Building dataset: {output_file} ===")
    selected_competitions = _select_competitions(competitions_df, competition_names)

    df_games, failed_competitions = _load_games(loader, selected_competitions)
    if df_games.empty:
        raise ValueError(f"No games loaded for output {output_file}")

    df_actions, failed_actions = _convert_games_to_actions(loader, df_games)
    df_teams, df_players, failed_lineups = _load_teams_players(
        loader, df_games.game_id.astype(int).tolist()
    )
    df_labels = _build_labels(df_actions, df_games)
    df_rich = _build_rich_output(
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
        store.put("rich_actions", df_rich, format="table")

    print(f"Saved: {output_file}")
    print(f"actions={df_actions.shape} games={df_games.shape} teams={df_teams.shape} players={df_players.shape}")
    print(f"labels={df_labels.shape} rich_actions={df_rich.shape}")
    print(
        "failures: "
        f"competitions={len(failed_competitions)}, "
        f"actions={len(failed_actions)}, "
        f"lineups={len(failed_lineups)}"
    )


def main() -> int:
    loader = StatsBombLoader(getter="local", root=str(DATA_ROOT))
    competitions_df = loader.competitions().copy()

    build_group_dataset(
        loader=loader,
        competitions_df=competitions_df,
        competition_names=MAJOR_LEAGUES,
        output_file=OUTPUT_DIR / "major_leagues.h5",
    )
    build_group_dataset(
        loader=loader,
        competitions_df=competitions_df,
        competition_names=WOMEN_LEAGUES,
        output_file=OUTPUT_DIR / "women_leagues.h5",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


    # Read and inspect the major leagues dataset
    store = pd.HDFStore(str(OUTPUT_DIR / "major_leagues.h5"), mode="r")
    print("Available keys:", store.keys())

    for key in store.keys():
        df = store[key]
        print(f"\n{key}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(df.head())

    store.close()