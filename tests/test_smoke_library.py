"""Smoke tests for football_ai library functions.

Fast, CPU-only, no external data files required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from football_ai.data import (
    make_dataset_key,
    slug,
    split_dataset_key,
    get_spadl_type_from_id,
    parse_season_sort_key,
    pick_three_seasons,
    sample_series,
    sample_dataframe,
    build_styles_dataframe,
    build_percentage_distribution,
    classify_columns,
    compute_data_quality,
    compute_missingness,
    detect_leakage_suspects,
    ACTION_TYPE_MAP,
    RESULT_MAP,
    BODYPART_MAP,
    REQUIRED_STYLE_COLS,
)
from football_ai.evaluation import evaluate_binary, get_positive_class_scores
from football_ai.training import (
    build_scores_xgb_feature_frame,
    build_sklearn_model,
    build_preprocessor,
    load_xy_competition_season_split,
    prepare_vaep_xgb_features,
    preprocess_split,
    resolve_competition_season_split_spec,
    resolve_season_aliases,
)


# ──────────────────────────────────────────────
# data.py
# ──────────────────────────────────────────────


def test_slug():
    assert slug("La Liga") == "la_liga"
    assert slug("1. Bundesliga") == "1_bundesliga"
    assert slug("Champions League 2015/2016") == "champions_league_2015_2016"


def test_make_dataset_key():
    assert make_dataset_key("La Liga", "2015/2016") == "la_liga_2015_2016"


def test_split_dataset_key():
    league, season = split_dataset_key("la_liga_2015_2016")
    assert league == "la liga"
    assert season == "2015/2016"


# ──────────────────────────────────────────────
# training.py
# ──────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ["rf", "logreg", "mlp"])
def test_build_sklearn_model(model_name):
    model = build_sklearn_model(model_name)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


# ──────────────────────────────────────────────
# evaluation.py
# ──────────────────────────────────────────────


def test_evaluate_binary():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)
    metrics = evaluate_binary(y_proba, y_true)
    expected_keys = {
        "rows",
        "positive_rate",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
        "logloss",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["rows"] == 200.0


def test_get_positive_class_scores():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((100, 5)), columns=[f"f{i}" for i in range(5)])
    y = rng.integers(0, 2, size=100)
    model = LogisticRegression(max_iter=100, random_state=42).fit(X, y)
    scores = get_positive_class_scores(model, X)
    assert scores.shape == (100,)
    assert 0 <= scores.min() and scores.max() <= 1


# ──────────────────────────────────────────────
# data.py – new EDA helpers
# ──────────────────────────────────────────────


def test_get_spadl_type_from_id():
    assert get_spadl_type_from_id(ACTION_TYPE_MAP, 0) == ACTION_TYPE_MAP[0]
    assert get_spadl_type_from_id(ACTION_TYPE_MAP, 9999) == "unknown_type_id=9999"
    assert get_spadl_type_from_id(ACTION_TYPE_MAP, None) == "missing"


def test_parse_season_sort_key():
    y, _ = parse_season_sort_key("2015/2016")
    assert y == 2015
    y2, _ = parse_season_sort_key("no-year")
    assert y2 == 10**9  # sentinel for unknown


def test_pick_three_seasons():
    seasons = ["2010/2011", "2012/2013", "2014/2015", "2016/2017", "2018/2019"]
    result = pick_three_seasons(seasons)
    assert len(result) == 3
    assert result[0] == "2010/2011"   # oldest
    assert result[-1] == "2018/2019"  # newest


def test_sample_series():
    s = pd.Series(range(1000))
    out = sample_series(s, max_rows=50, seed=42)
    assert len(out) == 50


def test_sample_dataframe():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    out = sample_dataframe(df, frac=0.1, seed=42)
    assert len(out) == 10


@pytest.fixture
def _tiny_spadl_df():
    """Minimal DataFrame matching REQUIRED_STYLE_COLS layout."""
    n = 50
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "game_id": rng.integers(1, 4, size=n),
        "competition_name": rng.choice(["La Liga", "Serie A"], size=n),
        "competition_id": rng.choice([1, 2], size=n),
        "season_name": rng.choice(["2015/2016", "2016/2017"], size=n),
        "season_id": rng.choice([10, 11], size=n),
        "type_id": rng.integers(0, 5, size=n),
        "result_id": rng.integers(0, 3, size=n),
        "bodypart_id": rng.integers(0, 3, size=n),
        "start_x": rng.uniform(0, 105, size=n),
        "start_y": rng.uniform(0, 68, size=n),
        "end_x": rng.uniform(0, 105, size=n),
        "end_y": rng.uniform(0, 68, size=n),
        "scores": rng.integers(0, 2, size=n),
        "concedes": rng.integers(0, 2, size=n),
    })


def test_build_styles_dataframe(_tiny_spadl_df):
    styles = build_styles_dataframe(_tiny_spadl_df)
    for col in ("action_type", "result", "bodypart", "dx", "dy", "distance"):
        assert col in styles.columns
    assert len(styles) == len(_tiny_spadl_df)


def test_build_percentage_distribution(_tiny_spadl_df):
    styles = build_styles_dataframe(_tiny_spadl_df)
    dist = build_percentage_distribution(styles, "action_type")
    assert "count_actions" in dist.columns
    assert "pct_actions" in dist.columns


def test_classify_columns():
    df = pd.DataFrame({
        "game_id": [1, 2],
        "scores": [0, 1],
        "start_x": [1.0, 2.0],
        "type_id": [0, 1],
        "competition_name": ["A", "B"],
    })
    groups = classify_columns(
        df,
        target_col="scores",
        known_id_cols={"game_id"},
        known_meta_cols={"competition_name"},
        leakage_suspect_cols=set(),
    )
    assert "scores" in groups["target"]
    assert "game_id" in groups["id"]
    assert "start_x" in groups["numeric"]


def test_compute_data_quality():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    qdf = compute_data_quality(df)
    assert "pct_missing" in qdf.columns
    assert len(qdf) == 2


def test_compute_missingness():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    mdf = compute_missingness(df)
    assert len(mdf) == 1  # only column 'a' has missing
    assert mdf.iloc[0]["column"] == "a"


def test_detect_leakage_suspects():
    rng = np.random.default_rng(42)
    n = 200
    scores = rng.integers(0, 2, size=n)
    df = pd.DataFrame({
        "scores": scores,
        "leaky_col": scores + rng.normal(0, 0.01, size=n),
        "clean_col": rng.uniform(0, 1, size=n),
    })
    suspects = detect_leakage_suspects(
        df,
        target_col="scores",
        leakage_suspect_cols={"leaky_col"},
        numeric_feature_cols=["clean_col"],
        corr_threshold=0.5,
    )
    assert any(s["column"] == "leaky_col" for s in suspects)


# ──────────────────────────────────────────────
# training.py – preprocessing
# ──────────────────────────────────────────────


def test_build_preprocessor_and_preprocess():
    rng = np.random.default_rng(42)
    n = 50
    X = pd.DataFrame({
        "num_a": rng.normal(0, 1, size=n),
        "num_b": rng.normal(5, 2, size=n),
        "cat_x": rng.choice([0, 1, 2], size=n),
        "cat_y": rng.choice([10, 20], size=n),
    })
    num = ["num_a", "num_b"]
    cat = ["cat_x", "cat_y"]

    scaler, encoder, feature_names = build_preprocessor(X, num, cat, min_frequency=1)
    assert scaler is not None
    assert encoder is not None
    assert len(feature_names) > len(num)  # encoded categoricals add columns

    X_preproc = preprocess_split(X, num, cat, scaler, encoder)
    assert X_preproc.shape[0] == n
    assert X_preproc.shape[1] == len(feature_names)


@pytest.fixture
def scores_feature_refactor_df():
    rows: list[dict[str, object]] = []
    team_sequence = [11, 11, 11, 22, 22]
    action_slots = [
        ("pass", None, None),
        ("pass", "pass", None),
        ("shot", "pass", "pass"),
        ("pass", "shot", "pass"),
        ("shot", "pass", "shot"),
    ]
    result_slots = [
        ("success", None, None),
        ("success", "success", None),
        ("fail", "success", "success"),
        ("success", "fail", "success"),
        ("success", "success", "fail"),
    ]
    bodypart_slots = [
        ("foot", None, None),
        ("foot", "foot", None),
        ("head", "foot", "foot"),
        ("foot", "head", "foot"),
        ("head", "foot", "head"),
    ]

    for action_id, team_id in enumerate(team_sequence):
        row: dict[str, object] = {
            "game_id": 1,
            "action_id": action_id,
            "team_id": team_id,
            "competition_name": "La Liga",
            "season_name": "2015/2016",
            "season_id": 101,
            "scores": int(action_id in {2, 4}),
            "concedes": 0,
            "start_x_a0": 60.0 + action_id,
            "start_y_a0": 34.0 + (action_id % 2),
            "end_x_a0": 70.0 + action_id,
            "end_y_a0": 33.0 + (action_id % 2),
            "start_x_a1": 58.0 + action_id,
            "start_y_a1": 33.0,
            "start_x_a2": 56.0 + action_id,
            "start_y_a2": 32.0,
            "end_x_a1": 67.0 + action_id,
            "end_y_a1": 33.0,
            "end_x_a2": 65.0 + action_id,
            "end_y_a2": 32.0,
            "dx_a0": 10.0,
            "dy_a0": -1.0,
            "movement_a0": 10.05,
            "dx_a1": 9.0,
            "dy_a1": 0.0,
            "movement_a1": 9.0,
            "dx_a2": 8.0,
            "dy_a2": 1.0,
            "movement_a2": 8.06,
            "dx_a01": 2.0,
            "dy_a01": 1.0,
            "mov_a01": 2.24,
            "dx_a02": 4.0,
            "dy_a02": 2.0,
            "mov_a02": 4.47,
            "goalscore_team": 0,
            "goalscore_opponent": 0,
            "goalscore_diff": 0,
            "period_id_a0": 1,
            "time_seconds_a0": 30.0 * action_id,
            "time_seconds_overall_a0": 30.0 * action_id,
            "period_id_a1": 1,
            "time_seconds_a1": max(0.0, 30.0 * (action_id - 1)),
            "time_seconds_overall_a1": max(0.0, 30.0 * (action_id - 1)),
            "period_id_a2": 1,
            "time_seconds_a2": max(0.0, 30.0 * (action_id - 2)),
            "time_seconds_overall_a2": max(0.0, 30.0 * (action_id - 2)),
        }
        for slot_idx, slot in enumerate(("a0", "a1", "a2")):
            action_name = action_slots[action_id][slot_idx]
            result_name = result_slots[action_id][slot_idx]
            bodypart_name = bodypart_slots[action_id][slot_idx]
            row[f"actiontype_pass_{slot}"] = int(action_name == "pass")
            row[f"actiontype_shot_{slot}"] = int(action_name == "shot")
            row[f"result_success_{slot}"] = int(result_name == "success")
            row[f"result_fail_{slot}"] = int(result_name == "fail")
            row[f"bodypart_foot_{slot}"] = int(bodypart_name == "foot")
            row[f"bodypart_head_{slot}"] = int(bodypart_name == "head")
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_scores_xgb_feature_frame_decodes_categories_and_possession(scores_feature_refactor_df):
    X, feature_cols, categorical_cols = build_scores_xgb_feature_frame(scores_feature_refactor_df)

    assert feature_cols == list(X.columns)
    assert categorical_cols == [
        "actiontype_a0", "actiontype_a1", "actiontype_a2",
        "result_a0", "result_a1", "result_a2",
        "bodypart_a0", "bodypart_a1", "bodypart_a2",
    ]
    assert str(X["actiontype_a0"].dtype) == "category"
    assert str(X["result_a1"].dtype) == "category"
    assert str(X["bodypart_a2"].dtype) == "category"
    assert X.loc[0, "actiontype_a0"] == "pass"
    assert pd.isna(X.loc[0, "actiontype_a1"])
    assert X.loc[2, "actiontype_a2"] == "pass"
    assert X.loc[2, "result_a0"] == "fail"
    assert X.loc[2, "bodypart_a0"] == "head"

    assert np.isclose(
        X.loc[0, "start_dist_to_goal"] - X.loc[0, "end_dist_to_goal"],
        X.loc[0, "dist_to_goal_delta"],
    )
    assert X["start_angle_to_goal"].gt(0).all()
    assert X["end_angle_to_goal"].gt(0).all()
    assert X["in_final_third"].isin([0, 1]).all()
    assert X["start_in_box"].isin([0, 1]).all()
    assert X["end_in_box"].isin([0, 1]).all()

    assert X["possession_chain_len"].tolist() == [1, 2, 3, 1, 2]
    assert X["same_team_a01"].tolist() == [0, 1, 1, 0, 1]
    assert X["same_team_a12"].tolist() == [0, 0, 1, 1, 0]
    assert X["same_team_a02"].tolist() == [0, 0, 1, 0, 0]
    assert X["turnover_a01"].tolist() == [0, 0, 0, 1, 0]
    assert X["turnover_a12"].tolist() == [0, 0, 0, 0, 1]


def test_prepare_vaep_xgb_features_preserves_scores_categoricals(scores_feature_refactor_df):
    X, feature_cols, categorical_cols = prepare_vaep_xgb_features(
        scores_feature_refactor_df,
        target_col="scores",
    )

    assert feature_cols == list(X.columns)
    assert categorical_cols
    assert "actiontype_pass_a0" not in X.columns
    assert "result_success_a1" not in X.columns
    assert "bodypart_head_a2" not in X.columns
    assert "actiontype_a0" in X.columns
    assert "same_team_a01" in X.columns
    assert "possession_chain_len" in X.columns
    assert str(X["actiontype_a0"].dtype) == "category"


@pytest.fixture
def split_refactor_df():
    rows: list[dict[str, object]] = []
    competitions = {
        ("La Liga", "2015/2016", 101): 6,
        ("La Liga", "2016/2017", 102): 6,
        ("La Liga", "2019/2020", 103): 6,
        ("La Liga", "2020/2021", 104): 6,
        ("Premier League", "2015/2016", 201): 4,
        ("Premier League", "2003/2004", 202): 4,
        ("Serie A", "2015/2016", 301): 4,
        ("Serie A", "1986/1987", 302): 4,
        ("1. Bundesliga", "2015/2016", 401): 4,
        ("1. Bundesliga", "2023/2024", 402): 4,
        ("Ligue 1", "2015/2016", 501): 4,
        ("Ligue 1", "2021/2022", 502): 4,
        ("Ligue 1", "2022/2023", 503): 4,
        ("Champions League", "2015/2016", 601): 4,
        ("Champions League", "2004/2005", 602): 4,
        ("Champions League", "1999/2000", 603): 4,
        ("UEFA Europa League", "1988/1989", 701): 4,
        ("FA Women's Super League", "2015/2016", 801): 4,
    }
    game_id = 1000
    for (competition_name, season_name, season_id), n_games in competitions.items():
        for _ in range(n_games):
            for action_id in range(3):
                rows.append({
                    "game_id": game_id,
                    "action_id": action_id,
                    "team_id": 10 + (game_id % 2),
                    "competition_name": competition_name,
                    "season_name": season_name,
                    "season_id": season_id,
                    "scores": int((game_id + action_id) % 2 == 0),
                    "concedes": int((game_id + action_id + 1) % 3 == 0),
                    "start_x_a0": float(game_id % 100) + action_id,
                    "start_y_a0": float(season_id % 100) + action_id,
                    "end_x_a0": float(game_id % 50) + action_id,
                    "end_y_a0": float(season_id % 50) + action_id,
                    "start_x_a1": float(game_id % 100) + max(0, action_id - 1),
                    "start_y_a1": float(season_id % 100) + max(0, action_id - 1),
                    "start_x_a2": float(game_id % 100) + max(0, action_id - 2),
                    "start_y_a2": float(season_id % 100) + max(0, action_id - 2),
                    "end_x_a1": float(game_id % 50) + max(0, action_id - 1),
                    "end_y_a1": float(season_id % 50) + max(0, action_id - 1),
                    "end_x_a2": float(game_id % 50) + max(0, action_id - 2),
                    "end_y_a2": float(season_id % 50) + max(0, action_id - 2),
                    "dx_a0": 1.0,
                    "dy_a0": 1.0,
                    "movement_a0": 1.414,
                    "dx_a1": 1.0,
                    "dy_a1": 0.0,
                    "movement_a1": 1.0,
                    "dx_a2": 0.0,
                    "dy_a2": 1.0,
                    "movement_a2": 1.0,
                    "dx_a01": 1.0,
                    "dy_a01": 1.0,
                    "mov_a01": 1.414,
                    "dx_a02": 2.0,
                    "dy_a02": 2.0,
                    "mov_a02": 2.828,
                    "goalscore_diff": float((action_id % 3) - 1),
                    "goalscore_team": 0.0,
                    "goalscore_opponent": 0.0,
                    "period_id_a0": 1,
                    "time_seconds_a0": float(action_id * 30),
                    "time_seconds_overall_a0": float(action_id * 30),
                    "period_id_a1": 1,
                    "time_seconds_a1": float(max(0, action_id - 1) * 30),
                    "time_seconds_overall_a1": float(max(0, action_id - 1) * 30),
                    "period_id_a2": 1,
                    "time_seconds_a2": float(max(0, action_id - 2) * 30),
                    "time_seconds_overall_a2": float(max(0, action_id - 2) * 30),
                    "actiontype_pass_a0": int(action_id % 2 == 0),
                    "actiontype_shot_a0": int(action_id % 2 == 1),
                    "actiontype_pass_a1": int(action_id % 2 == 0),
                    "actiontype_shot_a1": int(action_id % 2 == 1),
                    "actiontype_pass_a2": int(action_id % 2 == 0),
                    "actiontype_shot_a2": int(action_id % 2 == 1),
                    "result_success_a0": 1,
                    "result_fail_a0": 0,
                    "result_success_a1": 1,
                    "result_fail_a1": 0,
                    "result_success_a2": 1,
                    "result_fail_a2": 0,
                    "bodypart_foot_a0": 1,
                    "bodypart_head_a0": 0,
                    "bodypart_foot_a1": 1,
                    "bodypart_head_a1": 0,
                    "bodypart_foot_a2": 1,
                    "bodypart_head_a2": 0,
                })
            game_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def split_refactor_config():
    return {
        "source": {
            "competitions": ["La Liga"],
            "exclude_seasons": ["2019-20", "2020-21"],
        },
        "calib": {
            "competitions": ["Champions League", "UEFA Europa League"],
        },
        "test": {
            "competitions": [
                "La Liga",
                "Premier League",
                "Serie A",
                "1. Bundesliga",
                "Ligue 1",
                "Champions League",
                "UEFA Europa League",
            ],
            "year_shift": {
                "seasons": ["2019-20", "2020/2021"],
            },
            "league_season_shift": {
                "explicit": {},
            },
        },
    }


def test_resolve_season_aliases_accepts_flexible_aliases():
    available = ["2015/2016", "2019/2020", "2020/2021"]
    assert resolve_season_aliases(available, ["2019-20", "2020/2021"]) == ["2019/2020", "2020/2021"]


def test_resolve_season_aliases_rejects_unknown_alias():
    with pytest.raises(ValueError, match="Could not resolve requested season"):
        resolve_season_aliases(["2015/2016"], ["2099-00"])


def test_resolve_competition_season_split_spec_infers_shift_groups(split_refactor_df, split_refactor_config):
    resolved = resolve_competition_season_split_spec(split_refactor_df, split_refactor_config)

    assert resolved["source_keys"] == {("La Liga", "2015/2016"), ("La Liga", "2016/2017")}
    assert resolved["test_keys"]["year_shift"] == {("La Liga", "2019/2020"), ("La Liga", "2020/2021")}
    assert resolved["test_keys"]["league_shift"] == {
        ("Premier League", "2015/2016"),
        ("Serie A", "2015/2016"),
        ("1. Bundesliga", "2015/2016"),
        ("Ligue 1", "2015/2016"),
        ("Champions League", "2015/2016"),
    }
    assert resolved["test_keys"]["league_season_shift"] == {
        ("Premier League", "2003/2004"),
        ("Serie A", "1986/1987"),
        ("1. Bundesliga", "2023/2024"),
        ("Ligue 1", "2021/2022"),
        ("Ligue 1", "2022/2023"),
        ("Champions League", "2004/2005"),
        ("Champions League", "1999/2000"),
        ("UEFA Europa League", "1988/1989"),
    }
    assert ("Champions League", "2015/2016") in resolved["calib_keys"]
    assert ("UEFA Europa League", "1988/1989") in resolved["calib_keys"]
    assert resolved["target_keys"] == {("FA Women's Super League", "2015/2016")}


def test_load_xy_competition_season_split_stratifies_source_by_season(monkeypatch, split_refactor_df, split_refactor_config):
    monkeypatch.setattr("football_ai.training.read_h5_table", lambda data_file, key_candidates: split_refactor_df.copy())
    resolved = load_xy_competition_season_split(
        target_col="scores",
        data_file="unused.h5",
        key_candidates=["vaep_data"],
        split_config=split_refactor_config,
        validation_frac=0.5,
        random_state=7,
    )

    train_games = set(resolved.source_train.df["game_id"].unique())
    val_games = set(resolved.source_val.df["game_id"].unique())
    assert train_games.isdisjoint(val_games)

    train_counts = resolved.source_train.df[["game_id", "season_id"]].drop_duplicates()["season_id"].value_counts().to_dict()
    val_counts = resolved.source_val.df[["game_id", "season_id"]].drop_duplicates()["season_id"].value_counts().to_dict()
    assert train_counts == {101: 3, 102: 3}
    assert val_counts == {101: 3, 102: 3}

    assert ("Champions League", "2015/2016") in {
        tuple(row) for row in resolved.calib.competition_seasons[["competition_name", "season_name"]].itertuples(index=False, name=None)
    }
    assert ("Champions League", "2015/2016") in {
        tuple(row) for row in resolved.test["league_shift"].competition_seasons[["competition_name", "season_name"]].itertuples(index=False, name=None)
    }
    assert resolved.target.competitions == ["FA Women's Super League"]
