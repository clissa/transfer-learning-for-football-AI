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
from football_ai.training import build_sklearn_model, build_preprocessor, preprocess_split


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
