"""Smoke tests for football_ai.data_viz plotting functions.

Fast, CPU-only, no external data. Tests that functions run without errors
and produce output files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from football_ai.data import build_styles_dataframe
from football_ai.data_viz import (
    plot_missingness_bar,
    plot_target_distribution,
    plot_univariate_numeric_grid,
    plot_univariate_categorical_grid,
    plot_bivariate_target_corr,
    plot_correlation_heatmap,
    plot_feature_importance_comparison,
    plot_action_distributions,
    plot_displacement_histograms,
    generate_eda_summary_markdown,
)


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path


@pytest.fixture
def tiny_spadl_df():
    """Minimal DataFrame matching REQUIRED_STYLE_COLS layout."""
    n = 60
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


def test_plot_missingness_bar(tmp_output):
    miss_df = pd.DataFrame({"column": ["a", "b"], "pct_missing": [30.0, 10.0]})
    out_path = tmp_output / "miss.png"
    plot_missingness_bar(miss_df, out_path)
    assert out_path.exists()


def test_plot_target_distribution(tmp_output):
    vc = pd.Series({0: 900, 1: 100})
    out_path = tmp_output / "target.png"
    plot_target_distribution(vc, "scores", out_path)
    assert out_path.exists()


def test_plot_univariate_numeric_grid(tmp_output):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.uniform(size=100)})
    out_path = tmp_output / "num.png"
    plot_univariate_numeric_grid(df, ["a", "b"], out_path, seed=42)
    assert out_path.exists()


def test_plot_univariate_categorical_grid(tmp_output):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.choice(["A", "B", "C"], size=100)})
    out_path = tmp_output / "cat.png"
    plot_univariate_categorical_grid(df, ["x"], out_path, top_k=10)
    assert out_path.exists()


def test_plot_bivariate_target_corr(tmp_output):
    biv_df = pd.DataFrame({
        "column": ["a", "b", "c"],
        "pb_corr": [0.3, -0.2, 0.1],
    })
    out_path = tmp_output / "biv.png"
    plot_bivariate_target_corr(biv_df, "scores", out_path, top_k=5)
    assert out_path.exists()


def test_plot_correlation_heatmap(tmp_output):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd"))
    corr = df.corr()
    out_path = tmp_output / "heatmap.png"
    plot_correlation_heatmap(corr, out_path)
    assert out_path.exists()


def test_plot_feature_importance_comparison(tmp_output):
    lr_imp = pd.Series({"a": 0.5, "b": 0.3, "c": 0.1})
    rf_imp = pd.Series({"a": 0.2, "b": 0.6, "c": 0.2})
    out_path = tmp_output / "imp.png"
    plot_feature_importance_comparison(lr_imp, rf_imp, "LR", "RF", out_path)
    assert out_path.exists()


def test_plot_action_distributions(tiny_spadl_df, tmp_output):
    styles = build_styles_dataframe(tiny_spadl_df)
    out_path = tmp_output / "actions.png"
    plot_action_distributions(styles, "Test", out_path)
    assert out_path.exists()


def test_plot_displacement_histograms(tiny_spadl_df, tmp_output):
    styles = build_styles_dataframe(tiny_spadl_df)
    out_path = tmp_output / "disp.png"
    plot_displacement_histograms(styles, "Test", out_path, seed=42)
    assert out_path.exists()


def test_generate_eda_summary_markdown(tmp_output):
    md = generate_eda_summary_markdown(
        data_file=Path("test.h5"),
        key="test",
        shape=(100, 10),
        target_col="scores",
        target_info={"value_counts": {0: 90, 1: 10}, "positive_rate": 0.1, "n_missing": 0},
        col_groups={"id": [], "meta": [], "target": ["scores"], "numeric": ["a"], "categorical": ["b"],
                    "leakage_suspect": [], "datetime": [], "bool": []},
        miss_df=pd.DataFrame(columns=["column", "pct_missing"]),
        const_cols=[],
        leakage=[],
        cat_df=pd.DataFrame(columns=["column", "is_dominant", "top_value", "top_pct"]),
        corr_pairs=pd.DataFrame(),
        corr_threshold=0.85,
        baselines={},
        baseline_metrics_df=pd.DataFrame(),
        output_dir=tmp_output,
    )
    assert "# EDA Summary" in md
    assert "scores" in md
