"""Visualisation and reporting utilities for football AI EDA.

All plotting functions accept pre-computed data and write figures to disk.
Matplotlib is imported lazily so that the core package stays import-light.

Functions are organised into:

- **Generic EDA plots**: missingness, target distribution, univariate /
  bivariate summaries, correlation heatmaps, etc.
- **SPADL style plots**: action distributions, displacement histograms,
  per-scope analysis, league-shift heatmaps.
- **Spatial pitch plots**: zone heatmaps, pitch grid overlays.
- **Champions League comparison**: season-to-season bar charts and overlay
  histograms.
- **Markdown summary generator**.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    STYLE_CATEGORICAL_COLS,
    STYLE_DISPLACEMENT_COLS,
    build_percentage_distribution,
    build_zone_metrics_for_coordinates,
    parse_season_sort_key,
    sample_series,
    slug,
)

log = logging.getLogger(__name__)

# Small sentinel used when plotting on log-scale to replace exact zeros.
LOG_PCT_EPS = 0.01
SUCCESS_LABEL = "success"


# ──────────────────────────────────────────────
# Generic EDA plots (from eda.py)
# ──────────────────────────────────────────────


def plot_missingness_bar(
    miss_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 40,
) -> None:
    """Horizontal bar-chart of missing-value percentages.

    Args:
        miss_df: DataFrame with ``column`` and ``pct_missing``.
        output_path: Path for the saved PNG.
        top_n: Maximum number of columns to show.
    """
    import matplotlib.pyplot as plt

    if miss_df.empty:
        return
    top = miss_df.head(top_n)
    fig, ax = plt.subplots(figsize=(max(8, len(top) * 0.35), 5))
    ax.barh(top["column"], top["pct_missing"], color="salmon")
    ax.set_xlabel("% Missing")
    ax.set_title(f"Missing values by column (top {top_n})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_target_distribution(
    target_vc: pd.Series,
    target_col: str,
    output_path: Path,
) -> None:
    """Bar-chart of target-variable value counts (log y-axis).

    Args:
        target_vc: Value-count Series (from ``y.value_counts()``).
        target_col: Name of the target column (used in title).
        output_path: Path for the saved PNG.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    target_vc.plot.bar(ax=ax, color=["steelblue", "coral"])
    ax.set_title(f"Target '{target_col}' distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(target_vc.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_univariate_numeric_grid(
    df: pd.DataFrame,
    num_feats: list[str],
    output_path: Path,
    seed: int = 42,
    max_samples: int = 50_000,
) -> None:
    """Grid of histograms for numeric features.

    Args:
        df: Input DataFrame.
        num_feats: List of numeric column names.
        output_path: Path for the saved PNG.
        seed: Random seed for sub-sampling.
        max_samples: Each column is capped at this many samples.
    """
    import matplotlib.pyplot as plt

    if not num_feats:
        return
    n = len(num_feats)
    ncols_plot = min(4, n)
    nrows_plot = int(np.ceil(n / ncols_plot))
    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(4 * ncols_plot, 3 * nrows_plot))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(num_feats):
        ax = axes_flat[i]
        data = df[col].dropna()
        if len(data) > max_samples:
            data = data.sample(max_samples, random_state=seed)
        if pd.api.types.is_bool_dtype(data):
            data = data.astype(int)
        ax.hist(data, bins=50, edgecolor="white", alpha=0.8)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Numeric feature distributions", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_univariate_categorical_grid(
    df: pd.DataFrame,
    cat_feats: list[str],
    output_path: Path,
    top_k: int = 15,
) -> None:
    """Grid of horizontal bar-charts for low-cardinality categorical features.

    Args:
        df: Input DataFrame.
        cat_feats: List of categorical column names.
        output_path: Path for the saved PNG.
        top_k: Maximum unique values for a column to be plotted.
    """
    import matplotlib.pyplot as plt

    low_card = [c for c in cat_feats if df[c].nunique(dropna=True) <= top_k]
    if not low_card:
        return
    n = len(low_card)
    ncols_plot = min(3, n)
    nrows_plot = int(np.ceil(n / ncols_plot))
    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(5 * ncols_plot, 3.5 * nrows_plot))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(low_card):
        ax = axes_flat[i]
        vc = df[col].value_counts().head(top_k)
        vc.plot.barh(ax=ax)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Categorical distributions (low cardinality)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_bivariate_target_corr(
    biv_df: pd.DataFrame,
    target_col: str,
    output_path: Path,
    top_k: int = 20,
) -> None:
    """Horizontal bar-chart of point-biserial correlations with the target.

    Args:
        biv_df: DataFrame with ``column`` and ``pb_corr``.
        target_col: Target column name (used in title).
        output_path: Path for the saved PNG.
        top_k: Number of top features to display.
    """
    import matplotlib.pyplot as plt

    top = biv_df.head(top_k)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    colors = ["coral" if v > 0 else "steelblue" for v in top["pb_corr"]]
    ax.barh(top["column"], top["pb_corr"], color=colors)
    ax.set_xlabel("Point-biserial correlation with target")
    ax.set_title(f"Top {top_k} features by |correlation| with '{target_col}'")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Correlation heat-map for numeric features.

    Args:
        corr_matrix: Square correlation DataFrame.
        output_path: Path for the saved PNG.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_feats = len(corr_matrix)
    size = min(16, max(8, n_feats * 0.5))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        square=True,
        cbar_kws={"shrink": 0.7},
        annot=n_feats <= 20,
        fmt=".2f" if n_feats <= 20 else "",
        annot_kws={"fontsize": 7},
    )
    ax.set_title("Feature correlation heatmap")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_feature_importance_comparison(
    lr_imp: pd.Series,
    rf_imp: pd.Series,
    lr_label: str,
    rf_label: str,
    output_path: Path,
    top_k: int = 20,
) -> None:
    """Side-by-side bar-charts of LogReg |coef| and RF importance.

    Args:
        lr_imp: LogReg absolute coefficients (descending).
        rf_imp: Random-forest feature importances (descending).
        lr_label: Subtitle for LogReg panel.
        rf_label: Subtitle for RF panel.
        output_path: Path for the saved PNG.
        top_k: Number of features per panel.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lr_imp.head(top_k).plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title(lr_label)
    axes[0].invert_yaxis()
    rf_imp.head(top_k).plot.barh(ax=axes[1], color="coral")
    axes[1].set_title(rf_label)
    axes[1].invert_yaxis()
    fig.suptitle(f"Top {top_k} features — quick baselines", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
# SPADL style plots (from eda_styles.py)
# ──────────────────────────────────────────────


def plot_action_distributions(
    df_styles: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Three-panel horizontal bar-chart of action_type / result / bodypart.

    Args:
        df_styles: Style DataFrame produced by
            :func:`~football_ai.data.build_styles_dataframe`.
        title: Super-title for the figure.
        output_path: Path for the saved PNG.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, STYLE_CATEGORICAL_COLS):
        vc = build_percentage_distribution(df_styles=df_styles, col=col)
        if vc.empty or vc["pct_actions"].dropna().empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(col)
            ax.set_axis_off()
            continue
        vc = vc.sort_values("pct_actions", ascending=True)
        ax.barh(vc[col].astype(str), vc["pct_actions"], color="steelblue")
        ax.set_xlabel("% actions")
        ax.set_xscale("log")
        ax.set_title(f"{col} distribution (%)", fontsize=10)
        ax.invert_yaxis()
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_displacement_histograms(
    df_styles: pd.DataFrame,
    title: str,
    output_path: Path,
    seed: int,
    max_hist_samples: int = 50_000,
) -> None:
    """Three-panel histogram of dx / dy / distance.

    Args:
        df_styles: Style DataFrame.
        title: Super-title.
        output_path: Path for the saved PNG.
        seed: Random seed for sub-sampling.
        max_hist_samples: Cap per column.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, STYLE_DISPLACEMENT_COLS):
        data = sample_series(df_styles[col], max_rows=max_hist_samples, seed=seed)
        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(col)
            ax.set_axis_off()
            continue
        ax.hist(data, bins=50, edgecolor="white", alpha=0.8, color="coral")
        ax.set_title(f"{col} distribution", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_style_tables(
    df_styles: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> None:
    """Write per-scope metadata, category distributions and displacement stats.

    Args:
        df_styles: Style DataFrame.
        output_dir: Directory for CSVs.
        prefix: Filename prefix (e.g. ``"all_competitions"``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    game_id_col = "game_id"
    if game_id_col in df_styles.columns:
        n_games = int(df_styles[game_id_col].nunique(dropna=True))
        mean_apg = float(len(df_styles) / max(n_games, 1))
    else:
        n_games = 0
        mean_apg = float("nan")

    meta = pd.DataFrame(
        {
            "scope": [prefix],
            "rows": [len(df_styles)],
            "n_games": [n_games],
            "mean_actions_per_game": [mean_apg],
            "n_competitions": [df_styles["competition_name"].nunique(dropna=True)],
            "n_seasons": [df_styles["season_name"].nunique(dropna=True)],
        }
    )
    meta.to_csv(output_dir / f"{prefix}_metadata.csv", index=False)

    for col in STYLE_CATEGORICAL_COLS:
        vc = build_percentage_distribution(df_styles=df_styles, col=col)
        vc.to_csv(output_dir / f"{prefix}_{col}_distribution.csv", index=False)

    disp_stats = (
        df_styles[STYLE_DISPLACEMENT_COLS]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .T
    )
    disp_stats.to_csv(output_dir / f"{prefix}_displacement_stats.csv")


def run_scope_analysis(
    df_styles: pd.DataFrame,
    scope_name: str,
    scope_title: str,
    output_dir: Path,
    seed: int,
    max_hist_samples: int = 50_000,
) -> None:
    """Orchestrate table + plot generation for one analysis scope.

    Combines :func:`save_style_tables`, :func:`plot_action_distributions`,
    and :func:`plot_displacement_histograms`.

    Args:
        df_styles: Style DataFrame (possibly filtered to one league/season).
        scope_name: Short name used for file slugs.
        scope_title: Human-readable title for plot super-titles.
        output_dir: Output directory.
        seed: Random seed.
        max_hist_samples: Cap for displacement histograms.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = slug(scope_name)
    save_style_tables(df_styles=df_styles, output_dir=output_dir, prefix=prefix)
    plot_action_distributions(
        df_styles=df_styles,
        title=f"Action features — {scope_title}",
        output_path=output_dir / f"{prefix}_action_features_distributions.png",
    )
    plot_displacement_histograms(
        df_styles=df_styles,
        title=f"Displacement distributions — {scope_title}",
        output_path=output_dir / f"{prefix}_displacement_histograms.png",
        seed=seed,
        max_hist_samples=max_hist_samples,
    )


# ──────────────────────────────────────────────
# Actions-per-game stats (from eda_styles.py)
# ──────────────────────────────────────────────


def save_actions_per_game_stats(
    df_styles: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Compute and save actions-per-game summary tables and plots.

    Produces joint / by-league / by-season / by-league-season CSVs, a joint
    histogram, a box-plot by league, and a line chart by season.

    Args:
        df_styles: Style DataFrame.
        output_dir: Directory for all artifacts.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    game_id_col = "game_id"

    if game_id_col not in df_styles.columns:
        raise KeyError(f"Column '{game_id_col}' required for actions-per-game stats")

    apg = (
        df_styles.groupby(["competition_name", "season_name", game_id_col], dropna=False)
        .size()
        .reset_index(name="num_actions")
    )
    total_actions = int(len(df_styles))
    total_games = int(len(apg))

    # ── joint summary ──
    joint = pd.DataFrame(
        {
            "n_games": [total_games],
            "n_actions": [total_actions],
            "mean_actions_per_game": [apg["num_actions"].mean()],
            "median_actions_per_game": [apg["num_actions"].median()],
            "std_actions_per_game": [apg["num_actions"].std()],
            "p10_actions_per_game": [apg["num_actions"].quantile(0.10)],
            "p90_actions_per_game": [apg["num_actions"].quantile(0.90)],
            "min_actions_per_game": [apg["num_actions"].min()],
            "max_actions_per_game": [apg["num_actions"].max()],
        }
    )
    joint.to_csv(output_dir / "actions_per_game_joint_summary.csv", index=False)

    # ── joint distribution ──
    joint_dist = (
        apg["num_actions"]
        .value_counts()
        .sort_index()
        .rename_axis("num_actions")
        .reset_index(name="n_games")
    )
    joint_dist["pct_games"] = (100 * joint_dist["n_games"] / max(total_games, 1)).round(4)
    joint_dist["actions_contribution"] = joint_dist["num_actions"] * joint_dist["n_games"]
    joint_dist["pct_actions"] = (
        100 * joint_dist["actions_contribution"] / max(total_actions, 1)
    ).round(4)
    joint_dist.to_csv(output_dir / "actions_per_game_joint_distribution.csv", index=False)

    # ── by league ──
    league_summary = (
        apg.groupby("competition_name", dropna=False)
        .agg(
            n_games=(game_id_col, "nunique"),
            n_actions=("num_actions", "sum"),
            mean_actions_per_game=("num_actions", "mean"),
            median_actions_per_game=("num_actions", "median"),
            std_actions_per_game=("num_actions", "std"),
            p10_actions_per_game=("num_actions", lambda s: s.quantile(0.10)),
            p90_actions_per_game=("num_actions", lambda s: s.quantile(0.90)),
            min_actions_per_game=("num_actions", "min"),
            max_actions_per_game=("num_actions", "max"),
        )
        .reset_index()
    )
    league_summary["pct_actions_total"] = (
        100 * league_summary["n_actions"] / max(total_actions, 1)
    ).round(4)
    league_summary["pct_games_total"] = (
        100 * league_summary["n_games"] / max(total_games, 1)
    ).round(4)
    league_summary = league_summary.sort_values("pct_actions_total", ascending=False)
    league_summary.to_csv(output_dir / "actions_per_game_by_league.csv", index=False)

    # ── by season ──
    season_summary = (
        apg.groupby("season_name", dropna=False)
        .agg(
            n_games=(game_id_col, "nunique"),
            n_actions=("num_actions", "sum"),
            mean_actions_per_game=("num_actions", "mean"),
            median_actions_per_game=("num_actions", "median"),
            std_actions_per_game=("num_actions", "std"),
            p10_actions_per_game=("num_actions", lambda s: s.quantile(0.10)),
            p90_actions_per_game=("num_actions", lambda s: s.quantile(0.90)),
            min_actions_per_game=("num_actions", "min"),
            max_actions_per_game=("num_actions", "max"),
        )
        .reset_index()
    )
    season_summary["pct_actions_total"] = (
        100 * season_summary["n_actions"] / max(total_actions, 1)
    ).round(4)
    season_summary["pct_games_total"] = (
        100 * season_summary["n_games"] / max(total_games, 1)
    ).round(4)
    season_summary["season_sort_key"] = season_summary["season_name"].astype(str).map(
        lambda name: parse_season_sort_key(name)[0]
    )
    season_summary = season_summary.sort_values(["season_sort_key", "season_name"]).drop(
        columns=["season_sort_key"]
    )
    season_summary.to_csv(output_dir / "actions_per_game_by_season.csv", index=False)

    # ── by league × season ──
    ls_summary = (
        apg.groupby(["competition_name", "season_name"], dropna=False)
        .agg(
            n_games=(game_id_col, "nunique"),
            n_actions=("num_actions", "sum"),
            mean_actions_per_game=("num_actions", "mean"),
            median_actions_per_game=("num_actions", "median"),
            std_actions_per_game=("num_actions", "std"),
        )
        .reset_index()
    )
    ls_summary["pct_actions_total"] = (
        100 * ls_summary["n_actions"] / max(total_actions, 1)
    ).round(4)
    ls_summary.to_csv(output_dir / "actions_per_game_by_league_season.csv", index=False)

    # ── plots ──
    fig, ax = plt.subplots(figsize=(10, 5))
    weights = np.ones(len(apg)) * (100 / max(total_games, 1))
    ax.hist(apg["num_actions"], bins=60, weights=weights, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_title("Actions per game — joint distribution")
    ax.set_xlabel("num actions per game")
    ax.set_ylabel("% games")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "actions_per_game_joint_histogram.png", dpi=150)
    plt.close(fig)

    box_df = apg.copy()
    box_df["competition_name"] = box_df["competition_name"].fillna("missing")
    order = (
        box_df.groupby("competition_name")["num_actions"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=box_df, x="competition_name", y="num_actions", order=order, ax=ax, showfliers=False)
    ax.set_title("Actions per game by league")
    ax.set_xlabel("competition_name")
    ax.set_ylabel("num actions per game")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "actions_per_game_by_league_boxplot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        season_summary["season_name"].astype(str),
        season_summary["mean_actions_per_game"],
        marker="o",
        color="coral",
    )
    ax.set_title("Mean actions per game by season")
    ax.set_xlabel("season_name")
    ax.set_ylabel("mean actions per game")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "actions_per_game_by_season_line.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
# League-shift and comparison (from eda_styles.py)
# ──────────────────────────────────────────────


def save_league_shift_tables(
    df_styles: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Cross-league style distribution tables and heatmaps.

    Args:
        df_styles: Style DataFrame.
        output_dir: Directory for all artifacts.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LogNorm

    output_dir.mkdir(parents=True, exist_ok=True)
    total_actions = max(len(df_styles), 1)

    comp_season = (
        df_styles.groupby(["competition_name", "season_name"], dropna=False)
        .size()
        .reset_index(name="n_actions")
        .sort_values(["competition_name", "season_name"])
    )
    comp_season["pct_actions_total"] = (100 * comp_season["n_actions"] / total_actions).round(4)
    comp_season["pct_actions_within_competition"] = (
        100 * comp_season["n_actions"]
        / comp_season.groupby("competition_name")["n_actions"].transform("sum")
    ).round(4)
    comp_season.to_csv(output_dir / "competition_season_counts.csv", index=False)

    for col in STYLE_CATEGORICAL_COLS:
        shift = (
            df_styles.groupby(["competition_name", col], dropna=False)
            .size()
            .reset_index(name="count_actions")
        )
        shift["pct_within_competition"] = (
            100 * shift["count_actions"]
            / shift.groupby("competition_name")["count_actions"].transform("sum")
        ).round(4)
        shift = shift.sort_values(["competition_name", "pct_within_competition"], ascending=[True, False])
        shift.to_csv(output_dir / f"league_shift_{col}.csv", index=False)

        top_cats = df_styles[col].value_counts(dropna=False).head(20).index.tolist()
        heat = shift[shift[col].isin(top_cats)].copy()
        pivot = heat.pivot(index="competition_name", columns=col, values="pct_within_competition").fillna(0)
        if not pivot.empty:
            fig_w = min(24, max(10, 0.55 * pivot.shape[1]))
            fig_h = min(20, max(6, 0.45 * pivot.shape[0]))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            pos_vals = pivot.to_numpy()[pivot.to_numpy() > 0]
            norm = None
            plot_data = pivot
            if pos_vals.size > 0:
                vmin = float(max(np.nanmin(pos_vals), LOG_PCT_EPS))
                vmax = float(np.nanmax(pos_vals))
                norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin))
                plot_data = pivot.mask(pivot <= 0)
            sns.heatmap(plot_data, cmap="Blues", linewidths=0.2, cbar_kws={"label": "% within league"}, norm=norm, ax=ax)
            ax.set_title(f"League style shift heatmap — {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("competition_name")
            fig.tight_layout()
            fig.savefig(output_dir / f"league_shift_{col}_heatmap.png", dpi=150)
            plt.close(fig)

    disp_by_league = (
        df_styles.groupby("competition_name", dropna=False)[STYLE_DISPLACEMENT_COLS]
        .agg(["mean", "std", "median", "min", "max"])
        .round(4)
    )
    disp_by_league.columns = [f"{a}_{b}" for a, b in disp_by_league.columns]
    disp_by_league.reset_index().to_csv(output_dir / "league_shift_displacement_stats.csv", index=False)


def plot_all_leagues_feature_comparisons(
    df_styles: pd.DataFrame,
    output_dir: Path,
    top_k_categories: int = 15,
) -> None:
    """Per-feature grouped bar-plots comparing all leagues.

    Args:
        df_styles: Style DataFrame.
        output_dir: Directory for artifacts.
        top_k_categories: Max categories to show per plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    for col in STYLE_CATEGORICAL_COLS:
        comp = (
            df_styles.groupby(["competition_name", col], dropna=False)
            .size()
            .reset_index(name="count_actions")
        )
        comp["pct_actions"] = (
            100 * comp["count_actions"]
            / comp.groupby("competition_name")["count_actions"].transform("sum")
        ).round(4)
        top_cats = build_percentage_distribution(df_styles, col).head(top_k_categories)[col].tolist()
        plot_df = comp[comp[col].isin(top_cats)].copy()
        if plot_df.empty:
            continue
        plot_df.to_csv(output_dir / f"all_leagues_{col}_comparison_table.csv", index=False)
        fig_w = min(26, max(12, 0.85 * len(top_cats) + 8))
        fig, ax = plt.subplots(figsize=(fig_w, 6.5))
        sns.barplot(data=plot_df, x=col, y="pct_actions", hue="competition_name", ax=ax)
        ax.set_title(f"All leagues comparison — {col} (%)")
        ax.set_xlabel(col)
        ax.set_ylabel("% actions within league")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.legend(title="competition_name", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"all_leagues_{col}_comparison.png", dpi=150)
        plt.close(fig)


def save_league_success_rate_by_action_type(
    df_styles: pd.DataFrame,
    output_dir: Path,
    action_types: list[str],
) -> None:
    """Success-rate comparison across leagues for selected action types.

    Args:
        df_styles: Style DataFrame.
        output_dir: Directory for artifacts.
        action_types: Action types to compare.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    compare = df_styles[df_styles["action_type"].isin(action_types)].copy()
    compare["competition_name"] = compare["competition_name"].fillna("missing_competition").astype(str)
    if compare.empty:
        log.warning("No rows for success-rate comparison action types: %s", action_types)
        return

    compare["is_success"] = compare["result"].astype(str).str.casefold().eq(SUCCESS_LABEL)
    league_order = sorted(compare["competition_name"].unique().tolist())

    grouped = (
        compare.groupby(["competition_name", "action_type"], dropna=False)
        .agg(n_actions=("is_success", "size"), n_success=("is_success", "sum"))
        .reset_index()
    )
    full_idx = pd.MultiIndex.from_product(
        [league_order, action_types], names=["competition_name", "action_type"]
    )
    grouped = grouped.set_index(["competition_name", "action_type"]).reindex(full_idx, fill_value=0).reset_index()
    grouped["success_pct"] = np.where(
        grouped["n_actions"] > 0,
        100 * grouped["n_success"] / grouped["n_actions"],
        np.nan,
    ).round(4)
    grouped.to_csv(output_dir / "league_success_rate_by_action_type.csv", index=False)

    plot_g = grouped[grouped["n_actions"] > 0].copy()
    plot_g["success_pct_plot"] = plot_g["success_pct"].clip(lower=LOG_PCT_EPS)

    fig, ax = plt.subplots(figsize=(16, 6.5))
    sns.barplot(
        data=plot_g, x="action_type", y="success_pct_plot", hue="competition_name",
        order=action_types, hue_order=league_order, errorbar=None, ax=ax,
    )
    ax.set_title("League comparison: % success by action type")
    ax.set_xlabel("action_type")
    ax.set_ylabel("% success")
    ax.set_yscale("log")
    ax.set_ylim(LOG_PCT_EPS, 100)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="competition_name", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "league_success_rate_by_action_type.png", dpi=150)
    plt.close(fig)


def save_score_rate_overview(
    df_styles: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Score-rate overview by league and by season.

    Args:
        df_styles: Style DataFrame.
        output_dir: Directory for artifacts.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    score_df = df_styles.copy()
    score_df["competition_name"] = score_df["competition_name"].fillna("missing_competition").astype(str)
    score_df["season_name"] = score_df["season_name"].fillna("missing_season").astype(str)
    score_df["is_score"] = pd.to_numeric(score_df["scores"], errors="coerce").fillna(0).astype(float) > 0

    # ── by league ──
    league_scores = (
        score_df.groupby("competition_name", dropna=False)
        .agg(n_actions=("is_score", "size"), n_scores=("is_score", "sum"))
        .reset_index()
    )
    league_scores["score_pct"] = np.where(
        league_scores["n_actions"] > 0,
        100 * league_scores["n_scores"] / league_scores["n_actions"],
        np.nan,
    ).round(4)
    league_scores = league_scores.sort_values("score_pct", ascending=False)
    league_scores.to_csv(output_dir / "league_score_rate.csv", index=False)

    plot_l = league_scores[league_scores["n_actions"] > 0].copy()
    plot_l["score_pct_plot"] = plot_l["score_pct"].clip(lower=LOG_PCT_EPS)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_l, x="competition_name", y="score_pct_plot", ax=ax, color="steelblue")
    ax.set_title("% scores by league")
    ax.set_xlabel("competition_name")
    ax.set_ylabel("% scores")
    ax.set_yscale("log")
    ax.set_ylim(LOG_PCT_EPS, max(100, float(plot_l["score_pct_plot"].max()) * 1.2))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "league_score_rate.png", dpi=150)
    plt.close(fig)

    # ── by season ──
    season_scores = (
        score_df.groupby("season_name", dropna=False)
        .agg(n_actions=("is_score", "size"), n_scores=("is_score", "sum"))
        .reset_index()
    )
    season_scores["score_pct"] = np.where(
        season_scores["n_actions"] > 0,
        100 * season_scores["n_scores"] / season_scores["n_actions"],
        np.nan,
    ).round(4)
    season_scores["season_sort_key"] = season_scores["season_name"].map(
        lambda s: parse_season_sort_key(str(s))[0]
    )
    season_scores = season_scores.sort_values(["season_sort_key", "season_name"]).drop(columns=["season_sort_key"])
    season_scores.to_csv(output_dir / "season_score_rate.csv", index=False)

    plot_s = season_scores[season_scores["n_actions"] > 0].copy()
    plot_s["score_pct_plot"] = plot_s["score_pct"].clip(lower=LOG_PCT_EPS)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_s, x="season_name", y="score_pct_plot", ax=ax, color="coral")
    ax.set_title("% scores by season")
    ax.set_xlabel("season_name")
    ax.set_ylabel("% scores")
    ax.set_yscale("log")
    ax.set_ylim(LOG_PCT_EPS, max(100, float(plot_s["score_pct_plot"].max()) * 1.2))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "season_score_rate.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
# Spatial / pitch zone plots (from eda_styles.py)
# ──────────────────────────────────────────────


def zone_rows_to_matrix(zone_rows: pd.DataFrame, value_col: str) -> np.ndarray:
    """Pivot zone-metric rows into a 2-D NumPy array.

    Args:
        zone_rows: Output of
            :func:`~football_ai.data.build_zone_metrics_for_coordinates`.
        value_col: Column to place in cells.

    Returns:
        2-D array indexed ``[y_zone_idx, x_zone_idx]``.
    """
    return (
        zone_rows.pivot(index="y_zone_idx", columns="x_zone_idx", values=value_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
        .to_numpy()
    )


def draw_pitch_with_zone_grid(
    ax: Any,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    """Draw a football pitch outline with zone grid lines on *ax*.

    Args:
        ax: Matplotlib axes.
        x_edges: Zone bin edges along x.
        y_edges: Zone bin edges along y.
    """
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (x_edges[0], y_edges[0]),
            x_edges[-1] - x_edges[0],
            y_edges[-1] - y_edges[0],
            fill=False,
            linewidth=2,
            edgecolor="black",
        )
    )
    for x in x_edges[1:-1]:
        ax.plot([x, x], [y_edges[0], y_edges[-1]], color="black", linewidth=1.0)
    for y in y_edges[1:-1]:
        ax.plot([x_edges[0], x_edges[-1]], [y, y], color="black", linewidth=1.0)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xticks(x_edges)
    ax.set_yticks(y_edges)


def plot_zone_heatmap(
    ax: Any,
    matrix: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    title: str,
    value_label: str,
    value_suffix: str,
    use_log_scale: bool,
    cmap: str = "YlOrRd",
) -> Any:
    """Draw a zone-metric heatmap with annotations on a matplotlib axes.

    Args:
        ax: Matplotlib axes.
        matrix: 2-D array from :func:`zone_rows_to_matrix`.
        x_edges: Zone bin edges along x.
        y_edges: Zone bin edges along y.
        title: Axes title.
        value_label: Colour-bar label.
        value_suffix: Suffix for annotation text (e.g. ``"%"``).
        use_log_scale: Whether to use log-normalised colours.
        cmap: Colour-map name.

    Returns:
        The ``pcolormesh`` object.
    """
    from matplotlib.colors import LogNorm

    raw = np.array(matrix, dtype=float)
    pos_vals = raw[np.isfinite(raw) & (raw > 0)]

    norm = None
    plot_m = raw.copy()
    if use_log_scale and pos_vals.size > 0:
        vmin = float(max(np.nanmin(pos_vals), LOG_PCT_EPS))
        vmax = float(np.nanmax(pos_vals))
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin))
        plot_m = np.where(plot_m > 0, plot_m, np.nan)

    if norm is not None:
        mesh = ax.pcolormesh(x_edges, y_edges, plot_m, cmap=cmap, norm=norm, shading="flat", alpha=0.85)
        ref_min, ref_max = norm.vmin, norm.vmax
    else:
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
        mesh = ax.pcolormesh(x_edges, y_edges, raw, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", alpha=0.85)
        ref_min, ref_max = vmin, vmax

    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_label + (" (log)" if use_log_scale else ""))
    draw_pitch_with_zone_grid(ax=ax, x_edges=x_edges, y_edges=y_edges)
    ax.set_title(title)

    span = max(ref_max - ref_min, 1e-9)
    for y_idx in range(raw.shape[0]):
        for x_idx in range(raw.shape[1]):
            val = float(raw[y_idx, x_idx])
            if not np.isfinite(val):
                continue
            xc = (x_edges[x_idx] + x_edges[x_idx + 1]) / 2
            yc = (y_edges[y_idx] + y_edges[y_idx + 1]) / 2
            if use_log_scale and val > 0 and pos_vals.size > 0:
                log_span = max(np.log(ref_max) - np.log(ref_min), 1e-9)
                rel = (np.log(val) - np.log(ref_min)) / log_span
            else:
                rel = (val - ref_min) / span
            txt_color = "white" if rel > 0.55 else "black"
            ax.text(xc, yc, f"{val:.1f}{value_suffix}", ha="center", va="center", fontsize=9, color=txt_color)
    return mesh


def save_spatial_zone_summary_by_league(
    df_styles: pd.DataFrame,
    output_dir: Path,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    """Per-league spatial zone analysis with heatmap panels.

    Args:
        df_styles: Style DataFrame.
        output_dir: Root output directory.
        x_edges: Zone bin edges along x.
        y_edges: Zone bin edges along y.
    """
    import matplotlib.pyplot as plt

    spatial_dir = output_dir / "spatial_by_league"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    # Save zone grid definition
    zone_defs = pd.DataFrame(
        [
            {
                "x_zone_idx": x_idx,
                "y_zone_idx": y_idx,
                "x_min": float(x_edges[x_idx]),
                "x_max": float(x_edges[x_idx + 1]),
                "y_min": float(y_edges[y_idx]),
                "y_max": float(y_edges[y_idx + 1]),
            }
            for y_idx in range(len(y_edges) - 1)
            for x_idx in range(len(x_edges) - 1)
        ]
    )
    zone_defs.to_csv(spatial_dir / "zone_grid_definition.csv", index=False)

    grouped = list(df_styles.groupby("competition_name", dropna=False))
    grouped.sort(key=lambda item: str(item[0]))

    all_zone_rows: list[pd.DataFrame] = []

    for competition_name, group_df in grouped:
        label = "missing_competition" if pd.isna(competition_name) else str(competition_name)
        league_slug = slug(label)
        prefix = league_slug

        start_rows = build_zone_metrics_for_coordinates(group_df, "start_x", "start_y", x_edges, y_edges)
        end_rows = build_zone_metrics_for_coordinates(group_df, "end_x", "end_y", x_edges, y_edges)

        start_success = zone_rows_to_matrix(start_rows, "success_rate_pct")
        end_success = zone_rows_to_matrix(end_rows, "success_rate_pct")
        start_distance = zone_rows_to_matrix(start_rows, "mean_distance")
        start_score = zone_rows_to_matrix(start_rows, "score_rate_pct")

        start_rows["coord_type"] = "start"
        end_rows["coord_type"] = "end"
        zone_rows = pd.concat([start_rows, end_rows], ignore_index=True)
        zone_rows.insert(0, "competition_name", label)
        zone_rows.to_csv(spatial_dir / f"{prefix}_zone_percentages.csv", index=False)
        all_zone_rows.append(zone_rows)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes_flat = axes.flatten()
        plot_zone_heatmap(axes_flat[0], start_success, x_edges, y_edges, f"Start — % success ({label})", "% success", "%", True, "YlOrRd")
        plot_zone_heatmap(axes_flat[1], end_success, x_edges, y_edges, f"End — % success ({label})", "% success", "%", True, "YlOrRd")
        plot_zone_heatmap(axes_flat[2], start_distance, x_edges, y_edges, f"Start — mean distance ({label})", "distance", "m", False, "viridis")
        plot_zone_heatmap(axes_flat[3], start_score, x_edges, y_edges, f"Start — % scores ({label})", "% scores", "%", True, "YlOrRd")
        fig.suptitle(f"Spatial league summary — {label}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(spatial_dir / f"{prefix}_spatial_summary.png", dpi=150)
        plt.close(fig)

    if all_zone_rows:
        pd.concat(all_zone_rows, ignore_index=True).to_csv(spatial_dir / "zone_percentages_all_leagues.csv", index=False)


# ──────────────────────────────────────────────
# Champions League comparison (from eda_styles.py)
# ──────────────────────────────────────────────


def plot_champions_action_comparison(
    df_champ: pd.DataFrame,
    selected_seasons: list[str],
    output_path: Path,
    top_k_categories: int = 20,
) -> None:
    """Three-panel grouped bar-chart comparing Champions League seasons.

    Args:
        df_champ: Style DataFrame filtered to Champions League + selected seasons.
        selected_seasons: Season names to compare.
        output_path: Path for the saved PNG.
        top_k_categories: Max categories per panel.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, col in zip(axes, STYLE_CATEGORICAL_COLS):
        counts = (
            df_champ.groupby(["season_name", col], dropna=False)
            .size()
            .reset_index(name="count_actions")
        )
        counts["pct_actions"] = (
            100 * counts["count_actions"] / counts.groupby("season_name")["count_actions"].transform("sum")
        ).round(4)
        top_cats = df_champ[col].value_counts(dropna=False).head(top_k_categories).index.tolist()
        counts = counts[counts[col].isin(top_cats)]
        if counts.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(col)
            ax.set_axis_off()
            continue
        sns.barplot(data=counts, x="pct_actions", y=col, hue="season_name", hue_order=selected_seasons, ax=ax)
        ax.set_title(f"{col} by season (%)")
        ax.set_xlabel("% actions within season")
        ax.set_xscale("log")
        ax.set_ylabel(col)
        ax.legend(title="season", fontsize=7, title_fontsize=8)
    fig.suptitle("Champions League style comparison (3 seasons)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_champions_displacement_comparison(
    df_champ: pd.DataFrame,
    selected_seasons: list[str],
    output_path: Path,
    seed: int,
    max_hist_samples: int = 50_000,
) -> None:
    """Overlay histograms comparing displacement distributions across CL seasons.

    Args:
        df_champ: Style DataFrame filtered to Champions League + selected seasons.
        selected_seasons: Season names to compare.
        output_path: Path for the saved PNG.
        seed: Random seed for sub-sampling.
        max_hist_samples: Cap per season.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    blocks: list[pd.DataFrame] = []
    for season in selected_seasons:
        block = df_champ[df_champ["season_name"] == season].copy()
        if len(block) > max_hist_samples:
            block = block.sample(n=max_hist_samples, random_state=seed)
        blocks.append(block[["season_name", *STYLE_DISPLACEMENT_COLS]])
    sampled = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, STYLE_DISPLACEMENT_COLS):
        if sampled.empty or sampled[col].dropna().empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(col)
            ax.set_axis_off()
            continue
        sns.histplot(
            data=sampled, x=col, hue="season_name", hue_order=selected_seasons,
            bins=60, stat="density", common_norm=False, element="step", fill=False, ax=ax,
        )
        ax.set_title(f"{col} by season")
        ax.set_xlabel(col)
        ax.set_ylabel("density")
    fig.suptitle("Champions League displacement comparison (3 seasons)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_champions_comparison_tables(
    df_champ: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save cross-season distribution CSVs for Champions League.

    Args:
        df_champ: Style DataFrame filtered to CL + selected seasons.
        output_dir: Directory for CSVs.
    """
    for col in STYLE_CATEGORICAL_COLS:
        dist = (
            df_champ.groupby(["season_name", col], dropna=False)
            .size()
            .reset_index(name="count_actions")
        )
        dist["pct_within_season"] = (
            100 * dist["count_actions"] / dist.groupby("season_name")["count_actions"].transform("sum")
        ).round(4)
        dist = dist.sort_values(["season_name", "pct_within_season"], ascending=[True, False])
        dist.to_csv(output_dir / f"champions_3seasons_{col}_distribution.csv", index=False)

    disp = (
        df_champ.groupby("season_name", dropna=False)[STYLE_DISPLACEMENT_COLS]
        .agg(["mean", "std", "median", "min", "max"])
        .round(4)
    )
    disp.columns = [f"{a}_{b}" for a, b in disp.columns]
    disp.reset_index().to_csv(output_dir / "champions_3seasons_displacement_stats.csv", index=False)


# ──────────────────────────────────────────────
# Markdown summary generator (from eda.py section 12)
# ──────────────────────────────────────────────


def generate_eda_summary_markdown(
    *,
    data_file: str | Path,
    key: str,
    shape: tuple[int, int],
    target_col: str,
    target_info: dict[str, Any],
    col_groups: dict[str, list[str]],
    miss_df: pd.DataFrame,
    const_cols: list[str],
    leakage: list[dict[str, Any]],
    cat_df: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    corr_threshold: float,
    baselines: dict[str, Any],
    baseline_metrics_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Build the markdown EDA summary string.

    Args:
        data_file: Dataset path.
        key: HDF5 key.
        shape: (rows, cols) of the working DataFrame.
        target_col: Target column name.
        target_info: Dict with ``value_counts``, ``positive_rate``, ``n_missing``.
        col_groups: Column-group classification dict.
        miss_df: Missingness DataFrame.
        const_cols: List of constant column names.
        leakage: List of leakage-suspect dicts.
        cat_df: Categorical summary DataFrame (with ``is_dominant`` column).
        corr_pairs: Highly-correlated-pair DataFrame.
        corr_threshold: Correlation threshold used.
        baselines: Dict of baseline metric values.
        baseline_metrics_df: DataFrame of per-model per-split metrics.
        output_dir: Output directory (for listing saved artifacts).

    Returns:
        Markdown string.
    """
    lines: list[str] = []
    lines.append("# EDA Summary\n")
    lines.append(f"**Dataset**: `{data_file}` (key=`{key}`)\n")
    lines.append(f"**Shape**: {shape[0]:,} rows x {shape[1]} columns\n")
    lines.append(f"**Target column**: `{target_col}`\n")

    # Target
    lines.append("## Target distribution\n")
    for k, v in target_info.get("value_counts", {}).items():
        lines.append(f"- `{k}`: {v:,}")
    pos_rate = target_info.get("positive_rate")
    if pos_rate is not None:
        lines.append(f"- **Positive rate**: {pos_rate:.4f}")
    lines.append("")

    # Column groups
    lines.append("## Column typing\n")
    for grp, cols in col_groups.items():
        lines.append(
            f"- **{grp}** ({len(cols)}): {', '.join(cols[:10])}"
            + (" ..." if len(cols) > 10 else "")
        )
    lines.append("")

    # Missingness
    lines.append("## Highly missing columns (>5%)\n")
    if miss_df.empty:
        lines.append("_No missing values._\n")
    else:
        high = miss_df[miss_df["pct_missing"] > 5]
        if high.empty:
            lines.append("_No columns above 5% missing._\n")
        else:
            lines.append("| Column | % Missing |")
            lines.append("|--------|-----------|")
            for _, row in high.iterrows():
                lines.append(f"| {row['column']} | {row['pct_missing']:.1f}% |")
            lines.append("")

    # Constant columns
    if const_cols:
        lines.append("## Constant columns (consider dropping)\n")
        for c in const_cols:
            lines.append(f"- `{c}`")
        lines.append("")

    # Leakage
    lines.append("## Leakage suspicion\n")
    if not leakage:
        lines.append("_No suspicious columns detected._\n")
    else:
        lines.append("| Column | Reason | Corr with target |")
        lines.append("|--------|--------|-----------------|")
        for item in leakage:
            lines.append(f"| {item['column']} | {item['reason']} | {item['corr_with_target']:.4f} |")
        lines.append("")

    # Dominant categories
    if not cat_df.empty:
        dom = cat_df[cat_df["is_dominant"]]
        if not dom.empty:
            lines.append("## Dominant categories (>80% single value)\n")
            lines.append("| Column | Top value | % |")
            lines.append("|--------|-----------|---|")
            for _, row in dom.iterrows():
                lines.append(f"| {row['column']} | {row['top_value']} | {row['top_pct']:.1f}% |")
            lines.append("")

    # Collinearity
    lines.append(f"## Highly correlated feature pairs (|r| >= {corr_threshold})\n")
    if corr_pairs.empty:
        lines.append("_None._\n")
    else:
        lines.append("| Feature A | Feature B | Pearson r |")
        lines.append("|-----------|-----------|-----------|")
        for _, row in corr_pairs.head(30).iterrows():
            lines.append(f"| {row['col_a']} | {row['col_b']} | {row['pearson_r']:.4f} |")
        if len(corr_pairs) > 30:
            lines.append(f"\n_... and {len(corr_pairs) - 30} more pairs (see CSV)._")
        lines.append("")

    # Baselines
    lines.append("## Quick baseline results\n")
    if baselines and not baseline_metrics_df.empty:
        for split_name in ["validation", "test"]:
            split_df = baseline_metrics_df[baseline_metrics_df["split"] == split_name]
            if split_df.empty:
                continue
            lines.append(f"### {split_name.capitalize()} metrics\n")
            lines.append("| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |")
            lines.append("|-------|---------|--------|-----------|--------|----|")
            for _, row in split_df.iterrows():
                lines.append(
                    f"| {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} "
                    f"| {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
                )
            lines.append("")

        lines.append("### Top 10 features (RF importance)\n")
        for i, (feat, imp) in enumerate(list(baselines.get("rf_top_features", {}).items())[:10]):
            lines.append(f"{i + 1}. `{feat}` — {imp:.5f}")
        lines.append("")
        lines.append("### Top 10 features (LogReg |coef|)\n")
        for i, (feat, imp) in enumerate(list(baselines.get("logreg_top_features", {}).items())[:10]):
            lines.append(f"{i + 1}. `{feat}` — {imp:.5f}")
    else:
        lines.append("_Baselines not run._\n")

    # Artifact list
    lines.append("\n## Saved artifacts\n")
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            lines.append(f"- `{f.name}`")

    return "\n".join(lines)
