#!/usr/bin/env python
"""Exploratory analysis of playing styles across competitions and seasons.

Thin orchestration script — all reusable logic lives in the
``football_ai`` package (``data``, ``data_viz``).

Produces:
- Full (all competitions) analysis
- Per-league analysis (groupby competition_name)
- Champions League 3-season comparison
- Actions-per-game stats, success-rates, spatial heatmaps, league-shift tables

All artifacts are saved under ``results/eda/styles``.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from football_ai.data import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    build_styles_dataframe,
    pick_three_seasons,
    read_h5_table,
    sample_dataframe,
    slug,
)
from football_ai.data_viz import (
    plot_all_leagues_feature_comparisons,
    plot_champions_action_comparison,
    plot_champions_displacement_comparison,
    run_scope_analysis,
    save_actions_per_game_stats,
    save_champions_comparison_tables,
    save_league_shift_tables,
    save_league_success_rate_by_action_type,
    save_score_rate_overview,
    save_spatial_zone_summary_by_league,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ╔════════════════════════════════════════════════╗
# ║  CONFIG                                        ║
# ╚════════════════════════════════════════════════╝

DATA_FILE = Path("data/spadl_full_data/major_leagues.h5")
# DATA_FILE = Path("data/spadl_full_data/women_leagues.h5")
KEY = "full_data"
OUTPUT_DIR = Path("results/eda/styles")
SAMPLE_FRAC = 1.0
SEED = 20260306

CHAMPIONS_NAME = "Champions League"
MAX_HIST_SAMPLES = 50_000
TOP_K_CHAMPIONS_CATEGORIES = 20
TOP_K_ALL_LEAGUES_CATEGORIES = 15

SUCCESS_COMPARE_ACTION_TYPES = [
    "pass", "cross", "shot", "dribble",
    "tackle", "keeper_save", "foul", "bad_touch",
]

# Spatial zone grid
WING_ZONE_METERS = 15.0
X_ZONE_EDGES = np.linspace(0.0, FIELD_LENGTH, 5)
Y_ZONE_EDGES = np.array([0.0, WING_ZONE_METERS, FIELD_WIDTH - WING_ZONE_METERS, FIELD_WIDTH])

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)


# ╔════════════════════════════════════════════════╗
# ║  MAIN                                          ║
# ╚════════════════════════════════════════════════╝


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load data ──
    log.info("Loading %s key='%s'", DATA_FILE, KEY)
    df = read_h5_table(DATA_FILE, KEY)
    log.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])

    if 0 < SAMPLE_FRAC < 1.0:
        df = sample_dataframe(df, frac=SAMPLE_FRAC, seed=SEED)
        log.info("Subsampled to %s rows (frac=%.3f)", f"{len(df):,}", SAMPLE_FRAC)

    df_styles = build_styles_dataframe(df)
    log.info("Built style dataframe with %s rows", f"{len(df_styles):,}")

    # ── actions-per-game stats ──
    save_actions_per_game_stats(df_styles=df_styles, output_dir=OUTPUT_DIR)
    log.info("Saved actions-per-game stats (joint/by-league/by-season)")

    # ── success rate by action type ──
    save_league_success_rate_by_action_type(
        df_styles=df_styles,
        output_dir=OUTPUT_DIR,
        action_types=SUCCESS_COMPARE_ACTION_TYPES,
    )
    log.info("Saved league comparison: success rate by action type")

    # ── score-rate overviews ──
    save_score_rate_overview(df_styles=df_styles, output_dir=OUTPUT_DIR)
    log.info("Saved score-rate overviews by league and by season")

    # ── spatial zone analysis ──
    save_spatial_zone_summary_by_league(
        df_styles=df_styles,
        output_dir=OUTPUT_DIR,
        x_edges=X_ZONE_EDGES,
        y_edges=Y_ZONE_EDGES,
    )
    log.info("Saved spatial league summary plots (start/end success + start distance)")

    # ── league-shift tables & heatmaps ──
    save_league_shift_tables(df_styles=df_styles, output_dir=OUTPUT_DIR)
    plot_all_leagues_feature_comparisons(
        df_styles=df_styles,
        output_dir=OUTPUT_DIR,
        top_k_categories=TOP_K_ALL_LEAGUES_CATEGORIES,
    )
    log.info("Saved all-leagues per-feature comparison plots")

    # ── joint (all competitions) analysis ──
    run_scope_analysis(
        df_styles=df_styles,
        scope_name="all_competitions",
        scope_title="All competitions",
        output_dir=OUTPUT_DIR / "joint",
        seed=SEED,
        max_hist_samples=MAX_HIST_SAMPLES,
    )
    log.info("Saved joint analysis outputs")

    # ── per-league analysis ──
    by_league_dir = OUTPUT_DIR / "by_league"
    by_league_dir.mkdir(parents=True, exist_ok=True)
    for competition_name, df_comp in df_styles.groupby("competition_name", dropna=False):
        label = "missing_competition" if pd.isna(competition_name) else str(competition_name)
        comp_slug = slug(label)
        run_scope_analysis(
            df_styles=df_comp,
            scope_name=comp_slug,
            scope_title=f"League: {label}",
            output_dir=by_league_dir / comp_slug,
            seed=SEED,
            max_hist_samples=MAX_HIST_SAMPLES,
        )
    log.info("Saved per-league analysis outputs")

    # ── Champions League 3-season comparison ──
    champions_mask = (
        df_styles["competition_name"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.contains(CHAMPIONS_NAME.casefold())
    )
    df_champions = df_styles[champions_mask].copy()

    champions_dir = OUTPUT_DIR / "champions_3seasons"
    champions_dir.mkdir(parents=True, exist_ok=True)

    if df_champions.empty:
        log.warning("No Champions League rows found. Skipping Champions comparison.")
    else:
        selected_seasons = pick_three_seasons(
            df_champions["season_name"].dropna().astype(str).tolist()
        )
        if not selected_seasons:
            log.warning("No valid Champions League seasons found. Skipping.")
        else:
            pd.DataFrame({
                "selected_season_name": selected_seasons,
                "selection_rule": ["oldest", "middle", "most_recent"][: len(selected_seasons)],
            }).to_csv(champions_dir / "selected_seasons.csv", index=False)

            df_champ_sel = df_champions[
                df_champions["season_name"].astype(str).isin(selected_seasons)
            ].copy()

            save_champions_comparison_tables(df_champ=df_champ_sel, output_dir=champions_dir)

            plot_champions_action_comparison(
                df_champ=df_champ_sel,
                selected_seasons=selected_seasons,
                output_path=champions_dir / "champions_3seasons_action_comparison.png",
                top_k_categories=TOP_K_CHAMPIONS_CATEGORIES,
            )
            plot_champions_displacement_comparison(
                df_champ=df_champ_sel,
                selected_seasons=selected_seasons,
                output_path=champions_dir / "champions_3seasons_displacement_comparison.png",
                seed=SEED,
                max_hist_samples=MAX_HIST_SAMPLES,
            )

            seasons_dir = champions_dir / "per_season"
            seasons_dir.mkdir(parents=True, exist_ok=True)
            for season_name in selected_seasons:
                season_df = df_champions[df_champions["season_name"].astype(str) == season_name].copy()
                season_slug = slug(season_name)
                run_scope_analysis(
                    df_styles=season_df,
                    scope_name=f"champions_{season_slug}",
                    scope_title=f"Champions League — {season_name}",
                    output_dir=seasons_dir / season_slug,
                    seed=SEED,
                    max_hist_samples=MAX_HIST_SAMPLES,
                )

            log.info("Saved Champions League 3-season comparison outputs: %s", selected_seasons)

    log.info("Style EDA complete. Artifacts saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
