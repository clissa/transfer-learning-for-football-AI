#!/usr/bin/env python
"""Create SPADL datasets from StatsBomb open-data.

Produces a single HDF5 file containing actions, games, teams, players,
competitions, labels, and a merged actions table with full metadata.

Usage examples
--------------
# Default: major leagues only -> data/spadl_data/major_leagues.h5
python -m scripts.create_spadl_dataset

# All available leagues
python -m scripts.create_spadl_dataset --all-leagues --outname all_leagues.h5

# Custom league selection
python -m scripts.create_spadl_dataset --leagues "La Liga" "Serie A" --outname custom.h5

# Override paths
python -m scripts.create_spadl_dataset --data-root /path/to/open-data/data --output-dir data/my_output
"""
from __future__ import annotations

import argparse
from pathlib import Path

from socceraction.data.statsbomb import StatsBombLoader

from football_ai.config import load_config, merge_cli_overrides
from football_ai.data import build_and_save_dataset


# =========================
# Default configuration
# =========================

DATA_ROOT = "../open-data/data"
OUTPUT_DIR = "data/spadl_data"

MAJOR_LEAGUES: list[str] = [
    "La Liga",
    "Serie A",
    "Premier League",
    "1. Bundesliga",
    "Ligue 1",
    "Champions League",
    "UEFA Europa League",
]

# Uncomment and pass via --leagues to include women's leagues:
# WOMEN_LEAGUES: list[str] = [
#     "FA Women's Super League",
#     "Women's World Cup",
#     "UEFA Women's Euro",
# ]

NB_PREV_ACTIONS: int = 3
OUTNAME: str = "major_leagues.h5"


# =========================
# CLI
# =========================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create SPADL dataset from StatsBomb open-data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (e.g. configs/create_datasets.yaml)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=f"Path to StatsBomb open-data JSON folder (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory for the output HDF5 file (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--outname",
        type=str,
        default=None,
        help=f"Output HDF5 filename (default: {OUTNAME})",
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Competition names to include (default: major European leagues). "
            "Pass explicit names to override."
        ),
    )
    parser.add_argument(
        "--all-leagues",
        action="store_true",
        default=False,
        help="Include ALL available competitions (overrides --leagues)",
    )
    parser.add_argument(
        "--nb-prev-actions",
        type=int,
        default=None,
        help=f"Number of previous actions for context (default: {NB_PREV_ACTIONS})",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: parse args, build and save the dataset."""
    args = parse_args()

    # Build effective config: module defaults -> YAML -> CLI overrides
    cfg: dict = {
        "data_root": DATA_ROOT,
        "output_dir": OUTPUT_DIR,
        "outname": OUTNAME,
        "nb_prev_actions": NB_PREV_ACTIONS,
        "leagues": MAJOR_LEAGUES,
    }
    if args.config is not None:
        cfg.update(load_config(args.config))
    cfg = merge_cli_overrides(cfg, {
        "data_root": args.data_root,
        "output_dir": args.output_dir,
        "outname": args.outname,
        "leagues": args.leagues,
        "nb_prev_actions": args.nb_prev_actions,
    })

    data_root = Path(cfg["data_root"])
    output_dir = Path(cfg["output_dir"])
    output_file = output_dir / cfg["outname"]
    nb_prev_actions: int = int(cfg["nb_prev_actions"])

    # Resolve league selection
    if args.all_leagues or cfg.get("all_leagues", False):
        league_names = None  # None -> select all
    else:
        league_names = cfg.get("leagues", MAJOR_LEAGUES)

    print("Configuration:")
    print(f"  data_root       = {data_root}")
    print(f"  output_file     = {output_file}")
    print(f"  leagues          = {'ALL' if league_names is None else league_names}")
    print(f"  nb_prev_actions = {nb_prev_actions}")

    loader = StatsBombLoader(getter="local", root=str(data_root))
    competitions_df = loader.competitions().copy()

    build_and_save_dataset(
        loader=loader,
        competitions_df=competitions_df,
        league_names=league_names,
        output_file=output_file,
        nb_prev_actions=nb_prev_actions,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
