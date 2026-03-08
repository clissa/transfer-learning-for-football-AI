#!/usr/bin/env python
"""Create VAEP features from an existing SPADL full_data HDF5 table.

Reads the ``full_data`` key from an HDF5 file produced by
``scripts/create_spadl_dataset.py``, computes VAEP game-state features
(all 9 socceraction feature functions), and saves the result to a new
HDF5 file preserving all original metadata columns.

Usage examples
--------------
# Default config
python -m scripts.create_vaep_features --config configs/create_vaep_features.yaml

# Override nb_prev_actions
python -m scripts.create_vaep_features --config configs/create_vaep_features.yaml --nb-prev-actions 5

# Custom input/output
python -m scripts.create_vaep_features \\
    --config configs/create_vaep_features.yaml \\
    --input-file data/spadl_full_data/women_leagues.h5 \\
    --outname women_leagues_vaep.h5
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from football_ai.config import load_config, merge_cli_overrides
from football_ai.data import read_h5_table
from football_ai.features import build_vaep_dataset, save_vaep_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────
# CLI
# ─────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute VAEP features from a SPADL full_data HDF5 table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file (e.g. configs/create_vaep_features.yaml)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to the input HDF5 file (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the output HDF5 file (overrides config)",
    )
    parser.add_argument(
        "--outname",
        type=str,
        default=None,
        help="Output HDF5 filename (overrides config)",
    )
    parser.add_argument(
        "--nb-prev-actions",
        type=int,
        default=None,
        help="Number of previous actions for context (overrides config)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Limit to the first N games (for quick testing). Default: all games.",
    )
    return parser.parse_args()


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────


def main() -> int:
    """Entry point: load config, build VAEP features, save result."""
    args = parse_args()
    t0 = time.perf_counter()

    # Config: YAML → CLI overrides
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, {
        "input_file": args.input_file,
        "output_dir": args.output_dir,
        "outname": args.outname,
        "nb_prev_actions": args.nb_prev_actions,
    })

    input_file = Path(cfg["input_file"])
    input_key = cfg.get("input_key", "full_data")
    output_dir = Path(cfg.get("output_dir", "data/vaep_data"))
    outname = cfg.get("outname", "major_leagues_vaep.h5")
    output_key = cfg.get("output_key", "vaep_data")
    nb_prev_actions = int(cfg.get("nb_prev_actions", 3))
    output_path = output_dir / outname

    logger.info("Configuration:")
    logger.info("  input_file      = %s", input_file)
    logger.info("  input_key       = %s", input_key)
    logger.info("  output_path     = %s", output_path)
    logger.info("  output_key      = %s", output_key)
    logger.info("  nb_prev_actions = %d", nb_prev_actions)

    # input_file = Path("data/spadl_full_data/major_leagues.h5")

    max_games: int | None = args.max_games

    # 1. Load full_data
    df = read_h5_table(input_file, input_key)
    logger.info("Loaded %s — shape %s", input_file, df.shape)

    # 1b. Optional subset for quick testing
    if max_games is not None:
        game_ids = sorted(df["game_id"].unique())[:max_games]
        df = df[df["game_id"].isin(game_ids)].reset_index(drop=True)
        logger.info("Subset: keeping %d games → %d rows", len(game_ids), len(df))

    # 2. Build VAEP features + merge
    df_vaep = build_vaep_dataset(df, nb_prev_actions=nb_prev_actions)

    # 3. Save
    save_vaep_dataset(df_vaep, output_path, key=output_key)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done in %.1fs — input %s → output %s  (%s rows, %d cols)",
        elapsed,
        df.shape,
        df_vaep.shape,
        len(df_vaep),
        len(df_vaep.columns),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
