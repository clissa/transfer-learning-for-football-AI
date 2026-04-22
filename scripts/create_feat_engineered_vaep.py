#!/usr/bin/env python
"""Persist the engineered scores feature table from an existing VAEP dataset.

The input is a raw VAEP table under ``data/vaep_data``. The output is the
train-ready engineered table for the scores XGBoost workflow under
``data/feat_engineered_vaep_data``.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from football_ai.config import load_config, merge_cli_overrides
from football_ai.data import read_h5_table
from football_ai.features import save_vaep_dataset
from football_ai.training import build_scores_engineered_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Persist engineered scores features from a VAEP HDF5 table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file (e.g. configs/create_feat_engineered_vaep.yaml)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to the input VAEP HDF5 file (overrides config)",
    )
    parser.add_argument(
        "--input-key",
        type=str,
        default=None,
        help="Input HDF5 key (overrides config)",
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
        "--output-key",
        type=str,
        default=None,
        help="Output HDF5 key (overrides config)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Limit to the first N games for quick testing. Default: all games.",
    )
    return parser.parse_args()


def main() -> int:
    """Load config, build the engineered scores dataset, and save it."""
    args = parse_args()
    t0 = time.perf_counter()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(
        cfg,
        {
            "input_file": args.input_file,
            "input_key": args.input_key,
            "output_dir": args.output_dir,
            "outname": args.outname,
            "output_key": args.output_key,
        },
    )

    input_file = Path(cfg["input_file"])
    input_key = cfg.get("input_key", "vaep_data")
    output_dir = Path(cfg.get("output_dir", "data/feat_engineered_vaep_data"))
    outname = cfg.get("outname", input_file.name)
    output_key = cfg.get("output_key", "feat_engineered_vaep_data")
    output_path = output_dir / outname

    logger.info("Configuration:")
    logger.info("  input_file  = %s", input_file)
    logger.info("  input_key   = %s", input_key)
    logger.info("  output_path = %s", output_path)
    logger.info("  output_key  = %s", output_key)

    df = read_h5_table(input_file, input_key)
    logger.info("Loaded %s — shape %s", input_file, df.shape)

    if args.max_games is not None:
        game_ids = sorted(df["game_id"].unique())[: args.max_games]
        df = df[df["game_id"].isin(game_ids)].reset_index(drop=True)
        logger.info("Subset: keeping %d games → %d rows", len(game_ids), len(df))

    engineered_df, feature_cols, categorical_cols = build_scores_engineered_dataset(df)
    save_vaep_dataset(engineered_df, output_path, key=output_key)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done in %.1fs — output %s (%d rows, %d cols, %d features, %d categorical)",
        elapsed,
        output_path,
        len(engineered_df),
        len(engineered_df.columns),
        len(feature_cols),
        len(categorical_cols),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
