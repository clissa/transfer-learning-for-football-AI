"""YAML configuration loading, CLI-override merging, and seed helpers.

Provides a minimal config layer: YAML files as the single source of truth,
with optional argparse overrides on top.  No Hydra, no OmegaConf — just
plain ``pyyaml`` + stdlib.

Usage in scripts::

    from football_ai.config import load_config, merge_cli_overrides, resolve_random_state

    cfg = load_config("configs/train_sklearn.yaml")
    cfg = merge_cli_overrides(cfg, {"model.name": "rf", "train.seed": 42})
    seed = resolve_random_state(args.seed, cfg["model"].get("random_state"))
"""
from __future__ import annotations

import datetime
import logging
import random
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def setup_logging(
    script_name: str,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> Path:
    """Configure root logger with console + rotating file handler.

    Parameters
    ----------
    script_name : str
        Base name for the log file (e.g. ``"train_xgboost"``).
    log_dir : str | Path
        Directory where log files are written.
    level : int
        Logging level for both handlers.

    Returns
    -------
    Path
        Path to the created log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"logfile_{script_name}_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        root.addHandler(console)

    # File handler
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logger.info("Logging to %s", log_file)
    return log_file


def resolve_random_state(*candidates: int | str | None) -> int:
    """Return the first valid candidate, or today's date as YYYYMMDD, or a random sample.

    Priority order:
    1. First non-``None`` integer candidate.
    2. Candidate value "today" → today's date as YYYYMMDD integer.
    3. All candidates are ``None`` → random integer sampled from ``[0, 1_000_000)``.

    Parameters
    ----------
    *candidates : int | str | None
        Seed candidates in priority order (e.g. CLI arg, YAML value,
        module-level constant).  The first that is not ``None`` wins.

    Returns
    -------
    int
        Resolved random state.

    Examples
    --------
    >>> resolve_random_state(42)
    42
    >>> resolve_random_state(None, 99)
    99
    >>> resolve_random_state("today")  # doctest: +SKIP
    20260308
    >>> resolve_random_state(None, None)  # random int  # doctest: +SKIP
    123456
    """
    
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, str):
            stripped = c.strip().lower()
            if stripped == "today":
                fallback = int(datetime.date.today().strftime("%Y%m%d"))
                logger.info("Using today's date as seed: %d", fallback)
                return fallback
            if stripped in ("", "none"):
                continue
            # Numeric string, e.g. "42"
            try:
                return int(stripped)
            except ValueError:
                continue
        if isinstance(c, (int, float)):
            return int(c)
    
    # All candidates were None or invalid; sample random seed
    fallback = random.randint(0, 1_000_000 - 1)
    logger.warning("No random_state specified; sampled random seed: %d", fallback)
    return fallback


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a nested dict.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected a YAML mapping at top level, got {type(cfg).__name__}")
    return cfg


def merge_cli_overrides(
    cfg: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge flat CLI overrides into a (possibly nested) config dict.

    Keys in *overrides* use dot-separated paths to address nested values,
    e.g. ``"model.name"`` sets ``cfg["model"]["name"]``.  Top-level keys
    (no dot) are set directly.  Only non-``None`` values are applied so that
    argparse defaults don't clobber YAML values.

    Parameters
    ----------
    cfg : dict[str, Any]
        Base configuration (mutated in place and returned).
    overrides : dict[str, Any]
        Flat mapping of ``dotted.key -> value``.  ``None`` values are skipped.

    Returns
    -------
    dict[str, Any]
        The same *cfg* dict, updated.
    """
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        parts = dotted_key.split(".")
        target = cfg
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return cfg
