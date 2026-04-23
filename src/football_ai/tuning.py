"""Reusable XGBoost tuning routines for football VAEP experiments.

This module owns the Optuna objective, result dumping, and post-study
diagnostics.  CLI entrypoints should only parse YAML/argparse inputs and call
``run_xgboost_tuning``.
"""
from __future__ import annotations

import copy
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .config import resolve_random_state
from .evaluation import evaluate_binary, get_positive_class_scores, sweep_thresholds_for_f1
from .training import (
    ResolvedCompetitionSeasonSplit,
    ResolvedSplitFrame,
    drop_none_params,
    load_xy_competition_season_split,
    save_model,
)

logger = logging.getLogger(__name__)

optuna: Any | None = None
XGBClassifier: Any | None = None

SELECTED_METRIC = "roc_auc"
DEFAULT_CONFIG_PATH = Path("configs/tune_xgboost.yaml")
DEFAULT_KEY_CANDIDATES = ["feat_engineered_vaep_data", "vaep_data"]


@dataclass(frozen=True)
class XGBoostTuningResult:
    """Summary of one completed XGBoost tuning run."""

    run_dir: Path
    best_trial_number: int
    best_score: float
    best_params: dict[str, Any]
    manifest_path: Path


def _require_optuna() -> Any:
    global optuna
    if optuna is None:
        try:
            import optuna as optuna_module
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "optuna is required for Bayesian tuning. Install with: pip install optuna"
            ) from exc
        optuna = optuna_module
    return optuna


def _require_xgb_classifier() -> Any:
    global XGBClassifier
    if XGBClassifier is None:
        try:
            from xgboost import XGBClassifier as xgb_classifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "xgboost is required for XGBoost tuning. Install with: pip install xgboost"
            ) from exc
        XGBClassifier = xgb_classifier
    return XGBClassifier


def normalize_key_candidates(raw_value: str | Sequence[str]) -> list[str]:
    """Return HDF5 key candidates as a list of strings."""
    if isinstance(raw_value, str):
        return [raw_value]
    return [str(item) for item in raw_value]


def resolve_xgboost_tuning_config(
    cfg: Mapping[str, Any],
    cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge YAML config with CLI overrides and enforce ROC-AUC tuning semantics.

    Args:
        cfg: Parsed YAML mapping.
        cli_overrides: Optional keys from argparse. Supported keys are
            ``target_col``, ``data_file``, ``output_dir``, ``n_trials``,
            ``timeout_seconds``, ``seed``, and ``device``.

    Returns:
        Effective config used by the tuning run.
    """
    effective = copy.deepcopy(dict(cfg))
    overrides = dict(cli_overrides or {})

    effective.setdefault("data", {})
    effective.setdefault("split", {})
    effective.setdefault("optuna", {})
    effective.setdefault("training", {})
    effective.setdefault("model", {})
    effective.setdefault("threshold", {})
    effective.setdefault("output", {})
    effective.setdefault("search_space", {})

    if overrides.get("target_col") is not None:
        effective["data"]["target_col"] = overrides["target_col"]
    if overrides.get("data_file") is not None:
        effective["data"]["file"] = overrides["data_file"]
    if overrides.get("output_dir") is not None:
        effective["output"]["root"] = overrides["output_dir"]
    if overrides.get("n_trials") is not None:
        effective["optuna"]["n_trials"] = overrides["n_trials"]
    if overrides.get("timeout_seconds") is not None:
        effective["optuna"]["timeout_seconds"] = overrides["timeout_seconds"]
    if overrides.get("seed") is not None:
        effective["seed"] = overrides["seed"]
    if overrides.get("device") is not None:
        effective["device"] = overrides["device"]

    selected_metric = effective.get("selected_metric", SELECTED_METRIC)
    if selected_metric != SELECTED_METRIC:
        raise ValueError("XGBoost tuning ranks trials only by source-validation roc_auc")
    effective["selected_metric"] = SELECTED_METRIC

    effective["data"].setdefault("file", "data/feat_engineered_vaep_data/major_leagues_vaep.h5")
    effective["data"]["key_candidates"] = normalize_key_candidates(
        effective["data"].get("key_candidates", DEFAULT_KEY_CANDIDATES)
    )
    effective["data"].setdefault("target_col", "scores")

    seed = resolve_random_state(effective.get("seed"))
    effective["seed"] = seed
    device = str(effective.get("device", "cpu"))
    effective["device"] = device

    effective["split"].setdefault("validation_frac", 0.2)
    effective["optuna"].setdefault("n_trials", 300)
    effective["optuna"].setdefault("timeout_seconds", None)
    effective["training"].setdefault("early_stopping_rounds", 100)

    threshold = effective["threshold"]
    threshold.setdefault("min", 0.05)
    threshold.setdefault("max", 0.95)
    threshold.setdefault("steps", 90)

    effective["output"].setdefault("root", "results/xgboost_tuning/roc_auc_scores")

    base_params = dict(effective["model"].get("base_params", {}))
    base_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "device": device,
            "random_state": seed,
            "enable_categorical": True,
            "early_stopping_rounds": int(effective["training"]["early_stopping_rounds"]),
        }
    )
    effective["model"]["base_params"] = base_params
    return effective


def split_rules_from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Extract competition-season split rules for ``load_xy_competition_season_split``."""
    split_cfg = cfg.get("split", {})
    return {
        key: copy.deepcopy(split_cfg[key])
        for key in ("source", "calib", "test")
        if key in split_cfg
    }


def _space_range(
    search_space: Mapping[str, Any],
    name: str,
    default_low: float,
    default_high: float,
    default_log: bool = False,
) -> tuple[float, float, bool]:
    raw = search_space.get(name, {})
    if isinstance(raw, Mapping):
        return (
            raw.get("low", default_low),
            raw.get("high", default_high),
            bool(raw.get("log", default_log)),
        )
    return default_low, default_high, default_log


def sample_xgboost_tuning_params(
    trial: Any,
    *,
    base_params: Mapping[str, Any],
    search_space: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample one XGBoost parameter set from the configured Optuna space."""
    params = dict(base_params)

    low, high, log = _space_range(search_space, "n_estimators", 100, 5000, True)
    params["n_estimators"] = trial.suggest_int(
        "n_estimators", int(low), int(high), log=log
    )

    low, high, log = _space_range(search_space, "learning_rate", 0.01, 0.15, True)
    params["learning_rate"] = trial.suggest_float(
        "learning_rate", float(low), float(high), log=log
    )

    grow_choices = search_space.get("grow_policy", ["depthwise", "lossguide"])
    grow_policy = trial.suggest_categorical("grow_policy", list(grow_choices))
    params["grow_policy"] = grow_policy
    if grow_policy == "lossguide":
        params["max_depth"] = 0
        low, high, _ = _space_range(search_space, "max_leaves", 31, 511, False)
        params["max_leaves"] = trial.suggest_int("max_leaves", int(low), int(high))
    else:
        low, high, _ = _space_range(search_space, "max_depth", 4, 10, False)
        params["max_depth"] = trial.suggest_int("max_depth", int(low), int(high))
        params.pop("max_leaves", None)

    for name, default_low, default_high, default_log in (
        ("min_child_weight", 1.0, 64.0, True),
        ("gamma", 0.0, 10.0, False),
        ("subsample", 0.5, 1.0, False),
        ("colsample_bytree", 0.4, 1.0, False),
        ("reg_alpha", 1e-3, 5.0, True),
        ("reg_lambda", 0.5, 20.0, True),
        ("scale_pos_weight", 10.0, 120.0, False),
    ):
        low, high, log = _space_range(search_space, name, default_low, default_high, default_log)
        params[name] = trial.suggest_float(name, float(low), float(high), log=log)

    low, high, _ = _space_range(search_space, "max_delta_step", 0, 10, False)
    params["max_delta_step"] = trial.suggest_int("max_delta_step", int(low), int(high))

    max_bin = search_space.get("max_bin", params.get("max_bin", 256))
    params["max_bin"] = int(max_bin)
    return drop_none_params(params)


def empirical_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Return negative/positive imbalance for a binary target."""
    y_arr = np.asarray(y)
    positives = int(np.sum(y_arr == 1))
    negatives = int(np.sum(y_arr == 0))
    if positives == 0:
        return float("inf")
    return float(negatives / positives)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(_jsonable(dict(payload)), fp, indent=2, sort_keys=True)


def _safe_evaluate_binary(
    y_score: np.ndarray,
    y_true: pd.Series | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    try:
        return evaluate_binary(y_score, y_true, threshold=threshold)
    except ValueError:
        y_arr = np.asarray(y_true)
        y_pred = (y_score >= threshold).astype(int)
        precision = float(np.sum((y_pred == 1) & (y_arr == 1)) / max(np.sum(y_pred == 1), 1))
        recall = float(np.sum((y_pred == 1) & (y_arr == 1)) / max(np.sum(y_arr == 1), 1))
        f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
        return {
            "rows": float(len(y_arr)),
            "positive_rate": float(y_arr.mean()) if len(y_arr) else float("nan"),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": float(np.mean((y_arr - y_score) ** 2)) if len(y_arr) else float("nan"),
            "logloss": float("nan"),
        }


def source_val_metrics_by_season(
    source_val: ResolvedSplitFrame,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute source-validation diagnostics grouped by competition and season."""
    group_cols = [col for col in ("competition_name", "season_name") if col in source_val.df.columns]
    if not group_cols:
        group_cols = ["split"]
        frame = pd.DataFrame({"split": [source_val.name] * len(source_val.y)})
    else:
        frame = source_val.df.loc[source_val.y.index, group_cols].reset_index(drop=True).copy()
    frame["__y_true"] = np.asarray(source_val.y)
    frame["__y_score"] = np.asarray(y_score)

    rows: list[dict[str, Any]] = []
    for group_values, group_df in frame.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        metrics = _safe_evaluate_binary(
            group_df["__y_score"].to_numpy(),
            group_df["__y_true"].to_numpy(),
            threshold=threshold,
        )
        row = {col: value for col, value in zip(group_cols, group_values)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_xgboost_tuning_objective(
    *,
    resolved_split: ResolvedCompetitionSeasonSplit,
    base_params: Mapping[str, Any],
    search_space: Mapping[str, Any],
    run_dir: Path,
    source_train_empirical_spw: float,
) -> Any:
    """Build an Optuna objective using only source-train and source-val splits."""
    X_train = resolved_split.source_train.X
    y_train = resolved_split.source_train.y.astype(np.uint8)
    X_val = resolved_split.source_val.X.reindex(columns=list(X_train.columns))
    y_val = resolved_split.source_val.y.astype(np.uint8)

    def objective(trial: Any) -> float:
        params = sample_xgboost_tuning_params(
            trial,
            base_params=base_params,
            search_space=search_space,
        )
        model = _require_xgb_classifier()(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_val_score = get_positive_class_scores(model, X_val)
        metrics = evaluate_binary(y_val_score, y_val)

        trial.set_user_attr("source_val_roc_auc", metrics["roc_auc"])
        trial.set_user_attr("source_val_pr_auc", metrics["pr_auc"])
        trial.set_user_attr("source_val_logloss", metrics["logloss"])
        trial.set_user_attr("source_val_brier", metrics["brier"])
        trial.set_user_attr("source_train_empirical_scale_pos_weight", source_train_empirical_spw)
        trial.set_user_attr("xgb_params", _jsonable(params))

        _write_json(run_dir / f"trial_{trial.number}_source_val_metrics.json", metrics)
        _write_json(run_dir / f"trial_{trial.number}_params.json", params)
        source_val_metrics_by_season(resolved_split.source_val, y_val_score).to_csv(
            run_dir / f"trial_{trial.number}_la_liga_val_by_season.csv",
            index=False,
        )
        return float(metrics[SELECTED_METRIC])

    return objective


def detect_h5_key_used(data_file: Path, key_candidates: Sequence[str]) -> str | None:
    """Return the first available HDF5 key from *key_candidates*, if inspectable."""
    if not data_file.exists():
        return None
    with pd.HDFStore(str(data_file), mode="r") as store:
        available = set(store.keys())
        for key in key_candidates:
            if f"/{key}" in available:
                return key
    return None


def collect_runtime_versions() -> dict[str, Any]:
    """Collect package and runtime versions for the run manifest."""
    packages = {}
    for package_name in ("numpy", "pandas", "scikit-learn", "xgboost", "optuna"):
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _study_trials_dataframe(study: Any) -> pd.DataFrame:
    try:
        return study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    except TypeError:
        return study.trials_dataframe()


def write_post_study_diagnostics(
    *,
    model: Any,
    resolved_split: ResolvedCompetitionSeasonSplit,
    run_dir: Path,
    threshold_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Write best-model threshold, diagnostics, coverage, and feature artifacts."""
    feature_cols = list(resolved_split.source_train.X.columns)
    X_val = resolved_split.source_val.X.reindex(columns=feature_cols)
    y_val_score = get_positive_class_scores(model, X_val)

    sweep_df, selected_threshold = sweep_thresholds_for_f1(
        y_true=resolved_split.source_val.y,
        y_score=y_val_score,
        threshold_min=float(threshold_cfg.get("min", 0.05)),
        threshold_max=float(threshold_cfg.get("max", 0.95)),
        threshold_steps=int(threshold_cfg.get("steps", 90)),
    )
    threshold_sweep_path = run_dir / "best_threshold_sweep_source_val.csv"
    sweep_df.to_csv(threshold_sweep_path, index=False)

    diagnostics: list[dict[str, Any]] = []
    for split_name, split_frame in resolved_split.named_splits(include_lazy=True).items():
        split_frame.competition_seasons.to_csv(
            run_dir / f"coverage_{split_name}.csv",
            index=False,
        )
        X_eval = split_frame.X.reindex(columns=feature_cols)
        y_score = get_positive_class_scores(model, X_eval)
        metrics = _safe_evaluate_binary(
            y_score,
            split_frame.y.astype(np.uint8),
            threshold=selected_threshold,
        )
        diagnostics.append(
            {
                "split": split_name,
                "rows": len(split_frame.X),
                "games": split_frame.n_games,
                "selected_threshold": selected_threshold,
                "selected_threshold_source": "source_val_max_f1",
                **metrics,
            }
        )

    diagnostics_path = run_dir / "final_diagnostics_by_split.csv"
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)

    feature_columns_path = run_dir / "feature_columns.json"
    _write_json(
        feature_columns_path,
        {
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
        },
    )
    return {
        "selected_threshold": float(selected_threshold),
        "threshold_sweep_path": threshold_sweep_path,
        "final_diagnostics_path": diagnostics_path,
        "feature_columns_path": feature_columns_path,
    }


def run_xgboost_tuning(
    cfg: Mapping[str, Any],
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    run_timestamp: datetime | None = None,
) -> XGBoostTuningResult:
    """Run an ROC-AUC-ranked Optuna tuning study and write all artifacts."""
    optuna_module = _require_optuna()
    effective_cfg = resolve_xgboost_tuning_config(cfg, cli_overrides=cli_overrides)

    seed = int(effective_cfg["seed"])
    data_file = Path(effective_cfg["data"]["file"])
    key_candidates = normalize_key_candidates(effective_cfg["data"]["key_candidates"])
    target_col = str(effective_cfg["data"]["target_col"])
    split_rules = split_rules_from_config(effective_cfg)
    validation_frac = float(effective_cfg["split"].get("validation_frac", 0.2))
    output_root = Path(effective_cfg["output"]["root"])
    timestamp = run_timestamp or datetime.now()
    run_id = timestamp.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{run_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    effective_config_path = run_dir / "effective_config.yaml"
    with effective_config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(_jsonable(effective_cfg), fp, sort_keys=False)

    data_key_used = detect_h5_key_used(data_file, key_candidates)
    resolved_split = load_xy_competition_season_split(
        target_col=target_col,
        data_file=data_file,
        key_candidates=key_candidates,
        split_config=split_rules,
        validation_frac=validation_frac,
        random_state=seed,
    )

    source_train_spw = empirical_scale_pos_weight(resolved_split.source_train.y)
    logger.info("Source-train empirical scale_pos_weight: %.6f", source_train_spw)

    base_params = drop_none_params(effective_cfg["model"]["base_params"])
    search_space = effective_cfg["search_space"]
    objective = build_xgboost_tuning_objective(
        resolved_split=resolved_split,
        base_params=base_params,
        search_space=search_space,
        run_dir=run_dir,
        source_train_empirical_spw=source_train_spw,
    )

    sampler = optuna_module.samplers.TPESampler(seed=seed)
    study = optuna_module.create_study(
        study_name=f"xgb_roc_auc_{target_col}_{run_id}_seed{seed}",
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=int(effective_cfg["optuna"]["n_trials"]),
        timeout=effective_cfg["optuna"].get("timeout_seconds"),
        n_jobs=1,
    )

    trials_df = _study_trials_dataframe(study)
    if "value" in trials_df.columns:
        trials_df = trials_df.sort_values("value", ascending=False).reset_index(drop=True)
    trials_path = run_dir / "trials.csv"
    trials_df.to_csv(trials_path, index=False)

    best_params = dict(study.best_trial.user_attrs.get("xgb_params") or {})
    if not best_params:
        best_params = sample_xgboost_tuning_params(
            study.best_trial,
            base_params=base_params,
            search_space=search_space,
        )
    best_params = drop_none_params(best_params)
    best_params_path = run_dir / "best_params.json"
    _write_json(best_params_path, best_params)

    feature_cols = list(resolved_split.source_train.X.columns)
    best_model = _require_xgb_classifier()(**best_params)
    best_model.fit(
        resolved_split.source_train.X,
        resolved_split.source_train.y.astype(np.uint8),
        eval_set=[
            (
                resolved_split.source_val.X.reindex(columns=feature_cols),
                resolved_split.source_val.y.astype(np.uint8),
            )
        ],
        verbose=False,
    )
    model_path = run_dir / "best_model.pkl"
    save_model(best_model, model_path)

    diagnostic_outputs = write_post_study_diagnostics(
        model=best_model,
        resolved_split=resolved_split,
        run_dir=run_dir,
        threshold_cfg=effective_cfg["threshold"],
    )

    artifact_paths = {
        "effective_config": effective_config_path,
        "trials": trials_path,
        "best_params": best_params_path,
        "best_model": model_path,
        "threshold_sweep": diagnostic_outputs["threshold_sweep_path"],
        "final_diagnostics": diagnostic_outputs["final_diagnostics_path"],
        "feature_columns": diagnostic_outputs["feature_columns_path"],
    }
    coverage_paths = {
        split_name: run_dir / f"coverage_{split_name}.csv"
        for split_name in resolved_split.named_splits(include_lazy=True)
    }
    artifact_paths["coverage"] = coverage_paths

    manifest = {
        "data_file": data_file,
        "hdf5_key_candidates": key_candidates,
        "hdf5_key_used": data_key_used,
        "target_col": target_col,
        "split_config": effective_cfg["split"],
        "seed": seed,
        "device": effective_cfg["device"],
        "selected_metric": SELECTED_METRIC,
        "best_trial_number": study.best_trial.number,
        "best_source_val_roc_auc": float(study.best_value),
        "selected_threshold": diagnostic_outputs["selected_threshold"],
        "source_train_empirical_scale_pos_weight": source_train_spw,
        "artifact_paths": artifact_paths,
        "runtime_versions": collect_runtime_versions(),
    }
    manifest_path = run_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)

    logger.info(
        "Best trial %s source-val ROC-AUC %.6f; outputs in %s",
        study.best_trial.number,
        float(study.best_value),
        run_dir,
    )
    return XGBoostTuningResult(
        run_dir=run_dir,
        best_trial_number=int(study.best_trial.number),
        best_score=float(study.best_value),
        best_params=best_params,
        manifest_path=manifest_path,
    )
