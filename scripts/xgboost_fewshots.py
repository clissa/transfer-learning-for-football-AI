"""Few-shot XGBoost domain-adaptation experiment.

Compares three strategies for adapting a source-trained XGBoost model to a
new target domain at varying labelling budgets (1 %, 5 %, 20 % of target
games):

1. **Source-only** — evaluate the pre-trained model on the target set
   without any re-training (zero-shot baseline).
2. **Target-only** — train a fresh XGBoost from scratch on the few-shot
   subset of target games.
3. **Fine-tune** — continue boosting from the pre-trained model on the
   few-shot subset (warm-start via ``xgb_model``).

Budget fractions refer to unique ``game_id`` values (not individual
actions) to prevent within-match leakage.  Each budget × scenario is
repeated over multiple random seeds; the script reports mean ± std.

Outputs
-------
- ``fewshot_all_seeds_{target_col}.csv``   — per-seed raw metrics.
- ``fewshot_summary_{target_col}.csv``     — aggregated table (mean ± std).
- ``fewshot_curves_{target_col}.png``      — 2×2 few-shot learning curves.
- ``fewshot_table_{target_col}.csv``       — presentation-ready summary.

Usage
-----
# Pre-train source model first (if not done):
python -m scripts.train_xgboost --config configs/train_xgboost.yaml

# Run few-shot experiment:
python -m scripts.xgboost-few-shots --config configs/xgboost_fewshot.yaml
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from football_ai.config import load_config, resolve_random_state, setup_logging
from football_ai.evaluation import (
    evaluate_binary,
    get_positive_class_scores,
    sweep_thresholds_for_f1,
)
from football_ai.training import (
    drop_none_params,
    load_fewshot_splits,
    load_model,
    resolve_xgb_eval_metrics,
    sample_target_games,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Module-level defaults
# ──────────────────────────────────────────────
DATA_FILE = Path("data/feat_engineered_vaep_data/major_leagues_vaep.h5")
DATA_KEY = ["feat_engineered_vaep_data", "vaep_data"]
TARGET_COL = "scores"

SOURCE_COMPETITIONS = ["Premier League", "La Liga", "1. Bundesliga"]
TARGET_COMPETITIONS = ["Champions League", "UEFA Europa League"]
VALIDATION_FRAC = 0.2
RANDOM_STATE: int | None = 20260307

BUDGETS = [0.01, 0.05, 0.20]
SEEDS = [1, 2, 3, 4, 5]
# NOTE: update this path when re-training with a different seed
SOURCE_MODEL_PATH = Path("models/xgboost_scores_20260307.pkl")

RESULTS_PATH = Path("results/fewshot_xgboost")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Few-shot XGBoost domain-adaptation experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--source-model", type=str, default=None, help="Pre-trained model path")
    p.add_argument("--target-col", type=str, default=None, help="scores or concedes")
    p.add_argument("--output-dir", type=str, default=None, help="Results directory")
    p.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return p.parse_args()


def _normalize_key_candidates(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        return [raw_value]
    return [str(item) for item in raw_value]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _build_base_model_params(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build fresh XGBClassifier kwargs from config, dropping None values.

    XGBoost callback instances are stateful across training runs.  The
    few-shot loop trains many models sequentially, so we deep-copy nested
    params (especially ``callbacks``) to avoid callback state leakage.
    """
    return copy.deepcopy(drop_none_params(model_cfg))


def _train_from_scratch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    model_params: dict[str, Any],
) -> XGBClassifier:
    """Train a fresh XGBClassifier on the provided data."""
    model = XGBClassifier(**model_params)
    fit_kw: dict[str, Any] = {"verbose": False}
    if X_val is not None and len(X_val) > 0:
        fit_kw["eval_set"] = [(X_val, y_val)]
    model.fit(X_train, y_train, **fit_kw)
    return model


def _finetune_from_pretrained(
    pretrained_model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    finetune_params: dict[str, Any],
) -> XGBClassifier:
    """Continue boosting from a pre-trained XGBoost model (warm-start)."""
    model = XGBClassifier(**finetune_params)
    fit_kw: dict[str, Any] = {
        "verbose": False,
        "xgb_model": pretrained_model.get_booster(),
    }
    if X_val is not None and len(X_val) > 0:
        fit_kw["eval_set"] = [(X_val, y_val)]
    model.fit(X_train, y_train, **fit_kw)
    return model


def _evaluate(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Score a model and return metrics dict."""
    y_proba = get_positive_class_scores(model, X)
    return evaluate_binary(y_proba, y, threshold=threshold)


def _select_threshold(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thr_cfg: dict,
) -> float:
    """Sweep thresholds on validation set; fall back to 0.5 when val is tiny."""
    if len(X_val) < 30:
        return float(thr_cfg.get("pred_threshold", 0.5))
    y_proba = get_positive_class_scores(model, X_val)
    _, best = sweep_thresholds_for_f1(
        y_val, y_proba,
        threshold_min=float(thr_cfg.get("min", 0.05)),
        threshold_max=float(thr_cfg.get("max", 0.95)),
        threshold_steps=int(thr_cfg.get("steps", 90)),
    )
    return best


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def _plot_fewshot_curves(
    summary_df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
) -> None:
    """Save a 2×2 panel of few-shot learning curves (3 scenarios)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)

    metrics_to_plot = [
        ("pr_auc", "PR-AUC", True),
        ("roc_auc", "ROC-AUC", True),
        ("brier", "Brier Score", False),   # lower is better
        ("f1", "F1 Score", True),
    ]

    budgets_pct = sorted(summary_df["budget"].unique())
    x_labels = [f"{b*100:.0f}%" for b in budgets_pct]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()

    # (color, linestyle, nice label)
    palette = {
        "source_only": ("#2ca02c", "--", "Source-only (0-shot)"),
        "target_only": ("#1f77b4", "-",  "Target-only"),
        "finetune":    ("#ff7f0e", "-",  "Fine-tune"),
    }

    for ax, (metric_key, metric_label, higher_better) in zip(axes, metrics_to_plot):
        for scenario, (color, ls, nice) in palette.items():
            sub = summary_df[summary_df["scenario"] == scenario].sort_values("budget")
            if sub.empty:
                continue
            means = sub[f"{metric_key}_mean"].values
            stds = sub[f"{metric_key}_std"].values
            xs = range(len(budgets_pct))
            ax.plot(xs, means, marker="o", color=color, ls=ls, lw=2, label=nice)
            ax.fill_between(xs, means - stds, means + stds, alpha=0.18, color=color)

        ax.set_xticks(range(len(budgets_pct)))
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Few-shot domain adaptation — {target_col}",
        fontsize=14, weight="bold", y=1.01,
    )
    fig.supxlabel("Target-game budget", fontsize=11)
    plt.tight_layout()
    out = output_dir / f"fewshot_curves_{target_col}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved few-shot curves -> %s", out)


def _print_and_save_table(
    summary_df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Build, print and save the presentation-ready summary table."""
    display_metrics = ["pr_auc", "roc_auc", "brier", "f1", "logloss"]
    rows: list[dict[str, Any]] = []

    nice_names = {
        "source_only": "Source-only (0-shot)",
        "target_only": "Target-only",
        "finetune": "Fine-tune (xgb_model)",
    }

    budgets_pct = sorted(summary_df["budget"].unique())
    for budget in budgets_pct:
        budget_label = f"{budget*100:.0f}%"
        for scenario in ["source_only", "target_only", "finetune"]:
            sub = summary_df[
                (summary_df["budget"] == budget) & (summary_df["scenario"] == scenario)
            ]
            if sub.empty:
                continue
            r: dict[str, Any] = {"Model": nice_names[scenario], "Budget target": budget_label}
            for m in display_metrics:
                col_name = _col_display_name(m)
                mean_val = sub[f"{m}_mean"].values[0]
                std_val = sub[f"{m}_std"].values[0]
                r[col_name] = f"{mean_val:.4f} ± {std_val:.4f}"
            rows.append(r)

    table_df = pd.DataFrame(rows)
    out = output_dir / f"fewshot_table_{target_col}.csv"
    table_df.to_csv(out, index=False)
    logger.info("Saved summary table -> %s", out)

    # Pretty print
    logger.info("\n" + "=" * 100)
    logger.info("  Few-shot summary — %s", target_col)
    logger.info("=" * 100)
    try:
        logger.info("\n%s", table_df.to_string(index=False))
    except Exception:
        logger.info("%s", table_df)
    logger.info("=" * 100 + "\n")
    return table_df


def _col_display_name(metric_key: str) -> str:
    mapping = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "brier": "BRIER",
        "f1": "F1",
        "logloss": "LOGLOSS",
    }
    return mapping.get(metric_key, metric_key.upper())


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    setup_logging("xgboost_fewshots")

    # ── config resolution ──
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_config(args.config)

    data_file = Path(args.data_file if hasattr(args, "data_file") and args.data_file
                     else cfg.get("data", {}).get("file", str(DATA_FILE)))
    key_candidates = _normalize_key_candidates(
        cfg.get("data", {}).get("key_candidates", DATA_KEY)
    )
    target_col: str = args.target_col or cfg.get("data", {}).get("target_col", TARGET_COL)

    split_cfg = cfg.get("split", {})
    source_competitions = split_cfg.get("source_competitions", SOURCE_COMPETITIONS)
    target_competitions = split_cfg.get("target_competitions", TARGET_COMPETITIONS)
    validation_frac = float(split_cfg.get("validation_frac", VALIDATION_FRAC))
    random_state = resolve_random_state(split_cfg.get("random_state"), RANDOM_STATE)

    fewshot_cfg = cfg.get("fewshot", {})
    budgets: list[float] = [float(b) for b in fewshot_cfg.get("budgets", BUDGETS)]
    seeds: list[int] = [int(s) for s in fewshot_cfg.get("seeds", SEEDS)]
    source_model_path = Path(
        args.source_model
        or fewshot_cfg.get("source_model_path", str(SOURCE_MODEL_PATH))
    )

    # Model config (from-scratch)
    model_cfg_yaml = cfg.get("model", {})
    base_model_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cpu",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": random_state,
        "max_depth": 6,
        "max_bin": 256,
        "grow_policy": "depthwise",
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "max_delta_step": 1.0,
        "subsample": 0.8,
        "sampling_method": "uniform",
        "colsample_bytree": 0.8,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 80,
        "eval_metric": ["aucpr", "auc", "logloss"],
        "early_stopping_rounds": 50,
        "enable_categorical": False,
        "importance_type": "gain",
        "validate_parameters": True,
    }
    for k, v in model_cfg_yaml.items():
        if v is not None:
            base_model_params[k] = v
    base_model_params["verbosity"] = 0  # keep output clean
    if args.device:
        base_model_params["device"] = args.device

    # ── Resolve custom eval metrics for base model (f1, precision, recall → callables) ──
    early_stopping_metric_model = model_cfg_yaml.get("early_stopping_metric")
    base_model_params.pop("early_stopping_metric", None)
    resolved_m, cbs_m, es_rounds_m = resolve_xgb_eval_metrics(
        base_model_params.get("eval_metric"),
        early_stopping_rounds=base_model_params.get("early_stopping_rounds"),
        early_stopping_metric=early_stopping_metric_model,
    )
    base_model_params["eval_metric"] = resolved_m
    base_model_params["early_stopping_rounds"] = es_rounds_m
    if cbs_m:
        existing_cbs = base_model_params.get("callbacks") or []
        base_model_params["callbacks"] = cbs_m + list(existing_cbs)

    # Fine-tune overrides (apply ALL finetune YAML keys, not just a few)
    ft_cfg = cfg.get("finetune", {})
    finetune_params = dict(base_model_params)
    for k, v in ft_cfg.items():
        if v is not None:
            finetune_params[k] = v

    # ── Resolve custom eval metrics for fine-tune ──
    early_stopping_metric_ft = ft_cfg.get("early_stopping_metric", early_stopping_metric_model)
    finetune_params.pop("early_stopping_metric", None)
    resolved_ft, cbs_ft, es_rounds_ft = resolve_xgb_eval_metrics(
        finetune_params.get("eval_metric"),
        early_stopping_rounds=finetune_params.get("early_stopping_rounds"),
        early_stopping_metric=early_stopping_metric_ft,
    )
    finetune_params["eval_metric"] = resolved_ft
    finetune_params["early_stopping_rounds"] = es_rounds_ft
    if cbs_ft:
        existing_ft = finetune_params.get("callbacks") or []
        finetune_params["callbacks"] = cbs_ft + list(existing_ft)

    thr_cfg = cfg.get("threshold", {})

    results_dir = Path(
        args.output_dir
        or cfg.get("output", {}).get("dir", str(RESULTS_PATH))
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    logger.info("Loading data …")
    (
        X_source_train, y_source_train,
        X_source_val, y_source_val,
        X_target, y_target,
        target_game_ids,
        feature_cols,
    ) = load_fewshot_splits(
        target_col=target_col,
        data_file=data_file,
        key_candidates=key_candidates,
        source_competitions=source_competitions,
        target_competitions=target_competitions,
        validation_frac=validation_frac,
        random_state=random_state,
    )

    y_source_train = y_source_train.astype(np.uint8)
    y_source_val = y_source_val.astype(np.uint8)
    y_target = y_target.astype(np.uint8)

    n_target_games = target_game_ids.nunique()
    logger.info("Source train: %s actions", f"{len(X_source_train):,}")
    logger.info("Source val:   %s actions", f"{len(X_source_val):,}")
    logger.info("Target:       %s actions  (%d games)", f"{len(X_target):,}", n_target_games)
    logger.info("Features:     %d", len(feature_cols))
    logger.info("Budgets:      %s", budgets)
    logger.info("Seeds:        %s", seeds)

    # ── Source-only baseline (model + threshold — eval moved inside loop) ──
    logger.info("Loading pre-trained model from %s …", source_model_path)
    pretrained_model: XGBClassifier = load_model(source_model_path)

    # Align source-val columns for threshold selection
    X_source_val_aligned = X_source_val.reindex(columns=feature_cols)

    # Find threshold on source val (used for all zero-shot evals)
    source_threshold = _select_threshold(
        pretrained_model, X_source_val_aligned, y_source_val, thr_cfg,
    )
    logger.info("Source-only threshold (from source val): %.4f", source_threshold)

    # ── Budget × seed loop ──
    all_rows: list[dict[str, Any]] = []
    n_combos = len(budgets) * len(seeds) * 3  # 3 scenarios
    combo_i = 0

    for budget in budgets:
        for seed in seeds:
            # Sample few-shot games
            X_few, y_few, X_hold, y_hold = sample_target_games(
                X_target, y_target, target_game_ids, frac=budget, random_state=seed,
            )
            n_few_games = int(round(n_target_games * budget))

            # Split few-shot into train/val (80/20 by game)
            few_game_ids = target_game_ids.loc[X_few.index]
            few_unique = few_game_ids.unique()
            if len(few_unique) >= 5:
                n_val_games = max(1, int(round(len(few_unique) * 0.2)))
                rng = np.random.RandomState(seed + 1000)
                val_game_set = set(rng.choice(few_unique, size=n_val_games, replace=False))
                few_val_mask = few_game_ids.isin(val_game_set)
                X_few_train = X_few.loc[~few_val_mask]
                y_few_train = y_few.loc[~few_val_mask]
                X_few_val = X_few.loc[few_val_mask]
                y_few_val = y_few.loc[few_val_mask]
            else:
                # Too few games for a game-level split → fall back to a
                # random 20 % action-level split for early-stopping only.
                # Minor within-game leakage is acceptable here.
                logger.warning(
                    "Only %d game(s) in few-shot sample — using a random "
                    "20%% action-level val split (minor leakage possible).",
                    len(few_unique),
                )
                rng = np.random.RandomState(seed + 1000)
                n_val = max(1, int(round(len(X_few) * 0.2)))
                val_idx = rng.choice(X_few.index, size=n_val, replace=False)
                val_mask = X_few.index.isin(val_idx)
                X_few_train = X_few.loc[~val_mask]
                y_few_train = y_few.loc[~val_mask]
                X_few_val = X_few.loc[val_mask]
                y_few_val = y_few.loc[val_mask]

            X_few_train = X_few_train.reindex(columns=feature_cols)
            X_few_val = X_few_val.reindex(columns=feature_cols)
            X_hold = X_hold.reindex(columns=feature_cols)

            # ── 0) Source-only (zero-shot on same holdout) ──
            combo_i += 1
            logger.info(
                "[%d/%d] source-only  budget=%d%%  seed=%d  "
                "(%d holdout actions)",
                combo_i, n_combos, budget * 100, seed, len(X_hold),
            )
            src_metrics = _evaluate(
                pretrained_model, X_hold, y_hold, threshold=source_threshold,
            )
            src_metrics.update({
                "scenario": "source_only",
                "budget": budget,
                "seed": seed,
                "threshold": source_threshold,
                "n_few_games": n_few_games,
                "n_few_actions": len(X_few),
                "n_holdout_actions": len(X_hold),
            })
            all_rows.append(src_metrics)

            # ── 1) Target-only (from scratch) ──
            combo_i += 1
            logger.info(
                "[%d/%d] target-only  budget=%d%%  seed=%d  "
                "(%d train / %d val / %d holdout)",
                combo_i, n_combos, budget * 100, seed,
                len(X_few_train), len(X_few_val), len(X_hold),
            )

            scratch_params = dict(_build_base_model_params(base_model_params))
            scratch_params["random_state"] = seed
            scratch_model = _train_from_scratch(
                X_few_train, y_few_train, X_few_val, y_few_val, scratch_params,
            )
            scratch_thr = _select_threshold(scratch_model, X_few_val, y_few_val, thr_cfg)
            scratch_metrics = _evaluate(scratch_model, X_hold, y_hold, threshold=scratch_thr)
            scratch_metrics.update({
                "scenario": "target_only",
                "budget": budget,
                "seed": seed,
                "threshold": scratch_thr,
                "n_few_games": n_few_games,
                "n_few_actions": len(X_few),
                "n_holdout_actions": len(X_hold),
            })
            all_rows.append(scratch_metrics)

            # ── 2) Fine-tune (warm-start) ──
            combo_i += 1
            logger.info(
                "[%d/%d] fine-tune     budget=%d%%  seed=%d  "
                "(%d train / %d val / %d holdout)",
                combo_i, n_combos, budget * 100, seed,
                len(X_few_train), len(X_few_val), len(X_hold),
            )

            ft_params_this = dict(_build_base_model_params(finetune_params))
            ft_params_this["random_state"] = seed
            ft_model = _finetune_from_pretrained(
                pretrained_model, X_few_train, y_few_train,
                X_few_val, y_few_val, ft_params_this,
            )
            ft_thr = _select_threshold(ft_model, X_few_val, y_few_val, thr_cfg)
            ft_metrics = _evaluate(ft_model, X_hold, y_hold, threshold=ft_thr)
            ft_metrics.update({
                "scenario": "finetune",
                "budget": budget,
                "seed": seed,
                "threshold": ft_thr,
                "n_few_games": n_few_games,
                "n_few_actions": len(X_few),
                "n_holdout_actions": len(X_hold),
            })
            all_rows.append(ft_metrics)

    # ── Aggregate ──
    raw_df = pd.DataFrame(all_rows)
    raw_path = results_dir / f"fewshot_all_seeds_{target_col}.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info("Saved per-seed results -> %s", raw_path)

    metric_keys = ["pr_auc", "roc_auc", "brier", "f1", "logloss",
                    "precision", "recall"]
    agg_dict: dict[str, Any] = {}
    for m in metric_keys:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    agg_dict["n_seeds"] = ("seed", "count")

    summary_df = (
        raw_df
        .groupby(["scenario", "budget"], as_index=False)
        .agg(**agg_dict)
    )
    summary_path = results_dir / f"fewshot_summary_{target_col}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary -> %s", summary_path)

    # ── Table ──
    _print_and_save_table(summary_df, target_col, results_dir)

    # ── Plots ──
    _plot_fewshot_curves(summary_df, target_col, results_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
