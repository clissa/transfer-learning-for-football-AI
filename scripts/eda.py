#!/usr/bin/env python
"""Exploratory Data Analysis for tabular ML on SPADL features.

Thin orchestration script — calls reusable helpers from
``football_ai.data``, ``football_ai.data_viz``, ``football_ai.evaluation``,
and ``football_ai.training``.  All parameters are globals at the top: change
them and re-run the relevant section.

Sections
--------
 0. Config & imports
 1. Data loading
 2. Feature typing & SPADL enrichment
 3. Data quality
 4. Missingness
 5. Target analysis
 6. Leakage suspicion
 7. Univariate (numeric)
 8. Univariate (categorical)
 9. Bivariate (feature vs target)
10. Collinearity
11. Quick baseline models
12. Markdown summary
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # switch to "TkAgg" / "inline" for interactive plots

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from football_ai.data import (
    build_styles_dataframe,
    classify_columns,
    compute_data_quality,
    compute_missingness,
    detect_leakage_suspects,
    read_h5_table,
    sample_dataframe,
)
from football_ai.data_viz import (
    generate_eda_summary_markdown,
    plot_action_distributions,
    plot_bivariate_target_corr,
    plot_correlation_heatmap,
    plot_displacement_histograms,
    plot_feature_importance_comparison,
    plot_missingness_bar,
    plot_target_distribution,
    plot_univariate_categorical_grid,
    plot_univariate_numeric_grid,
)
from football_ai.evaluation import evaluate_binary
from football_ai.training import build_preprocessor, preprocess_split

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ╔════════════════════════════════════════════════╗
# ║  0. CONFIG — change these freely, then re-run ║
# ╚════════════════════════════════════════════════╝

DATA_FILE = Path("data/spadl_full_data/all_leagues.h5")
# DATA_FILE   = Path("data/spadl_full_data/women_leagues.h5")
KEY = "full_data"
TARGET_COL = "scores"  # "scores" or "concedes"
OUTPUT_DIR = Path("data/EDA/spadl")
SAMPLE_FRAC = 1.0  # (0, 1] — set <1 for quick iteration
SEED = 20140916

# Column role definitions (edit as needed)
KNOWN_ID_COLS = {
    "game_id",
    "original_event_id",
    "action_id",
}
KNOWN_META_COLS = {
    "competition_name",
    "country_name",
    "competition_gender",
    "season_name",
    "game_date",
    "venue",
    "referee",
    "player_name",
    "nickname",
    "team_name",
    "starting_position_name",
    "competition_stage",
}
LEAKAGE_SUSPECT_COLS = {
    "home_score",
    "away_score",
    "scores",
    "concedes",
}

# Plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)

# Thresholds
CORR_THRESHOLD = 0.85  # for collinearity pairs
LEAKAGE_CORR_THRESH = 0.5  # for auto-flagging leakage
DOMINANT_CAT_PCT = 80  # % above which a category is "dominant"

# Feature lists (manually curated for this dataset)
# NOTE: home_score / away_score are leakage suspects — excluded from features.
TIMESPACE_FEATS = ["time_seconds", "start_x", "start_y", "end_x", "end_y"]
ACTION_FEATS = ["type_id", "result_id", "bodypart_id"]
TEAM_FEATS = ["is_home"]
PLAYER_FEATS = ["starting_position_id", "is_starter"]

NUM_FEATS = TIMESPACE_FEATS
CAT_FEATS = ACTION_FEATS + TEAM_FEATS + PLAYER_FEATS
ALL_FEATS = NUM_FEATS + CAT_FEATS

# Competition-based split for baselines
TRAINING_COMPETITIONS = [
    "Premier League",
    "La Liga",
    "Copa del Rey",
    "1. Bundesliga",
    "Serie A",
    "Ligue 1",
    "Champions League",
    "UEFA Europa League",
]
VALIDATION_COMPETITIONS = [
    "FA Women's Super League",
    "Women's World Cup",
    "UEFA Women's Euro",
]
TEST_COMPETITIONS = [
    "FIFA World Cup",
    "UEFA Euro",
    "FIFA U20 World Cup",
    "Copa America",
    "Liga Profesional",
    "African Cup of Nations",
    "Indian Super league",
    "Major League Soccer",
    "NWSL",
    "North American League",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════
#  1. DATA LOADING
# ════════════════════════════════════════════════

log.info("Loading %s  key='%s' ...", DATA_FILE, KEY)
df = read_h5_table(DATA_FILE, KEY)
log.info("Loaded %s rows  x  %s columns", f"{len(df):,}", df.shape[1])

if 0 < SAMPLE_FRAC < 1.0:
    df = sample_dataframe(df, frac=SAMPLE_FRAC, seed=SEED)
    log.info("Subsampled to %s rows (frac=%.2f)", f"{len(df):,}", SAMPLE_FRAC)

assert (
    TARGET_COL in df.columns
), f"Target '{TARGET_COL}' not found. Available: {list(df.columns)}"

print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Columns\t\tDtypes:\n{df.dtypes}\n")


# ════════════════════════════════════════════════
#  2. FEATURE TYPING & SPADL ENRICHMENT
# ════════════════════════════════════════════════

col_groups = classify_columns(
    df,
    target_col=TARGET_COL,
    known_id_cols=KNOWN_ID_COLS,
    known_meta_cols=KNOWN_META_COLS,
    leakage_suspect_cols=LEAKAGE_SUSPECT_COLS,
)

# Derive is_home flag (team vs home_team)
df["is_home"] = (df["team_id"] == df["home_team_id"]).astype(int)

# Build enriched SPADL-style DataFrame for action-level plots
df_styles = build_styles_dataframe(df)
plot_action_distributions(
    df_styles=df_styles,
    title="Action features (all data)",
    output_path=OUTPUT_DIR / "action_features_distributions.png",
)
plot_displacement_histograms(
    df_styles=df_styles,
    title="Displacement distributions (all data)",
    output_path=OUTPUT_DIR / "displacement_histograms.png",
    seed=SEED,
)
log.info("Saved action features and displacement plots.")


# ════════════════════════════════════════════════
#  3. DATA QUALITY
# ════════════════════════════════════════════════

quality_df = compute_data_quality(df)
quality_df.to_csv(OUTPUT_DIR / "data_quality.csv", index=False)

print("\nData quality (top rows by missingness):")
print(quality_df.head(15).to_string(index=False))

const_cols = quality_df[quality_df["is_constant"]]["column"].tolist()
if const_cols:
    print(f"\n⚠ Constant columns (consider dropping): {const_cols}")


# ════════════════════════════════════════════════
#  4. MISSINGNESS
# ════════════════════════════════════════════════

miss_df = compute_missingness(df)
if miss_df.empty:
    log.info("No missing values found.")
else:
    miss_df.to_csv(OUTPUT_DIR / "missingness.csv", index=False)
    plot_missingness_bar(miss_df, OUTPUT_DIR / "missingness.png", top_n=40)
    log.info("Saved missingness plot.")
    print("\nMissing columns:")
    print(miss_df.to_string(index=False))


# ════════════════════════════════════════════════
#  5. TARGET ANALYSIS
# ════════════════════════════════════════════════

y = df[TARGET_COL]
target_vc = y.value_counts(dropna=False)
target_pos_rate = float(y.mean()) if pd.api.types.is_bool_dtype(y) else None

target_info = {
    "value_counts": target_vc.to_dict(),
    "positive_rate": target_pos_rate,
    "n_missing": int(y.isna().sum()),
}

plot_target_distribution(target_vc, TARGET_COL, OUTPUT_DIR / "target_distribution.png")

print(f"\nTarget '{TARGET_COL}':")
print(f"  Value counts: {target_vc.to_dict()}")
print(f"  Positive rate: {target_pos_rate}")
print(f"  Missing: {target_info['n_missing']}")


# ════════════════════════════════════════════════
#  6. LEAKAGE SUSPICION
# ════════════════════════════════════════════════

leakage = detect_leakage_suspects(
    df=df,
    target_col=TARGET_COL,
    leakage_suspect_cols=set(col_groups.get("leakage_suspect", [])),
    numeric_feature_cols=NUM_FEATS,
    corr_threshold=LEAKAGE_CORR_THRESH,
)

if leakage:
    pd.DataFrame(leakage).to_csv(OUTPUT_DIR / "leakage_suspects.csv", index=False)
    print("\n⚠ Leakage suspects:")
    for item in leakage:
        print(
            f"  {item['column']:25s}  {item['reason']:25s}  corr={item['corr_with_target']:.4f}"
        )
else:
    print("\nNo leakage suspects detected.")


# ════════════════════════════════════════════════
#  7. UNIVARIATE — NUMERIC
# ════════════════════════════════════════════════

if NUM_FEATS:
    num_desc = (
        df[NUM_FEATS].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    )
    num_desc["skew"] = df[NUM_FEATS].skew()
    num_desc["kurtosis"] = df[NUM_FEATS].kurtosis()
    num_desc["n_zeros"] = (df[NUM_FEATS] == 0).sum()
    num_desc["pct_zeros"] = ((df[NUM_FEATS] == 0).mean() * 100).round(2)
    num_desc.to_csv(OUTPUT_DIR / "univariate_numeric.csv")

    plot_univariate_numeric_grid(
        df,
        NUM_FEATS,
        OUTPUT_DIR / "univariate_numeric.png",
        seed=SEED,
    )
    log.info("Saved univariate numeric plot (%d features).", len(NUM_FEATS))
    print(
        f"\nNumeric descriptive stats ({len(NUM_FEATS)} cols) — saved to univariate_numeric.csv"
    )
    print(num_desc[["mean", "std", "min", "max", "skew"]].to_string())
else:
    num_desc = pd.DataFrame()
    print("\nNo numeric columns found.")


# ════════════════════════════════════════════════
#  8. UNIVARIATE — CATEGORICAL
# ════════════════════════════════════════════════

TOP_K_CAT = 15

if CAT_FEATS:
    _cat_records = []
    for col in CAT_FEATS:
        vc = df[col].value_counts(dropna=False)
        top_val = vc.index[0] if len(vc) > 0 else None
        top_pct = round(100 * vc.iloc[0] / len(df), 2) if len(vc) > 0 else 0
        _cat_records.append(
            {
                "column": col,
                "n_unique": int(df[col].nunique(dropna=True)),
                "top_value": top_val,
                "top_pct": top_pct,
                "is_dominant": top_pct > DOMINANT_CAT_PCT,
            }
        )
    cat_df = pd.DataFrame(_cat_records).sort_values("n_unique", ascending=False)
    cat_df.to_csv(OUTPUT_DIR / "univariate_categorical.csv", index=False)

    plot_univariate_categorical_grid(
        df,
        CAT_FEATS,
        OUTPUT_DIR / "univariate_categorical.png",
        top_k=TOP_K_CAT,
    )
    log.info("Saved univariate categorical plot.")
    print("\nCategorical summary:")
    print(cat_df.to_string(index=False))
else:
    cat_df = pd.DataFrame()
    print("\nNo categorical columns found.")


# ════════════════════════════════════════════════
#  9. BIVARIATE — FEATURE VS TARGET
# ════════════════════════════════════════════════

TOP_K_BIV = 20
y_float = df[TARGET_COL].astype(float)

_biv_records = []
for col in NUM_FEATS:
    x = df[col].astype(float)
    mask = x.notna() & y_float.notna()
    if mask.sum() < 10:
        continue
    corr, pval = stats.pointbiserialr(y_float[mask], x[mask])
    _biv_records.append({"column": col, "pb_corr": round(corr, 5), "pval": pval})

biv_df = pd.DataFrame(_biv_records).sort_values("pb_corr", key=abs, ascending=False)
biv_df.to_csv(OUTPUT_DIR / "bivariate_target.csv", index=False)

plot_bivariate_target_corr(
    biv_df, TARGET_COL, OUTPUT_DIR / "bivariate_target_corr.png", top_k=TOP_K_BIV
)
log.info("Saved bivariate target correlation plot.")

print(f"\nBivariate correlations with '{TARGET_COL}' (top {TOP_K_BIV}):")
print(biv_df.head(TOP_K_BIV).to_string(index=False))


# ════════════════════════════════════════════════
# 10. COLLINEARITY
# ════════════════════════════════════════════════

if len(NUM_FEATS) >= 2:
    corr_matrix = df[NUM_FEATS].corr()
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    plot_correlation_heatmap(corr_matrix, OUTPUT_DIR / "correlation_heatmap.png")
    log.info("Saved correlation heatmap.")

    # Extract highly correlated pairs
    _pairs = []
    _cols = corr_matrix.columns.tolist()
    for i in range(len(_cols)):
        for j in range(i + 1, len(_cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= CORR_THRESHOLD:
                _pairs.append(
                    {"col_a": _cols[i], "col_b": _cols[j], "pearson_r": round(r, 4)}
                )
    corr_pairs = pd.DataFrame(_pairs).sort_values("pearson_r", key=abs, ascending=False)
    if not corr_pairs.empty:
        corr_pairs.to_csv(OUTPUT_DIR / "high_corr_pairs.csv", index=False)
        print(f"\nHighly correlated pairs (|r| >= {CORR_THRESHOLD}):")
        print(corr_pairs.to_string(index=False))
    else:
        print(f"\nNo feature pairs above |r| >= {CORR_THRESHOLD}.")
else:
    corr_pairs = pd.DataFrame()
    print("\nNot enough numeric columns for collinearity analysis.")


# ════════════════════════════════════════════════
# 11. QUICK BASELINE MODELS
# ════════════════════════════════════════════════

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

baselines: dict = {}
baseline_metrics_df = pd.DataFrame()

if ALL_FEATS:
    X_all = df[ALL_FEATS]
    y_all = df[TARGET_COL].astype(int).values

    # Competition-based split
    idx_train = df.competition_name.isin(TRAINING_COMPETITIONS)
    idx_val = df.competition_name.isin(VALIDATION_COMPETITIONS)
    idx_test = df.competition_name.isin(TEST_COMPETITIONS)

    X_tr, y_tr = X_all[idx_train], y_all[idx_train]
    X_val, y_val = X_all[idx_val], y_all[idx_val]
    X_te, y_te = X_all[idx_test], y_all[idx_test]

    # Preprocessing: scale numeric + one-hot encode categorical
    scaler, encoder, all_feature_names = build_preprocessor(
        X_tr,
        num_feats=NUM_FEATS,
        cat_feats=CAT_FEATS,
        min_frequency=10,
    )
    X_tr_preproc = preprocess_split(X_tr, NUM_FEATS, CAT_FEATS, scaler, encoder)
    X_val_preproc = preprocess_split(X_val, NUM_FEATS, CAT_FEATS, scaler, encoder)
    X_te_preproc = preprocess_split(X_te, NUM_FEATS, CAT_FEATS, scaler, encoder)

    n_obs, n_cols = X_tr_preproc.shape
    log.info(
        "Preprocessed: %d obs, %d features (%d numeric + %d categorical encoded)",
        n_obs,
        n_cols,
        len(NUM_FEATS),
        n_cols - len(NUM_FEATS),
    )

    baseline_metrics_rows: list[dict[str, float | str]] = []

    def _register(
        model_name: str, split_name: str, y_true: np.ndarray, y_proba: np.ndarray
    ) -> dict:
        m = evaluate_binary(y_proba=y_proba, y_true=y_true, threshold=0.5)
        baseline_metrics_rows.append({"model": model_name, "split": split_name, **m})
        log.info(
            "  %s [%s] — AUC=%.4f, PR-AUC=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f",
            model_name,
            split_name,
            m["roc_auc"],
            m["pr_auc"],
            m["precision"],
            m["recall"],
            m["f1"],
        )
        return m

    # --- Logistic Regression ---
    log.info("Fitting LogisticRegression baseline ...")
    lr = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=2000,
        solver="saga",
        random_state=SEED,
        n_jobs=-1,
    )
    lr.fit(X_tr_preproc, y_tr)

    _register("LogReg", "train", y_tr, lr.predict_proba(X_tr_preproc)[:, 1])
    lr_metrics_val = _register(
        "LogReg", "validation", y_val, lr.predict_proba(X_val_preproc)[:, 1]
    )
    lr_metrics_test = _register(
        "LogReg", "test", y_te, lr.predict_proba(X_te_preproc)[:, 1]
    )

    lr_imp = pd.Series(np.abs(lr.coef_[0]), index=all_feature_names).sort_values(
        ascending=False
    )
    baselines["logreg_top_features"] = lr_imp.head(20).to_dict()

    # --- Random Forest (on raw features — handles mixed types) ---
    log.info("Fitting RandomForest baseline (small) ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    _register("RandomForest", "train", y_tr, rf.predict_proba(X_tr)[:, 1])
    rf_metrics_val = _register(
        "RandomForest", "validation", y_val, rf.predict_proba(X_val)[:, 1]
    )
    rf_metrics_test = _register(
        "RandomForest", "test", y_te, rf.predict_proba(X_te)[:, 1]
    )

    rf_imp = pd.Series(rf.feature_importances_, index=ALL_FEATS).sort_values(
        ascending=False
    )
    baselines["rf_top_features"] = rf_imp.head(20).to_dict()

    # --- MLP ---
    log.info("Fitting MLP baseline ...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=200,
        random_state=SEED,
    )
    mlp.fit(X_tr_preproc, y_tr)

    _register("MLP", "train", y_tr, mlp.predict_proba(X_tr_preproc)[:, 1])
    _register("MLP", "validation", y_val, mlp.predict_proba(X_val_preproc)[:, 1])
    _register("MLP", "test", y_te, mlp.predict_proba(X_te_preproc)[:, 1])

    # Collect and save
    baseline_metrics_df = pd.DataFrame(baseline_metrics_rows)
    baseline_metrics_df.to_csv(
        OUTPUT_DIR / "baseline_metrics_by_split.csv", index=False
    )
    log.info("Saved baseline split metrics.")

    # Importance comparison plot
    plot_feature_importance_comparison(
        lr_imp,
        rf_imp,
        lr_label=f"LogReg |coef| (F1={lr_metrics_test['f1']:.4f})",
        rf_label=f"RF importance (F1={rf_metrics_test['f1']:.4f})",
        output_path=OUTPUT_DIR / "baseline_feature_importance.png",
    )
    imp_df = pd.DataFrame({"logreg_abs_coef": lr_imp, "rf_importance": rf_imp})
    imp_df.to_csv(OUTPUT_DIR / "feature_importances_baseline.csv")

    for split_name in ("validation", "test"):
        subset = baseline_metrics_df[baseline_metrics_df["split"] == split_name]
        print(f"\n{split_name.capitalize()} metrics by model:")
        print(
            subset[
                ["model", "roc_auc", "pr_auc", "precision", "recall", "f1"]
            ].to_string(index=False)
        )

    print("\nTop 10 features (RF):")
    print(rf_imp.head(10).to_string())
    print("\nTop 10 features (LogReg |coef|):")
    print(lr_imp.head(10).to_string())
else:
    print("\nNo usable features for baselines.")


# ════════════════════════════════════════════════
# 12. MARKDOWN SUMMARY
# ════════════════════════════════════════════════

summary_md = generate_eda_summary_markdown(
    data_file=DATA_FILE,
    key=KEY,
    shape=df.shape,
    target_col=TARGET_COL,
    target_info=target_info,
    col_groups=col_groups,
    miss_df=miss_df,
    const_cols=const_cols,
    leakage=leakage,
    cat_df=cat_df,
    corr_pairs=corr_pairs,
    corr_threshold=CORR_THRESHOLD,
    baselines=baselines,
    baseline_metrics_df=baseline_metrics_df,
    output_dir=OUTPUT_DIR,
)
summary_path = OUTPUT_DIR / "eda_summary.md"
summary_path.write_text(summary_md, encoding="utf-8")
log.info("Wrote summary → %s", summary_path)


# ════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════
log.info("EDA complete. All artifacts saved to %s/", OUTPUT_DIR)


# ════════════════════════════════════════════════
# TODO — future extensions
# ════════════════════════════════════════════════
# - Stratified sampling: sample preserving target balance + competition mix
# - Temporal split awareness: check feature drift across seasons / game_date
# - Outlier handling: IQR / z-score flagging per numeric column
# - Deeper leakage checks: train a model with each suspect, compare AUC
