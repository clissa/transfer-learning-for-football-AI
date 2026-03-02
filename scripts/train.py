from pathlib import Path

from football_ai.training import (
    build_models,
    build_param_grids,
    evaluate_models_on_datasets,
    tune_models,
)
from football_ai.utils import list_available_dataset_keys, load_xy, make_dataset_key

# =========================
# Macro parameters
# =========================
DATA_DIR = "data/spadl_data"
TARGET_COL = "scores"  # or 'concedes'

TRAIN_LEAGUE = "La Liga"
TRAIN_SEASON = "2015/2016"

# Keep both as None to test on all available datasets except train set.
# Set both to explicit values to test on one specific league-season.
TEST_LEAGUE = "Champions League"
TEST_SEASON = "2015/2016"

RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True)
# =========================


def resolve_test_dataset_keys(
    available_dataset_keys: list[str],
    train_dataset_key: str,
    test_league: str | None,
    test_season: str | None,
) -> list[str]:
    """Resolve test dataset keys from TEST_LEAGUE/TEST_SEASON parameters."""
    if (test_league is None) != (test_season is None):
        raise ValueError("TEST_LEAGUE and TEST_SEASON must both be set or both be None")
    if test_league is None and test_season is None:
        return [key for key in available_dataset_keys if key != train_dataset_key]
    test_dataset_key = make_dataset_key(test_league, test_season)
    if test_dataset_key not in available_dataset_keys:
        raise ValueError(
            f"Test dataset key not found: {test_dataset_key}. "
            f"Available keys: {available_dataset_keys[:20]}"
        )
    return [test_dataset_key]


def main() -> int:
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir.resolve()}")

    available_dataset_keys = list_available_dataset_keys(data_dir)
    if not available_dataset_keys:
        raise ValueError("No features/labels dataset pairs found in DATA_DIR")

    train_dataset_key = make_dataset_key(TRAIN_LEAGUE, TRAIN_SEASON)
    if train_dataset_key not in available_dataset_keys:
        raise ValueError(
            f"Train dataset key not found: {train_dataset_key}. "
            f"Available keys: {available_dataset_keys[:20]}"
        )

    X_train, y_train = load_xy(
        dataset_key=train_dataset_key,
        target_col=TARGET_COL,
        data_dir=data_dir,
    )
    train_feature_cols = list(X_train.columns)

    test_dataset_keys = resolve_test_dataset_keys(
        available_dataset_keys=available_dataset_keys,
        train_dataset_key=train_dataset_key,
        test_league=TEST_LEAGUE,
        test_season=TEST_SEASON,
    )

    X_test, y_test = load_xy(
        dataset_key=test_dataset_keys[0],
        target_col=TARGET_COL,
        data_dir=data_dir,
    )

    models = build_models(random_state=42)
    param_grids = build_param_grids()

    trained_models, tuning_table = tune_models(
        X_train=X_train[:1000],
        y_train=y_train[:1000],
        models=models,
        param_grids=param_grids,
        scoring="f1",
        cv=5,
        n_jobs=8,
        verbose=1,
    )

    print("\n=== Tuning summary ===")
    print(tuning_table)
    print("Trained tuned models:", list(trained_models.keys()))
    print("Train rows:", len(X_train), "Target:", TARGET_COL)

    results_tables_by_model, comparison_table = evaluate_models_on_datasets(
        trained_models=trained_models,
        test_dataset_keys=test_dataset_keys,
        train_feature_cols=train_feature_cols,
        target_col=TARGET_COL,
        data_dir=str(data_dir),
    )

    for model_name, table in results_tables_by_model.items():
        print(f"\n=== {model_name} ===")
        print(table)

    print("\n=== Combined table ===")
    print(comparison_table)

    for model_name, table in results_tables_by_model.items():
        model_slug = model_name.lower().replace(" ", "_")
        out_path = RESULTS_PATH / f"eval_{TARGET_COL}_train_{train_dataset_key}_{model_slug}.csv"
        table.to_csv(out_path, index=False)

    out_all = RESULTS_PATH / f"eval_{TARGET_COL}_train_{train_dataset_key}_all_models.csv"
    comparison_table.to_csv(out_all, index=False)

    print("\nSaved CSV outputs in", RESULTS_PATH.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
