import importlib.util
from pathlib import Path

import pandas as pd
from football_ai import utils as football_utils

# =========================
# Macro parameters
# =========================
# StatsBomb open-data JSON folder (specify relative to current working directory, i.e. where you run this script from)
DATA_ROOT = "../open-data/data"  # currently expected to be the repo root
OUTPUT_DIR = "test-data/spadl_data"

SAVE_ALL_AVAILABLE = False

# Used only when SAVE_ALL_AVAILABLE=False
SELECTED_NAME_PAIRS: list[tuple[str, str]] = [
    # ("Serie A", "2015/2016"),
    ("Champions League", "2015/2016"),
]

# Used only when SAVE_ALL_AVAILABLE=False
SELECTED_ID_PAIRS: list[tuple[int, int]] = [
    # (11, 27),
]

# Number of previous actions to include in the dataset (for context)
NUMBER_PREVIOUS_ACTIONS = 3
# =========================


# def _load_utils_module():
#     """Load utilities from src/football-ai without requiring package installation."""
#     repo_root = Path(__file__).resolve().parent.parent
#     utils_path = repo_root / "src" / "football-ai" / "utils.py"
#     if not utils_path.exists():
#         raise FileNotFoundError(f"Missing expected module file: {utils_path}")

#     spec = importlib.util.spec_from_file_location("football_ai_utils", utils_path)
#     if spec is None or spec.loader is None:
#         raise ImportError(f"Cannot create module spec for: {utils_path}")

#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module


def main() -> int:
    # football_utils = _load_utils_module()

    data_root = Path(DATA_ROOT)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = football_utils.make_statsbomb_loader(data_root)
    competitions_df = football_utils.list_competitions(loader)

    cols = ["competition_id", "season_id", "competition_name", "season_name"]
    print("output_dir:", output_dir)
    print("\nAvailable competitions/seasons:")
    print(
        competitions_df[cols]
        .sort_values(["competition_name", "season_name"])
        .reset_index(drop=True)
    )

    selected = football_utils.select_competition_seasons(
        competitions_df=competitions_df,
        save_all_available=SAVE_ALL_AVAILABLE,
        selected_name_pairs=SELECTED_NAME_PAIRS,
        selected_id_pairs=SELECTED_ID_PAIRS,
    )

    print("\nTotal selected (competition_id, season_id):", len(selected))
    print("Preview:", selected[:10])

    if not selected:
        raise ValueError(
            "No league/season selected. Configure SAVE_ALL_AVAILABLE or selection lists."
        )

    outputs: list[tuple[str, str, dict]] = []
    failed: list[tuple[int, int, str]] = []

    for competition_id, season_id in selected:
        try:
            outputs.append(
                football_utils.build_and_save_vaep_for_competition_season(
                    loader=loader,
                    competitions_df=competitions_df,
                    output_dir=output_dir,
                    competition_id=competition_id,
                    season_id=season_id,
                    nb_prev_actions=NUMBER_PREVIOUS_ACTIONS,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((competition_id, season_id, str(exc)))
            print(f"Skipped (cid={competition_id}, sid={season_id}): {exc}")

    print("\nDone.")
    print("Saved datasets:", len(outputs))
    print("Failed datasets:", len(failed))

    if outputs:
        print("\nSaved files:")
        for features_path, labels_path, _meta in outputs:
            print("-", features_path)
            print("-", labels_path)

    if failed:
        print("\nFirst failed items:")
        print(
            pd.DataFrame(
                failed,
                columns=["competition_id", "season_id", "error"],
            ).head(10)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
