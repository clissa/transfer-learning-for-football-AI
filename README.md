# supervised-learning-statsbomb

Utilities and scripts for creating SPADL/VAEP datasets from StatsBomb open data.

## Project structure

```text
.
├── data/                          # Local output/data workspace
├── notebooks/                     # Exploratory and pipeline notebooks
├── scripts/
│   └── create_datasets.py         # Main data preparation script
├── src/
│   └── football_ai/
│       ├── __init__.py
│       └── utils.py               # Dataset build/save utilities
├── pyproject.toml
└── requirements.txt
```

## Installation

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

Option A (exact pinned environment):

```bash
pip install -r requirements.txt
```

Option B (install package in editable mode):

```bash
pip install -e .
```

If you use Option B only, also install runtime dependencies (for example `socceraction`, `pandas`, `tables`, `scikit-learn`) if they are not already available in your environment.

## Running data preparation

The main entrypoint is:

```bash
python scripts/create_datasets.py
```

### Prerequisite data location

By default, `scripts/create_datasets.py` expects StatsBomb open-data JSON files at:

```text
../open-data/data
```

This path is controlled by `DATA_ROOT` in [`scripts/create_datasets.py`](scripts/create_datasets.py).  
Update it if your open-data checkout is elsewhere.

### Competition/season selection

Edit these constants in `scripts/create_datasets.py`:

- `SAVE_ALL_AVAILABLE`: `True` to process all available competitions/seasons.
- `SELECTED_NAME_PAIRS`: list of `(competition_name, season_name)` pairs when `SAVE_ALL_AVAILABLE=False`.
- `SELECTED_ID_PAIRS`: list of `(competition_id, season_id)` pairs (alternative to names).
- `NUMBER_PREVIOUS_ACTIONS`: context window for VAEP feature generation.

Use either `SELECTED_NAME_PAIRS` or `SELECTED_ID_PAIRS`, not both.

### Output files

The script writes HDF5 feature/label files to `OUTPUT_DIR` (default: `test-data/spadl_data`) with names like:

- `features_<competition>_<season>.h5`
- `labels_<competition>_<season>.h5`

Each file contains one table key:

- features file: `features`
- labels file: `labels`
