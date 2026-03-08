# Transfer Learning to Overcome Domain Shift in Football Analytics and Beyond

Python ML research project for football action valuation (VAEP) using StatsBomb open-data.

Pipeline: StatsBomb JSON → SPADL actions → VAEP features/labels → model training (sklearn / XGBoost) → evaluation.

Data pipeline steps:
1. `scripts/create_spadl_dataset.py` — StatsBomb → SPADL + labels in HDF5
2. `scripts/create_vaep_features.py` — SPADL full_data → VAEP features in HDF5
3. `scripts/train.py` / `scripts/train_xgboost.py` — Model training

## Project structure

```text
├── configs/                       # YAML experiment configs
│   ├── create_spadl_dataset.yaml
│   ├── create_vaep_features.yaml
│   ├── train_sklearn.yaml
│   ├── train_xgboost.yaml
│   └── tune_xgboost.yaml
├── data/                          # Local data (gitignored) — see data/README.md
├── logs/                          # Captured stdout/stderr logs
├── notebooks/                     # Exploratory notebooks
├── scripts/
│   ├── create_spadl_dataset.py    # SPADL data preparation CLI
│   ├── create_vaep_features.py    # VAEP feature extraction CLI
│   ├── train.py                   # sklearn training CLI
│   ├── train_xgboost.py           # XGBoost training CLI
│   └── tune_xgboost_bayes_v2.py   # Bayesian tuning CLI (Optuna)
├── src/football_ai/
│   ├── __init__.py
│   ├── config.py                  # YAML loading, CLI override merging
│   ├── data.py                    # Data loading, SPADL/VAEP conversion, HDF5 I/O
│   ├── features.py                # VAEP feature extraction from SPADL actions
│   ├── training.py                # Model building, training helpers, grid search
│   └── evaluation.py              # Metrics, threshold sweep, visualization
├── tests/
│   ├── test_smoke_library.py
│   └── test_smoke_train.py
├── archive/                       # Superseded scripts (kept for reference)
├── pyproject.toml
└── requirements.txt
```

## Installation

### 1) Create and activate a virtual environment

```bash
python3 -m venv ./football_ai_venv
source ./football_ai_venv/bin/activate
```

### 2) Install dependencies

```bash
# Runtime dependencies
pip install -r requirements.txt

# Install the football_ai package (editable)
pip install -e .

# Optional extras (see pyproject.toml for all options)
pip install -e ".[xgboost,tuning,viz,dev]"
```

## Data preparation

Clone the [StatsBomb open-data](https://github.com/statsbomb/open-data) repository and run:

```bash
# Step 1: Create SPADL dataset
python -m scripts.create_spadl_dataset --config configs/create_spadl_dataset.yaml

# Step 2: Compute VAEP features
python -m scripts.create_vaep_features --config configs/create_vaep_features.yaml
```

The `data_root` path in the config defaults to `../open-data/data`. Update it if your checkout is elsewhere. See [data/README.md](data/README.md) for the expected output layout and HDF5 key structure.

## Training

### sklearn models (Logistic Regression, Random Forest, MLP)

```bash
python -m scripts.train --config configs/train_sklearn.yaml
```

Override options via CLI, e.g. `--model rf`, `--target-col concedes`.

### XGBoost

```bash
python -m scripts.train_xgboost --config configs/train_xgboost.yaml
```

Override options via CLI, e.g. `--target-col concedes`.

### Bayesian hyperparameter tuning (Optuna + XGBoost)

```bash
python -m scripts.tune_xgboost_bayes_v2 --config configs/tune_xgboost.yaml
```

Override options via CLI, e.g. `--n-trials 50`.

### Capturing logs

Pipe stdout/stderr to `logs/` for reproducibility:

```bash
python -m scripts.train --config configs/train_sklearn.yaml 2>&1 | tee logs/train_sklearn.log
```

## Tests

```bash
pytest tests/ -v
```

## Notebooks

The `notebooks/` folder contains exploratory Jupyter notebooks:

- `process_statsbomb_data.ipynb` — Minimal tutorial: load one competition, convert to SPADL, generate VAEP features/labels.
- `create_spadl_dataset.ipynb` — End-to-end pipeline covering all competitions: SPADL conversion, VAEP generation, merged dataset.
- `create_spadl_dataset_major_leagues.ipynb` — Data pipeline + quick RF evaluation for major men's leagues.
- `create_spadl_dataset_women_league_season.ipynb` — Same pipeline for women's leagues.
- `socceraction_supervised_learning.ipynb` — Multi-model comparison: train on one league-season, test on all others.
- `socceraction_supervised_learning_simple.ipynb` — Simplified version with GridSearchCV tuning.

> **Note:** Notebooks are exploratory and may not reflect the latest library API. For production workflows, use the scripts above.

