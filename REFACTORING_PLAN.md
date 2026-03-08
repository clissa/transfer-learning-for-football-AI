# Piano di Armonizzazione Strutturale — `supervised-learning-statsbomb`

> **Scopo**: pulizia post-refactoring. Nessuna nuova feature, nessun over-engineering.
> Priorità: chiarezza, semplicità, coerenza, riusabilità.

---

## Contesto Generale

### Cosa fa questa repo

Repo Python ML di ricerca per valutazione azioni calcistiche (VAEP) da dati StatsBomb open-data.
Pipeline: StatsBomb JSON → SPADL actions → VAEP features/labels → training modelli (sklearn + XGBoost) → evaluation.

### Stack

- **scikit-learn + XGBoost** (modelli)
- **socceraction** (SPADL/VAEP conversion)
- **pandas + HDF5** (storage)
- **optuna** (tuning bayesiano, opzionale)
- **argparse** (CLI) — migrare verso YAML + argparse

### Struttura attuale (`src/football_ai/`)

```
src/football_ai/
├── __init__.py      # Re-esporta tutto da training.py e utils.py
├── training.py      # 665 righe: data loading, model building, evaluation, viz, grid search
└── utils.py         # 403 righe: StatsBomb loading, SPADL conversion, VAEP features/labels, HDF5 I/O
```

### Struttura target (`src/football_ai/`)

```
src/football_ai/
├── __init__.py      # Re-esporta tutto da data, training, evaluation
├── data.py          # Unifica utils.py + logica estratta da create_datasets.py
├── training.py      # Solo: data loading helpers, model building, XGBoost helpers, grid search
└── evaluation.py    # Estratto da training.py: metriche, threshold sweep, plot
```

### Formato dati H5

**Formato target** (un file per combinazione lega/stagione, es. `major_leagues.h5`):
- Chiavi HDF5: `actions`, `games`, `teams`, `players`, `competitions`, `labels`, `full_data`
- La chiave `full_data` è il dataset di training (merge completo di tutte le tabelle)

**Formato legacy** (ancora usato da `train.py`):
- Due file separati per lega/stagione: `features_<key>.h5` (chiave `features`) + `labels_<key>.h5` (chiave `labels`)

### Scripts attuali

| Script | Righe | Stato | Problema |
|--------|-------|-------|----------|
| `scripts/create_datasets.py` | 582 | Attivo | ~500 righe di core logic inline, NON importa da `football_ai` |
| `scripts/train.py` | 157 | Attivo, thin wrapper | Importa da `football_ai.utils` (→ cambiare in `football_ai.data`) |
| `scripts/train_xgboost.py` | 304 | Attivo, thin wrapper | OK, importa da `football_ai.training` |
| `scripts/tune_xgboost_bayes_v2.py` | 605 | Attivo | ~300 righe di helper duplicati, NON importa da `football_ai` |
| `scripts/create_datasets_old.py` | ~100 | **Legacy** | Da archiviare |
| `scripts/create_spadl_rich_leagues_old.py` | ~340 | **Legacy** | Da archiviare |
| `scripts/tune_xgboost_bayes.py` | ~320 | **Legacy** | Superseded da v2, da archiviare |

### `__init__.py` attuale (re-exports)

```python
from .training import (
    build_models, build_param_grids, build_sklearn_model, build_xgb_eval_set,
    drop_none_params, evaluate_binary, evaluate_binary_with_baselines,
    evaluate_models_on_datasets, get_positive_class_scores,
    load_xy_all, load_xy_competition_split, load_xy_game_split,
    plot_confusion_matrix, print_metrics, read_h5_table,
    sweep_thresholds_for_f1, tune_models,
)
from .utils import (
    build_and_save_vaep_for_competition_season, build_vaep_dataset_for_competition_season,
    list_available_dataset_keys, list_competitions, load_dataset_tables,
    load_xy, make_dataset_key, make_statsbomb_loader,
    output_paths_for_competition_season, resolve_competition_season_ids,
    save_vaep_dataset, select_competition_seasons, slug, split_dataset_key,
)
```

---

## Piano Completo — Checklist per priorità

| # | Priorità | Steps | Descrizione |
|---|----------|-------|-------------|
| ~~P1~~ | ~~Core: `data.py` unificato~~ | ~~1–3~~ | ~~Assorbire `utils.py` + logica da `create_datasets.py` in `data.py`~~ | ✅
| ~~P2~~ | ~~Core: `evaluation.py`~~ | ~~4~~ | ~~Estrarre evaluation da `training.py`~~ | ✅
| ~~P3~~ | ~~Scripts: consolidare duplicati~~ | ~~5–7~~ | ~~Aggiornare script per importare dalla libreria~~ | ✅
| ~~P4~~ | ~~Pulizia: legacy & artefatti~~ | ~~8–9~~ | ~~Archiviare script `_old`, pulire artefatti~~ | ✅
| ~~P5~~ | ~~Config: YAML + argparse~~ | ~~10~~ | ~~Creare `configs/` con template YAML~~ | ✅
| ~~P6~~ | ~~Quality: test & packaging~~ | ~~11–12~~ | ~~Test smoke per libreria, dipendenze in pyproject.toml~~ | ✅
| ~~P7~~ | ~~Docs: README & notebook~~ | ~~13–14~~ | ~~Aggiornare README, data README, semplificare notebook~~ | ✅

---

## P1 — Creare `data.py` unificato (assorbire `utils.py`)

### Obiettivo
Un unico modulo `src/football_ai/data.py` che contiene TUTTA la logica dati: loading StatsBomb, conversione SPADL, VAEP features/labels, merge, HDF5 I/O, lettura generica H5.

### Cosa fare

**Step 1: Creare `src/football_ai/data.py`**

Contenuto = tutto `utils.py` + logica riutilizzabile estratta da `create_datasets.py` + `read_h5_table` da `training.py`.

Funzioni da `utils.py` (tutte, integrali):
- `slug`, `make_dataset_key`, `split_dataset_key`
- `list_available_dataset_keys`, `load_dataset_tables`, `load_xy`
- `make_statsbomb_loader`, `list_competitions`
- `resolve_competition_season_ids`, `select_competition_seasons`
- `_as_series`, `_competition_row`
- `build_vaep_dataset_for_competition_season`
- `output_paths_for_competition_season`, `save_vaep_dataset`
- `build_and_save_vaep_for_competition_season`

Funzioni da `create_datasets.py` (logica riutilizzabile):
- `_ensure_cols_from_index` — helper generico DataFrame
- `_drop_overlaps` — helper merge
- `select_competitions` — filtro competizioni per nome
- `load_games` — carica partite da StatsBombLoader per competizioni selezionate
- `convert_games_to_actions` — converte eventi → SPADL actions
- `load_teams_players` — carica teams + players
- `build_labels` — calcola label VAEP (scores, concedes) per azioni
- `build_merged_output` — merge tutte le tabelle in un DataFrame wide
- `_stringify_for_hdf` — cast colonne problematiche per serializzazione HDF5
- `build_and_save_dataset` — pipeline end-to-end: build + save H5 multi-key

Funzione da `training.py`:
- `read_h5_table` — lettura generica H5 con key candidates (usata da `load_xy_competition_split` e da `tune_xgboost_bayes_v2.py`)

Nota: `_as_series` esiste sia in `utils.py` che in `create_datasets.py` (identica). Tenerne UNA sola copia in `data.py`.

**Costante da portare in `data.py`**: `REQUESTED_COLUMNS` (lista colonne per il merge output). Attualmente definita come costante in `create_datasets.py` — spostarla in `data.py` come default, ma renderla parametro opzionale di `build_merged_output`.

**Step 2: Eliminare `utils.py`**

Dopo la migrazione, cancellare `src/football_ai/utils.py`.

**Step 3: Aggiornare `training.py`**

- Cambiare `from .utils import load_dataset_tables, load_xy, split_dataset_key` → `from .data import ...`
- Rimuovere `read_h5_table` (ora vive in `data.py`) e importarlo: `from .data import read_h5_table`

**Step 4: Aggiornare `__init__.py`**

- Cambiare `from .utils import ...` → `from .data import ...`
- Aggiungere i re-export delle nuove funzioni pubbliche (es. `select_competitions`, `load_games`, `convert_games_to_actions`, `build_labels`, `build_merged_output`, `build_and_save_dataset`, `read_h5_table`)
- Mantenere TUTTI i nomi pubblici precedenti (backward compat)

### Dettagli implementativi — Funzioni da `create_datasets.py`

Ecco le signature e la logica di ogni funzione da estrarre. Il codice completo è in `scripts/create_datasets.py`.

```python
# --- Helpers ---

def _ensure_cols_from_index(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Reset index if required columns are missing from columns (but present in index)."""

def _drop_overlaps(right: pd.DataFrame, left_cols: pd.Index, join_keys: set[str]) -> pd.DataFrame:
    """Drop columns from *right* that already exist in *left_cols* (except join keys)."""

# --- Data loading ---

def select_competitions(competitions_df: pd.DataFrame, names: list[str] | None) -> pd.DataFrame:
    """Select competitions by name. If *names* is None, return all."""

def load_games(loader: StatsBombLoader, selected_competitions: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, int, str]]]:
    """Load games for all selected competition/season pairs."""

def convert_games_to_actions(loader: StatsBombLoader, df_games: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Convert raw StatsBomb events to SPADL actions for every game."""

def load_teams_players(loader: StatsBombLoader, game_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, str]]]:
    """Load teams and players tables for a list of game ids."""

# --- Labels & merge ---

def build_labels(df_actions: pd.DataFrame, df_games: pd.DataFrame) -> pd.DataFrame:
    """Compute VAEP labels (scores, concedes) for every action."""

def build_merged_output(df_actions, df_games, df_teams, df_players, df_competitions, df_labels, requested_columns=None) -> pd.DataFrame:
    """Merge all tables into one wide actions DataFrame with full metadata."""
    # Se requested_columns è None, usa REQUESTED_COLUMNS default

def _stringify_for_hdf(df_games: pd.DataFrame, df_players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast problematic object columns to str so HDF5 serialisation works."""

# --- End-to-end pipeline ---

def build_and_save_dataset(loader, competitions_df, league_names, output_file, nb_prev_actions=3) -> None:
    """Build the full dataset for the given leagues and write it to *output_file*."""
    # Chiavi H5 salvate: actions, games, teams, players, competitions, labels, full_data

# --- Da training.py ---

def read_h5_table(data_file: str | Path, key_candidates: Sequence[str]) -> pd.DataFrame:
    """Read a table from an HDF5 file, trying *key_candidates* in order."""
```

### Verifica dopo P1

```bash
python -c "from football_ai.data import slug, build_and_save_dataset, load_xy, read_h5_table"
python -c "from football_ai import slug, load_xy, read_h5_table"  # backward compat
grep -rn "from football_ai.utils import\|from .utils import" src/ scripts/  # deve dare 0 risultati
grep -rn "def _as_series" src/ scripts/  # una sola definizione, in data.py
```

---

## P2 — Creare `evaluation.py` (estrarre da `training.py`)

### Obiettivo
Estrarre la logica di valutazione da `training.py` (665 righe) in un nuovo modulo `evaluation.py`. `training.py` scende a ~400 righe.

### Cosa fare

**Step 1: Creare `src/football_ai/evaluation.py`**

Funzioni da spostare da `training.py`:
- `print_metrics(title, metrics)` — display helper
- `get_positive_class_scores(model, X)` — estrae P(y=1)
- `evaluate_binary(y_proba, y_true, threshold)` — metriche binarie
- `evaluate_binary_with_baselines(y_proba, y_true, threshold)` — eval + baselines naive
- `sweep_thresholds_for_f1(y_true, y_score, ...)` — threshold sweep
- `plot_confusion_matrix(cm, split_name, save_path)` — visualizzazione

Import necessari in `evaluation.py`:
```python
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
```

**Step 2: Aggiornare `training.py`**

- Rimuovere le funzioni spostate
- Aggiungere in cima: `from .evaluation import evaluate_binary, get_positive_class_scores, print_metrics`
  (per le funzioni che `training.py` usa internamente: `evaluate_models_on_datasets` usa `split_dataset_key` e `load_xy` ma NON usa evaluation functions direttamente — verificare)
- Rimuovere import sklearn.metrics non più necessari

**Step 3: Aggiornare `__init__.py`**

- Aggiungere `from .evaluation import ...` per le funzioni spostate
- Rimuovere i corrispondenti `from .training import ...`
- `__all__` invariato (stessi nomi)

### Dettagli — Contenuto `evaluation.py`

```python
"""Evaluation utilities for binary classification models.

Provides scoring, metrics computation, threshold sweeps, and visualization.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)

def drop_none_params(d: dict[str, Any]) -> dict[str, Any]: ...
def print_metrics(title: str, metrics: dict[str, float]) -> None: ...
def get_positive_class_scores(model: Any, X: pd.DataFrame) -> np.ndarray: ...
def evaluate_binary(y_proba, y_true, threshold=0.5) -> dict[str, float]: ...
def evaluate_binary_with_baselines(y_proba, y_true, threshold=0.5) -> dict[str, float]: ...
def sweep_thresholds_for_f1(y_true, y_score, ...) -> tuple[pd.DataFrame, float]: ...
def plot_confusion_matrix(cm, split_name, save_path) -> None: ...
```

Nota: `drop_none_params` è una utility generica (non strettamente evaluation), ma è usata da script di training/tuning. Può restare in `training.py` oppure in `evaluation.py`. Decisione: **lasciarla in `training.py`** (è un helper per param dicts, non per metriche).

### Funzioni che restano in `training.py` dopo P2

```
# Data loading
read_h5_table → spostata in data.py (P1)
load_xy_game_split
load_xy_all
load_xy_competition_split

# Model building
build_sklearn_model
build_models
build_param_grids

# Small utilities
drop_none_params

# XGBoost helpers
build_xgb_eval_set

# Grid search
tune_models
evaluate_models_on_datasets
```

### Nota su dipendenze interne

`evaluate_models_on_datasets` in `training.py` usa: `split_dataset_key`, `load_xy` (da `data.py`) e `accuracy_score`, `precision_score`, `recall_score`, `f1_score` (da sklearn). NON usa `evaluate_binary` — calcola le metriche direttamente. Questo è OK, può restare in `training.py`.

### Verifica dopo P2

```bash
python -c "from football_ai.evaluation import evaluate_binary, sweep_thresholds_for_f1, get_positive_class_scores"
python -c "from football_ai import evaluate_binary"  # backward compat
python -c "from football_ai.training import build_sklearn_model, load_xy_competition_split"  # training ancora funziona
```

---

## P3 — Consolidare script (rimuovere duplicati) ✅ COMPLETATO

> **Completato**: 2026-03-06. Scripts aggiornati per importare da `football_ai.evaluation` e `football_ai.data`.
> Re-export backward-compat rimosso da `training.py`. `tune_xgboost_bayes_v2.py` ridotto
> da 605 a 566 righe (~39 righe di helper duplicati rimossi).

### Obiettivo
Aggiornare gli script attivi per importare dalla libreria invece di avere logica inline duplicata. Gli script devono essere thin wrappers: config + CLI + chiamate a `football_ai.*`.

### Cosa fare

**Step 5: Slim-ificare `scripts/create_datasets.py`**

Dopo P1, tutta la logica core è in `football_ai.data`. Lo script deve contenere solo:
- Costanti di configurazione (`DATA_ROOT`, `OUTPUT_DIR`, `MAJOR_LEAGUES`, `NB_PREV_ACTIONS`, `OUTNAME`)
- `parse_args()` — argparse (già presente, non cambia)
- `main()` — chiama `football_ai.data.build_and_save_dataset(...)` (già presente, non cambia molto)

Rimuovere dallo script: `_as_series`, `_ensure_cols_from_index`, `_drop_overlaps`, `select_competitions`, `load_games`, `convert_games_to_actions`, `load_teams_players`, `build_labels`, `build_merged_output`, `_stringify_for_hdf`, `build_and_save_dataset`, `REQUESTED_COLUMNS`.

Aggiungere import:
```python
from football_ai.data import build_and_save_dataset, select_competitions
from socceraction.data.statsbomb import StatsBombLoader
```

Target: ~80–100 righe (da 582).

**Step 6: Aggiornare `scripts/tune_xgboost_bayes_v2.py`**

Sostituire helper duplicati con import dalla libreria:

| Funzione nello script | Sostituzione |
|----------------------|--------------|
| `_drop_none(d)` | `from football_ai.training import drop_none_params` |
| `get_scores(model, X)` | `from football_ai.evaluation import get_positive_class_scores` |
| `_read_rich_actions(data_file, key_candidates)` | `from football_ai.data import read_h5_table` |
| `_eval_ranking(y_true, y_score)` | `from football_ai.evaluation import evaluate_binary` (restituisce stesse metriche; adattare la chiamata: `evaluate_binary(y_score, y_true, threshold=0.5)` e prendere le chiavi `pr_auc`, `brier`, `roc_auc` + aggiungere `logloss` se non presente) |
| `_best_f1_threshold(y_true, y_score)` | `from football_ai.evaluation import sweep_thresholds_for_f1` (adattare output: ritorna `(sweep_df, best_threshold)`, estrarre f1/precision/recall dalla riga best) |
| `_eval_at_threshold(y_true, y_score, thr)` | `from football_ai.evaluation import evaluate_binary` con `threshold=thr` |

**Funzioni che RESTANO nello script** (troppo specifiche):
- `DataBundle` dataclass
- `_normalize_name` — normalizzazione nomi per matching
- `_detect_competition_and_season_cols` — heuristic colonne
- `_prepare_features` — casting specifico XGBoost
- `_first_existing` — piccolo helper
- `sample_xgb_params` — search space Optuna
- `load_data_bundle` — orchestrazione (ma userà `read_h5_table` da libreria)
- `_save_exploration` — salvataggio CSV
- `main` + `objective`

**Nota importante su `evaluate_binary` vs `_eval_ranking`**: `evaluate_binary` restituisce anche `precision`, `recall`, `f1` (a una threshold data), oltre a `pr_auc`, `brier`, `roc_auc`. Manca `logloss`. Due opzioni:
1. Aggiungere `logloss` a `evaluate_binary` in `evaluation.py` (consigliato — è una metrica standard)
2. Calcolare `logloss` separatamente nello script

Decisione: **aggiungere `logloss` a `evaluate_binary`** durante P2.

Target: ridurre lo script di ~80–100 righe (da 605 a ~500–520).

**Step 7: Aggiornare `scripts/train.py`**

Cambiare:
```python
from football_ai.utils import make_dataset_key
```
→
```python
from football_ai.data import make_dataset_key
```
(oppure `from football_ai import make_dataset_key` — funziona grazie al re-export in `__init__.py`)

Aggiornare anche `from football_ai.training import ...` se qualche funzione è migrata (es. `evaluate_binary`, `get_positive_class_scores`, `print_metrics` → ora in `evaluation`).

**Step 7b: Verificare `scripts/train_xgboost.py`**

- Già thin wrapper, importa da `football_ai.training`
- Aggiornare: `evaluate_binary_with_baselines`, `get_positive_class_scores`, `print_metrics`, `sweep_thresholds_for_f1`, `plot_confusion_matrix` → ora da `football_ai.evaluation`
- `drop_none_params`, `build_xgb_eval_set`, `load_xy_competition_split` → restano in `football_ai.training`

### Verifica dopo P3

```bash
python scripts/create_datasets.py --help   # deve funzionare
python scripts/train.py                     # (fallirà se mancano dati, ma import OK)
python scripts/train_xgboost.py             # (idem)
python scripts/tune_xgboost_bayes_v2.py     # (idem)
grep -rn "def _as_series\|def get_scores\|def _drop_none\|def _eval_ranking" scripts/  # deve dare 0 risultati
```

---

## P4 — Pulizia file legacy & artefatti ✅ COMPLETATO

> **Completato**: 2026-03-06. Script legacy spostati in `archive/scripts/`, rimosso
> `src/football_ai_statsbomb.egg-info/`, `data/spadl_data_test/`, `logs/repo_structure.txt`.
> Cache Python pulite, package reinstallato in editable mode.

### Obiettivo
Spostare script superseded in `archive/`, eliminare artefatti stale dal disco.

### Cosa fare

**Step 8: Archiviare script legacy**

```bash
mkdir -p archive/scripts
git mv scripts/create_datasets_old.py archive/scripts/
git mv scripts/create_spadl_rich_leagues_old.py archive/scripts/
git mv scripts/tune_xgboost_bayes.py archive/scripts/
```

Creare `archive/README.md`:
```markdown
# Archive

Script superseded, conservati per riferimento storico.
Recuperabili dalla git history se necessario.

- `scripts/create_datasets_old.py` — vecchia pipeline per-lega (sostituita da `scripts/create_datasets.py`)
- `scripts/create_spadl_rich_leagues_old.py` — vecchia pipeline merged (sostituita da `scripts/create_datasets.py`)
- `scripts/tune_xgboost_bayes.py` — tuning v1 per-lega (sostituita da `scripts/tune_xgboost_bayes_v2.py`)
```

**Step 9: Pulire artefatti su disco**

```bash
# Stale egg-info da vecchio nome package
rm -rf src/football_ai_statsbomb.egg-info/

# Directory dati vuota
rmdir data/spadl_data_test/

# Repo structure log obsoleto
rm logs/repo_structure.txt

# Cache Python stale
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null

# Rigenerare egg-info
pip install -e .
```

### Verifica dopo P4

```bash
ls archive/scripts/  # 3 file + README
ls scripts/           # solo create_datasets.py, train.py, train_xgboost.py, tune_xgboost_bayes_v2.py
test ! -d data/spadl_data_test  # non esiste più
test ! -d src/football_ai_statsbomb.egg-info  # non esiste più
```

---

## P5 — Creare configs YAML + argparse ✅ COMPLETATO

> **Completato**: 2026-03-06. Aggiunto `src/football_ai/config.py` con `load_config` e
> `merge_cli_overrides`. Creati 4 template YAML in `configs/`. Tutti gli script (`create_datasets.py`,
> `train.py`, `train_xgboost.py`, `tune_xgboost_bayes_v2.py`) ora supportano `--config`
> con override CLI. `pyyaml` aggiunto a `pyproject.toml`. Backward-compat: tutti gli script
> funzionano identicamente senza `--config`.

### Obiettivo
Aggiungere `configs/` con template YAML. Pattern: `--config configs/xxx.yaml` come argomento CLI, override opzionali via argparse.

### Cosa fare

**Step 10: Creare template YAML e aggiornare script**

**`configs/create_datasets.yaml`**
```yaml
data_root: "../open-data/data"
output_dir: "data/spadl_data"
outname: "major_leagues.h5"
nb_prev_actions: 3
leagues:
  - "La Liga"
  - "Serie A"
  - "Premier League"
  - "1. Bundesliga"
  - "Ligue 1"
  - "Champions League"
  - "UEFA Europa League"
# all_leagues: false  # set to true to override leagues list
```

**`configs/train_sklearn.yaml`**
```yaml
data:
  dir: "data/spadl_data"
  target_col: "scores"
train:
  league: "La Liga"
  season: "2015/2016"
test:
  league: "Champions League"
  season: "2015/2016"
model:
  name: "logreg"    # rf, logreg, mlp
  random_state: 20260304
  val_pct: 0.20
  config:
    logreg:
      C: 0.1
      class_weight: "balanced"
      max_iter: 5000
    rf:
      n_estimators: 500
      min_samples_leaf: 5
      class_weight: "balanced_subsample"
      criterion: "log_loss"
      n_jobs: -1
    mlp:
      hidden_layer_sizes: [256, 128, 64]
      alpha: 0.001
      solver: "adam"
      early_stopping: true
      max_iter: 200
output:
  dir: "results/debug_train"
```

**`configs/train_xgboost.yaml`**
```yaml
data:
  file: "data/spadl_data/major_leagues.h5"
  key_candidates: ["full_data"]
  target_col: "scores"
split:
  validation_competitions: ["Serie A", "Ligue 1"]
  test_competitions: ["Champions League", "UEFA Europa League"]
  train_competitions: null  # null = all remaining
model:
  objective: "binary:logistic"
  eval_metric: ["aucpr", "auc", "logloss"]
  n_estimators: 2000
  learning_rate: 0.05
  max_depth: 6
  # ... (altri parametri XGBoost)
  random_state: 20260305
threshold:
  enabled: true
  min: 0.05
  max: 0.95
  steps: 90
output:
  dir: "results/debug_train_xgboost_rich"
```

**`configs/tune_xgboost.yaml`**
```yaml
data:
  file: "data/spadl_data/major_leagues.h5"
  key_candidates: ["full_data"]
  target_col: "scores"
split:
  validation_competitions: ["Serie A", "Ligue 1"]
  test_competitions: ["Champions League", "UEFA Europa League"]
optuna:
  n_trials: 300
  timeout_seconds: null
  random_state: 20260305
training:
  n_estimators: 20000
  early_stopping_rounds: 200
  base_params:
    booster: "gbtree"
    tree_method: "hist"
    device: "cuda"
    objective: "binary:logistic"
    eval_metric: "aucpr"
output:
  dir: "results/xgboost_bayes_tuning"
```

**Pattern di caricamento YAML** (aggiungere a ogni script):
```python
import yaml

def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)
```

In ogni script aggiungere `--config` a argparse:
```python
parser.add_argument("--config", type=str, default=None, help="YAML config file")
```

Merge: YAML è la base, argparse overrides sovrascrivono.

### Nota
- `pyyaml` è già nel `requirements.txt` come dipendenza di altri pacchetti. Verificare e aggiungere esplicitamente se mancante.
- Per ora tenere i config semplici (flat o max 2 livelli). No Hydra, no Omegaconf.

### Verifica dopo P5

```bash
ls configs/  # 4 file yaml
python scripts/create_datasets.py --config configs/create_datasets.yaml --help
```

---

## P6 — Test & packaging

> **Completato**: 2026-03-06. Aggiunto `tests/test_smoke_library.py` con 8 test smoke
> (slug, make/split_dataset_key, build_sklearn_model×3, evaluate_binary, get_positive_class_scores).
> `pyproject.toml` aggiornato con tutte le dipendenze core e optional groups
> (xgboost, tuning, viz, dev, all). 10/10 test passano.

### Obiettivo
Aggiungere test smoke che importano da `football_ai`. Dichiarare dipendenze in `pyproject.toml`.

### Cosa fare

**Step 11: Aggiungere `tests/test_smoke_library.py`**

```python
"""Smoke tests for football_ai library functions."""
import numpy as np
import pandas as pd
import pytest

from football_ai.data import slug, make_dataset_key, split_dataset_key
from football_ai.training import build_sklearn_model
from football_ai.evaluation import evaluate_binary, get_positive_class_scores


def test_slug():
    assert slug("La Liga") == "la_liga"
    assert slug("1. Bundesliga") == "1_bundesliga"
    assert slug("Champions League 2015/2016") == "champions_league_2015_2016"


def test_make_dataset_key():
    assert make_dataset_key("La Liga", "2015/2016") == "la_liga_2015_2016"


def test_split_dataset_key():
    league, season = split_dataset_key("la_liga_2015_2016")
    assert league == "la liga"
    assert season == "2015/2016"


@pytest.mark.parametrize("model_name", ["rf", "logreg", "mlp"])
def test_build_sklearn_model(model_name):
    model = build_sklearn_model(model_name)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_evaluate_binary():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)
    metrics = evaluate_binary(y_proba, y_true)
    expected_keys = {"rows", "positive_rate", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"}
    assert expected_keys.issubset(metrics.keys())
    assert metrics["rows"] == 200.0


def test_get_positive_class_scores():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((100, 5)), columns=[f"f{i}" for i in range(5)])
    y = rng.integers(0, 2, size=100)
    model = LogisticRegression(max_iter=100, random_state=42).fit(X, y)
    scores = get_positive_class_scores(model, X)
    assert scores.shape == (100,)
    assert 0 <= scores.min() and scores.max() <= 1
```

Mantenere `tests/test_smoke_train.py` (test indipendenti, comunque utili).

**Step 12: Aggiornare `pyproject.toml`**

```toml
[project]
name = "football-ai"
version = "0.1.0"
description = "Utilities and scripts for StatsBomb SPADL/VAEP dataset creation and model training."
requires-python = ">=3.9"
dependencies = [
    "scikit-learn>=1.4",
    "pandas>=2.0",
    "numpy>=1.24",
    "socceraction>=1.5",
    "tables>=3.9",
    "tqdm",
    "pyyaml",
]

[project.optional-dependencies]
xgboost = ["xgboost>=2.0"]
tuning = ["optuna>=3.0"]
viz = ["matplotlib>=3.5"]
dev = ["pytest>=7.0"]
all = ["football-ai[xgboost,tuning,viz,dev]"]
```

### Verifica dopo P6

```bash
pip install -e ".[all]"
pytest tests/ -v
# Tutti i test devono passare
```

---

## P7 — Documentazione & notebook

> **Completato**: 2026-03-06. README.md riscritto con struttura e CLI attuali.
> Creato `data/README.md` con layout HDF5 e istruzioni di generazione.
> Notebook aggiornati: inline duplicati sostituiti con import da `football_ai.data` e
> `football_ai.evaluation`. Logica narrativa e struttura celle preservate.

### Obiettivo
Aggiornare README con struttura e comandi attuali. Creare `data/README.md`. Semplificare notebook.

### Cosa fare

**Step 13: Aggiornare `README.md`**

Nuova struttura progetto:
```text
├── configs/                       # YAML experiment configs
├── data/                          # Local data (gitignored)
├── logs/                          # Captured stdout/stderr logs
├── notebooks/                     # Exploratory notebooks
├── scripts/
│   ├── create_datasets.py         # Data preparation CLI
│   ├── train.py                   # sklearn training CLI
│   ├── train_xgboost.py           # XGBoost training CLI
│   └── tune_xgboost_bayes_v2.py   # Bayesian tuning CLI
├── src/football_ai/
│   ├── __init__.py
│   ├── data.py                    # Data loading, SPADL/VAEP, HDF5 I/O
│   ├── training.py                # Model building, training helpers
│   └── evaluation.py              # Metrics, threshold sweep, visualization
├── tests/
│   ├── test_smoke_train.py
│   └── test_smoke_library.py
├── archive/                       # Superseded scripts
├── pyproject.toml
└── requirements.txt
```

Aggiungere sezioni:
- **Training**: `python scripts/train.py --config configs/train_sklearn.yaml`
- **XGBoost Training**: `python scripts/train_xgboost.py --config configs/train_xgboost.yaml`
- **Bayesian Tuning**: `python scripts/tune_xgboost_bayes_v2.py --config configs/tune_xgboost.yaml`
- **Logging**: `python scripts/train.py --config configs/train_sklearn.yaml 2>&1 | tee logs/train.log`

**Step 13b: Creare `data/README.md`**

```markdown
# Data Directory

This directory contains HDF5 datasets (gitignored).

## Expected layout

```text
data/
├── spadl_data/
│   ├── major_leagues.h5      # Major European leagues
│   └── women_leagues.h5      # Women's leagues
```

## HDF5 file structure

Each `.h5` file contains these keys:

| Key | Format | Description |
|-----|--------|-------------|
| `actions` | table | SPADL actions |
| `games` | fixed | Games metadata |
| `teams` | fixed | Teams |
| `players` | table | Players |
| `competitions` | fixed | Competitions included |
| `labels` | table | VAEP labels (scores, concedes) |
| `full_data` | table | Merged actions + metadata + labels (for training) |

## How to generate data

1. Clone StatsBomb open-data: `git clone https://github.com/statsbomb/open-data ../open-data`
2. Run: `python scripts/create_datasets.py --config configs/create_datasets.yaml`
```

**Step 14: Semplificare notebook**

Per `create_spadl_dataset_major_leagues.ipynb` e `create_spadl_dataset_women_league_season.ipynb`:
- Rimuovere tutte le funzioni inline duplicate
- Sostituire con import: `from football_ai.data import ...`
- Mantenere la struttura narrativa (markdown cells)

Per `socceraction_supervised_learning.ipynb`:
- Aggiornare import: `from football_ai.training import ...`, `from football_ai.evaluation import ...`

### Verifica dopo P7

```bash
cat README.md  # struttura aggiornata
cat data/README.md  # esiste e descrive layout
```

---

## Riepilogo dipendenze tra priorità

```
P1 (data.py) ─────┐
                   ├──→ P3 (scripts)
P2 (evaluation.py) ┘
                        P3 ──→ P5 (YAML configs)
P4 (pulizia legacy) ← indipendente, può andare in parallelo a P1-P2
P6 (test+packaging) ← dopo P1+P2 (testa le nuove API)
P7 (docs+notebooks) ← dopo P1+P2+P3 (documenta lo stato finale)
```

**Ordine consigliato**: P1 → P2 → P3 → P4 → P5 → P6 → P7

P4 è indipendente e può essere fatto in qualsiasi momento (anche per primo, come warm-up).
