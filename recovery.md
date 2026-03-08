# Aggiornamento modifiche XGBoost — sessione 2026-03-08 (timestep: 2026-03-08T16:00Z)

Modifiche a metriche XGBoost, early stopping, pickling, config, e logica custom discusse/implementate in questa sessione.

---

1. **File**: src/football_ai/training.py  
    **Tipo**: modifica  
    **Cosa cambia**: Aggiunta gestione robusta di metriche custom XGBoost (recall, precision, f1), composite callable, callback injection, e decoupling early stopping metric.  
    **Codice**:
    ```python
    def _xgb_recall(y_true, y_score) -> float: ...
    def _xgb_precision(y_true, y_score) -> float: ...
    def _xgb_f1(y_true, y_score) -> float: ...
    _CUSTOM_XGB_METRICS = {"recall": ..., "precision": ..., "f1": ...}
    def _make_composite_xgb_metric(custom_names): ...
    def resolve_xgb_eval_metrics(eval_metric, early_stopping_rounds=None, early_stopping_metric=None): ...
    # Gestione EarlyStopping callback dinamica, maximize, metric_name, composite callable, companion callback.
    ```

2. **File**: src/football_ai/training.py  
    **Tipo**: modifica  
    **Cosa cambia**: Fix serializzazione modelli XGBoost con metriche custom: callables sostituiti da stringhe prima di pickling, callbacks rimossi.  
    **Codice**:
    ```python
    def _strip_xgb_callables(model): ...
    def save_model(model, filepath): ...
    # Temporaneamente sostituisce eval_metric/callbacks con versioni picklable.
    ```

3. **File**: scripts/train_xgboost.py  
    **Tipo**: modifica  
    **Cosa cambia**: Import e uso di resolve_xgb_eval_metrics; gestione 3-tuple (metrics, callbacks, es_rounds); lettura early_stopping_metric dal YAML; injection callbacks; rimozione early_stopping_metric dai parametri XGB.  
    **Codice**:
    ```python
    from football_ai.training import resolve_xgb_eval_metrics
    ...
    early_stopping_metric = model_cfg_from_yaml.get("early_stopping_metric")
    resolved_metrics, extra_callbacks, es_rounds = resolve_xgb_eval_metrics(...)
    effective_model_config["eval_metric"] = resolved_metrics
    effective_model_config["early_stopping_rounds"] = es_rounds
    if extra_callbacks:
         existing_cbs = effective_model_config.get("callbacks") or []
         effective_model_config["callbacks"] = extra_callbacks + list(existing_cbs)
    ```

4. **File**: configs/train_xgboost.yaml  
    **Tipo**: modifica  
    **Cosa cambia**: Aggiornate metriche di eval (tutte e 6); aggiunto campo early_stopping_metric per selezionare la metrica di early stopping indipendentemente dall'ordine in eval_metric.  
    **Codice**:
    ```yaml
    eval_metric: ['aucpr', 'auc', 'logloss', 'f1', 'precision', 'recall']
    early_stopping_rounds: 10
    early_stopping_metric: "recall"
    ```

5. **File**: src/football_ai/training.py  
    **Tipo**: modifica  
    **Cosa cambia**: Import esplicito TrainingCallback per tipizzazione e callback injection.  
    **Codice**:
    ```python
    from xgboost.callback import TrainingCallback
    ```

6. **File**: scripts/train_xgboost.py  
    **Tipo**: modifica (minore)  
    **Cosa cambia**: Aggiornato commento su EVAL_METRIC per riflettere nuovo ordine e logica.  
    **Codice**:
    ```python
    # EVAL_METRIC: str | list[str] = ["logloss", "auc", "aucpr"]
    EVAL_METRIC: str | list[str] = ["aucpr", "auc", "logloss"]
    ```

7. **File**: scripts/train_xgboost.py  
    **Tipo**: modifica (minore)  
    **Cosa cambia**: Esclusione di early_stopping_metric dai parametri passati a XGBClassifier (solo usato per callback injection).  
    **Codice**:
    ```python
    _NON_XGB_KEYS = {"early_stopping_metric"}
    ...
    for k, v in model_cfg_from_yaml.items():
         if k in _NON_XGB_KEYS:
              continue
    ```

---

Tutte queste modifiche sono attive e testate (salvo dove indicato).  
Timestamp: 2026-03-08T16:00Z
# Recovery log — sessione 2026-03-08

Modifiche al codice discusse/implementate durante la sessione Copilot del 2026-03-08.
Alcune modifiche a `scripts/train_xgboost.py` sono andate perse per un bug EOS filesystem e vanno ri-applicate.

---

## 1. Import `datetime` in `train_xgboost.py` (~12:00 UTC)

- **File**: `scripts/train_xgboost.py` (riga 27)
- **Tipo**: modifica
- **Descrizione**: aggiunto `from datetime import datetime` per generare `run_id` timestamp.
- **Stato**: ⚠️ PERSO — file tornato alla versione git

```diff
 import argparse
 import logging
+from datetime import datetime
 from pathlib import Path
```

---

## 2. Variabile `run_id` timestamp (~12:00 UTC)

- **File**: `scripts/train_xgboost.py` (dopo riga ~285, dopo `models_path.mkdir`)
- **Tipo**: modifica
- **Descrizione**: aggiunta `run_id = datetime.now().strftime("%Y%m%d_%H%M%S")` subito dopo la creazione delle directory di output, per rendere unici tutti gli artefatti.
- **Stato**: ⚠️ PERSO

```diff
     models_path.mkdir(parents=True, exist_ok=True)
+
+    # Unique run identifier (timestamp) to avoid overwriting previous artifacts
+    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
```

---

## 3. Filename modello con timestamp + seed + metrica early stopping (~12:00–12:30 UTC)

- **File**: `scripts/train_xgboost.py` (riga ~370)
- **Tipo**: modifica
- **Descrizione**: da `f"xgboost_{target_col}_{random_state}.pkl"` a filename con `run_id`, `seed`, e valore della metrica di early stopping per autodescrizione.
- **Stato**: ⚠️ PERSO

```diff
-    model_filename = f"xgboost_{target_col}_{random_state}.pkl"
-    model_path = models_path / model_filename
-    save_model(model, model_path)
+    es_metric_name = early_stopping_metric or "best"
+    es_metric_value = getattr(model, "best_score", None)
+    if es_metric_value is not None:
+        metric_tag = f"_{es_metric_name}={es_metric_value:.4f}"
+    else:
+        metric_tag = ""
+    model_filename = (
+        f"xgboost_{target_col}_{run_id}_seed{random_state}{metric_tag}.pkl"
+    )
+    model_path = models_path / model_filename
+    save_model(model, model_path)
```

Esempio output: `xgboost_scores_20260308_115602_seed583201_recall=0.7123.pkl`

---

## 4. Timestamp aggiunto a tutti gli artefatti CSV e PNG (~12:00 UTC)

- **File**: `scripts/train_xgboost.py` (righe ~392–452)
- **Tipo**: modifica
- **Descrizione**: `{run_id}` aggiunto ai filename di threshold_sweep CSV, metrics CSV, confusion matrix PNG, config CSV.
- **Stato**: ⚠️ PERSO

```diff
-    threshold_sweep_df.to_csv(results_path / f"threshold_sweep_xgboost_{target_col}.csv", ...)
+    threshold_sweep_df.to_csv(results_path / f"threshold_sweep_xgboost_{target_col}_{run_id}.csv", ...)

-    results_df.to_csv(results_path / f"metrics_xgboost_positive-focus_{target_col}.csv", ...)
+    results_df.to_csv(results_path / f"metrics_xgboost_positive-focus_{target_col}_{run_id}.csv", ...)

-    plot_confusion_matrix(train_cm, "train", results_path / f"confusion_matrix_train_{target_col}.png")
+    plot_confusion_matrix(train_cm, "train", results_path / f"confusion_matrix_train_{target_col}_{run_id}.png")
     # (idem per validation e target)

-    config_df.to_csv(results_path / f"config_xgboost_{target_col}.csv", ...)
+    config_df.to_csv(results_path / f"config_xgboost_{target_col}_{run_id}.csv", ...)
```

---

## 5. `RANDOM_STATE` da `20260307` a `None` (~12:15 UTC)

- **File**: `scripts/train_xgboost.py` (riga 81)
- **Tipo**: modifica
- **Descrizione**: rimosso seed hardcoded che causava overwrite identico ad ogni run; il seed viene ora dal YAML/CLI o generato random dal fallback.
- **Stato**: ⚠️ PERSO — attualmente ancora `20260307`

```diff
-RANDOM_STATE: int | None = 20260307
+RANDOM_STATE: int | None = None
```

---

## 6. `resolve_random_state` — fallback random + gestione robusta tipi (~12:15 UTC)

- **File**: `src/football_ai/config.py` (righe 85–140)
- **Tipo**: modifica
- **Descrizione**: (a) fallback da `int(date.today())` a `random.randint(0, 999_999)`, (b) aggiunto `import random`, (c) gestione di stringhe numeriche (`"42"` → `42`), float (`3.0` → `3`), e stringa `"today"` → data YYYYMMDD.
- **Stato**: ✅ ATTIVO

```python
for c in candidates:
    if c is None:
        continue
    if isinstance(c, str):
        stripped = c.strip().lower()
        if stripped == "today":
            return int(datetime.date.today().strftime("%Y%m%d"))
        if stripped in ("", "none"):
            continue
        try:
            return int(stripped)
        except ValueError:
            continue
    if isinstance(c, (int, float)):
        return int(c)
fallback = random.randint(0, 1_000_000 - 1)
```

---

## 7. `setup_logging` function aggiunta (~12:20 UTC)

- **File**: `src/football_ai/config.py` (righe 28–82)
- **Tipo**: modifica (nuova funzione)
- **Descrizione**: configura root logger con console + file handler e genera logfile con timestamp `logfile_{script_name}_{YYYYMMDD_HHMMSS}.log`.
- **Stato**: ✅ ATTIVO

---

## 8. Test aggiornati per il nuovo fallback random (~12:15 UTC)

- **File**: `tests/test_resolve_random_state.py` (righe 33–50)
- **Tipo**: modifica
- **Descrizione**: 3 test sostituiti: `test_fallback_to_today` → `test_fallback_to_random` (verifica range `[0, 1M)`), `test_fallback_no_args` aggiornato, `test_fallback_uses_specific_date` → `test_fallback_uses_random_module` (mock di `random.randint`). 12/12 test passano.
- **Stato**: ✅ ATTIVO

```diff
-    def test_fallback_to_today(self) -> None:
-        expected = int(datetime.date.today().strftime("%Y%m%d"))
-        assert resolve_random_state(None, None) == expected
+    def test_fallback_to_random(self) -> None:
+        result = resolve_random_state(None, None)
+        assert isinstance(result, int)
+        assert 0 <= result < 1_000_000

-    @patch("football_ai.config.datetime")
-    def test_fallback_uses_specific_date(self, mock_dt):
-        mock_dt.date.today.return_value = datetime.date(2030, 1, 15)
-        assert resolve_random_state(None) == 20300115
+    @patch("football_ai.config.random")
+    def test_fallback_uses_random_module(self, mock_random):
+        mock_random.randint.return_value = 42
+        assert resolve_random_state(None) == 42
+        mock_random.randint.assert_called_once_with(0, 999_999)
```

---

## 9. Restore `src/football_ai/` dopo cancellazione EOS (~12:25 UTC)

- **File**: directory `src/football_ai/` (7 file)
- **Tipo**: restore
- **Descrizione**: l'intera directory era sparita per bug EOS filesystem (`ls` → "No such file", `mkdir` → "File exists"). Risolto con `mv src src_old && git checkout -- src/`, poi `config.py` ri-modificato con le patch sopra.
- **Stato**: ✅ ATTIVO

---

## Riepilogo stato

| # | File | Stato |
|---|------|-------|
| 1–5 | `scripts/train_xgboost.py` | ⚠️ Da ri-applicare |
| 6–7 | `src/football_ai/config.py` | ✅ Attivo |
| 8 | `tests/test_resolve_random_state.py` | ✅ Attivo |
| 9 | `src/football_ai/` (restore) | ✅ Attivo |

---
---

# Sessione 2026-03-08 (pomeriggio) — Custom XGBoost eval metrics

Modifiche implementate per supportare `recall`, `precision` e `f1` come
metriche di valutazione/early-stopping in XGBoost (che non le supporta
come built-in nel motore C++).

---

## 10. Aggiunto import `resolve_xgb_eval_metrics` in `train_xgboost.py` (~14:00 UTC)

- **File**: `scripts/train_xgboost.py` (riga 47)
- **Tipo**: modifica
- **Descrizione**: aggiunto `resolve_xgb_eval_metrics` nel blocco import da `football_ai.training`.
- **Stato**: ✅ ATTIVO

```diff
 from football_ai.training import (
     build_xgb_eval_set,
     drop_none_params,
     load_xy_source_calib_target_split,
+    resolve_xgb_eval_metrics,
     save_model,
 )
```

---

## 11. Primo `resolve_xgb_eval_metrics` — solo `recall` (~14:00 UTC)

- **File**: `src/football_ai/training.py` (dopo `drop_none_params`, ~riga 145)
- **Tipo**: modifica (nuovo blocco)
- **Descrizione**: aggiunta funzione `_xgb_recall` con signature `(y_true, y_score) -> float` (threshold 0.5), costante `_XGB_BUILTIN_METRICS`, registro `_CUSTOM_XGB_METRICS`, e funzione `resolve_xgb_eval_metrics` che sostituisce stringhe non-built-in con le callable corrispondenti.
- **Stato**: ⚠️ SOSTITUITO dalla versione completa (punto 13)

---

## 12. Prima chiamata `resolve_xgb_eval_metrics` in `train_xgboost.py` (~14:00 UTC)

- **File**: `scripts/train_xgboost.py` (~riga 327)
- **Tipo**: modifica
- **Descrizione**: aggiunta risoluzione metriche custom prima di creare `XGBClassifier`. Return type era semplice `list | None`.
- **Stato**: ⚠️ SOSTITUITO dalla versione a 3-tuple (punto 14)

```python
# Versione iniziale (solo recall, return semplice):
effective_model_config["eval_metric"] = resolve_xgb_eval_metrics(
    effective_model_config.get("eval_metric")
)
```

---

## 13. `resolve_xgb_eval_metrics` v2 — `precision`, `f1`, composite callable (~15:00 UTC)

- **File**: `src/football_ai/training.py` (righe 145–447)
- **Tipo**: modifica (riscrittura completa del blocco custom metrics)
- **Descrizione**: riscrittura per gestire il limite di XGBoost ≤ 3.2 (max 1 callable custom in `eval_metric`). Componenti:
  1. **3 funzioni custom**: `_xgb_recall`, `_xgb_precision`, `_xgb_f1`
  2. **`_make_composite_xgb_metric`**: quando servono ≥ 2 custom, le raggruppa in un'unica callable composite + un `TrainingCallback` (`_ExtraMetricsInjector`) che inietta i valori extra in `evals_log`
  3. **`resolve_xgb_eval_metrics`** con signature estesa `(eval_metric, early_stopping_rounds, early_stopping_metric) -> (resolved, callbacks, es_rounds_out)`; gestisce direzione maximize/minimize e inietta `EarlyStopping` callback esplicito se necessario
  4. **Costanti**: `_CUSTOM_METRICS_MAXIMIZE`, `_BUILTIN_METRICS_MAXIMIZE` per determinare la direzione early-stopping
- **Stato**: ✅ ATTIVO — versione finale

```python
# Funzioni custom (signature: y_true, y_score -> float, threshold 0.5):
def _xgb_recall(y_true, y_score) -> float: ...
def _xgb_precision(y_true, y_score) -> float: ...
def _xgb_f1(y_true, y_score) -> float: ...

_xgb_recall.__name__ = "recall"
_xgb_precision.__name__ = "precision"
_xgb_f1.__name__ = "f1"

_CUSTOM_XGB_METRICS = {"recall": _xgb_recall, "precision": _xgb_precision, "f1": _xgb_f1}

# Composite per gestire limite 1-callable di XGBoost:
def _make_composite_xgb_metric(custom_names) -> tuple[callable, TrainingCallback]: ...

# Resolver pubblico:
def resolve_xgb_eval_metrics(
    eval_metric, early_stopping_rounds=None, early_stopping_metric=None,
) -> tuple[list, list[callback], int | None]: ...
```

---

## 14. Aggiornata chiamata `resolve_xgb_eval_metrics` in `train_xgboost.py` (~15:00 UTC)

- **File**: `scripts/train_xgboost.py` (righe 330–347)
- **Tipo**: modifica
- **Descrizione**: aggiornata la call site per gestire il return a 3 tuple `(metrics, callbacks, es_rounds)`. Legge `early_stopping_metric` dal YAML. Inietta callbacks e annulla `early_stopping_rounds` sul modello se gestito dal callback esplicito.
- **Stato**: ✅ ATTIVO

```python
early_stopping_metric: str | None = model_cfg_from_yaml.get("early_stopping_metric")
resolved_metrics, extra_callbacks, es_rounds = resolve_xgb_eval_metrics(
    effective_model_config.get("eval_metric"),
    early_stopping_rounds=effective_model_config.get("early_stopping_rounds"),
    early_stopping_metric=early_stopping_metric,
)
effective_model_config["eval_metric"] = resolved_metrics
effective_model_config["early_stopping_rounds"] = es_rounds
if extra_callbacks:
    existing_cbs = effective_model_config.get("callbacks") or []
    effective_model_config["callbacks"] = extra_callbacks + list(existing_cbs)
```

---

## 15. Aggiornato YAML `configs/train_xgboost.yaml` (~15:00 UTC)

- **File**: `configs/train_xgboost.yaml` (righe 60–66)
- **Tipo**: modifica
- **Descrizione**: aggiornate metriche di eval per includere tutte e 6; aggiunto campo `early_stopping_metric` per selezionare la metrica di early stopping indipendentemente dall'ordine in `eval_metric`.
- **Stato**: ✅ ATTIVO

```diff
-  eval_metric: ['aucpr', 'auc', 'logloss', 'recall']
-  early_stopping_rounds: 100
+  eval_metric: ['aucpr', 'auc', 'logloss', 'f1', 'precision', 'recall']
+  early_stopping_rounds: 10
+  # Which metric to monitor for early stopping (default: last in eval_metric).
+  # Decouples stopping criterion from eval_metric order.
+  early_stopping_metric: "recall"
```

---

## 16. Import `TrainingCallback` in `training.py` (~15:00 UTC)

- **File**: `src/football_ai/training.py` (riga 161)
- **Tipo**: modifica
- **Descrizione**: aggiunto `from xgboost.callback import TrainingCallback` inline nel blocco custom metrics (usato da `_ExtraMetricsInjector` e dal tipo di ritorno di `_make_composite_xgb_metric`).
- **Stato**: ✅ ATTIVO

```python
from xgboost.callback import TrainingCallback
```

---

## Riepilogo stato aggiornato

| # | File | Stato |
|---|------|-------|
| 1–5 | `scripts/train_xgboost.py` (timestamp, run_id, seed) | ⚠️ Da ri-applicare |
| 6–7 | `src/football_ai/config.py` | ✅ Attivo |
| 8 | `tests/test_resolve_random_state.py` | ✅ Attivo |
| 9 | `src/football_ai/` (restore) | ✅ Attivo |
| 10 | `scripts/train_xgboost.py` (import) | ✅ Attivo |
| 11 | `src/football_ai/training.py` (recall only) | ⚠️ Sostituito da #13 |
| 12 | `scripts/train_xgboost.py` (call site v1) | ⚠️ Sostituito da #14 |
| 13 | `src/football_ai/training.py` (custom metrics v2) | ✅ Attivo |
| 14 | `scripts/train_xgboost.py` (call site v2, 3-tuple) | ✅ Attivo |
| 15 | `configs/train_xgboost.yaml` (metriche + early_stopping_metric) | ✅ Attivo |
| 16 | `src/football_ai/training.py` (import TrainingCallback) | ✅ Attivo |
---
---

# Sessione 2026-03-08 (sera) — Fair zero-shot baseline in few-shot script

Timestamp: 2026-03-08T~18:00 UTC

Unica modifica al codice in questa sessione. Le altre interazioni erano
istruzioni d'uso (`python scripts/xgboost_fewshots.py --config …`) e
spiegazione ad alto livello dello script.

---

## 17. Fair zero-shot baseline: eval spostata dentro il loop budget×seed (~18:00 UTC)

- **File**: `scripts/xgboost_fewshots.py`
- **Tipo**: modifica (6 edit atomici applicati insieme)
- **Descrizione**: la baseline source-only veniva valutata 1 volta su **tutto** il target; ora viene valutata per ogni `(budget, seed)` sullo **stesso holdout** usato da target-only e fine-tune, rendendo il confronto fair.
- **Stato**: ✅ ATTIVO

### 17a. Rimossa eval globale zero-shot, mantenuto solo caricamento modello + soglia

```diff
-    # ── Source-only baseline ──
-    ...
-    X_target_aligned = X_target.reindex(columns=feature_cols, fill_value=0)
-    X_source_val_aligned = X_source_val.reindex(columns=feature_cols, fill_value=0)
-    source_threshold = _select_threshold(...)
-    source_metrics = _evaluate(
-        pretrained_model, X_target_aligned, y_target, threshold=source_threshold,
-    )
-    logger.info("Source-only (zero-shot) on target: PR-AUC=%.4f ...")
+    # ── Source-only baseline (model + threshold — eval moved inside loop) ──
+    ...
+    X_source_val_aligned = X_source_val.reindex(columns=feature_cols, fill_value=0)
+    source_threshold = _select_threshold(...)
+    logger.info("Source-only threshold (from source val): %.4f", source_threshold)
```

### 17b. `n_combos` da 2 a 3 scenari

```diff
-    n_combos = len(budgets) * len(seeds) * 2  # 2 scenarios
+    n_combos = len(budgets) * len(seeds) * 3  # 3 scenarios
```

### 17c. Aggiunto step 0 (source-only) dentro il loop, prima di target-only

```diff
             X_hold = X_hold.reindex(columns=feature_cols, fill_value=0)

+            # ── 0) Source-only (zero-shot on same holdout) ──
+            combo_i += 1
+            logger.info(
+                "[%d/%d] source-only  budget=%d%%  seed=%d  "
+                "(%d holdout actions)",
+                combo_i, n_combos, budget * 100, seed, len(X_hold),
+            )
+            src_metrics = _evaluate(
+                pretrained_model, X_hold, y_hold, threshold=source_threshold,
+            )
+            src_metrics.update({
+                "scenario": "source_only",
+                "budget": budget,
+                "seed": seed,
+                "threshold": source_threshold,
+                "n_few_games": n_few_games,
+                "n_few_actions": len(X_few),
+                "n_holdout_actions": len(X_hold),
+            })
+            all_rows.append(src_metrics)
+
             # ── 1) Target-only (from scratch) ──
```

### 17d. `_plot_fewshot_curves` — rimosso param `source_metrics`, 3 curve con banda d'errore

```diff
-def _plot_fewshot_curves(summary_df, source_metrics, target_col, output_dir):
-    ...
-    palette = {"target_only": "#1f77b4", "finetune": "#ff7f0e"}
-    for ax, ... in zip(axes, metrics_to_plot):
-        src_val = source_metrics.get(metric_key, np.nan)
-        ax.axhline(src_val, color="#2ca02c", ls="--", ...)
-        for scenario, color in palette.items():
-            ...
+def _plot_fewshot_curves(summary_df, target_col, output_dir):
+    ...
+    palette = {
+        "source_only": ("#2ca02c", "--"),
+        "target_only": ("#1f77b4", "-"),
+        "finetune":    ("#ff7f0e", "-"),
+    }
+    for ax, ... in zip(axes, metrics_to_plot):
+        for scenario, (color, ls) in palette.items():
+            sub = summary_df[summary_df["scenario"] == scenario]...
+            ax.plot(xs, means, marker="o", color=color, ls=ls, ...)
+            ax.fill_between(xs, means - stds, means + stds, ...)
```

### 17e. `_print_and_save_table` — rimosso param `source_metrics`, fonte da summary_df

```diff
-def _print_and_save_table(summary_df, source_metrics, target_col, output_dir):
-    ...
-    # Source-only row (singola, hardcoded)
-    row = {"Model": "Source-only (no adapt)", "Budget target": "0%"}
-    ...
-    for budget in budgets_pct:
-        for scenario in ["target_only", "finetune"]:
+def _print_and_save_table(summary_df, target_col, output_dir):
+    ...
+    nice_names = {
+        "source_only": "Source-only (0-shot)",
+        "target_only": "Target-only",
+        "finetune": "Fine-tune (xgb_model)",
+    }
+    for budget in budgets_pct:
+        for scenario in ["source_only", "target_only", "finetune"]:
+            ...  # legge mean ± std da summary_df come gli altri
```

### 17f. Aggiornate chiamate finali (rimosso `source_metrics`)

```diff
-    _print_and_save_table(summary_df, source_metrics, target_col, results_dir)
-    _plot_fewshot_curves(summary_df, source_metrics, target_col, results_dir)
+    _print_and_save_table(summary_df, target_col, results_dir)
+    _plot_fewshot_curves(summary_df, target_col, results_dir)
```

---

## Riepilogo stato aggiornato (inclusa sessione sera)

| # | File | Stato |
|---|------|-------|
| 1–5 | `scripts/train_xgboost.py` (timestamp, run_id, seed) | ⚠️ Da ri-applicare |
| 6–7 | `src/football_ai/config.py` | ✅ Attivo |
| 8 | `tests/test_resolve_random_state.py` | ✅ Attivo |
| 9 | `src/football_ai/` (restore) | ✅ Attivo |
| 10 | `scripts/train_xgboost.py` (import) | ✅ Attivo |
| 11 | `src/football_ai/training.py` (recall only) | ⚠️ Sostituito da #13 |
| 12 | `scripts/train_xgboost.py` (call site v1) | ⚠️ Sostituito da #14 |
| 13 | `src/football_ai/training.py` (custom metrics v2) | ✅ Attivo |
| 14 | `scripts/train_xgboost.py` (call site v2, 3-tuple) | ✅ Attivo |
| 15 | `configs/train_xgboost.yaml` (metriche + early_stopping_metric) | ✅ Attivo |
| 16 | `src/football_ai/training.py` (import TrainingCallback) | ✅ Attivo |
| **17** | **`scripts/xgboost_fewshots.py` (fair zero-shot baseline)** | **✅ Attivo** |