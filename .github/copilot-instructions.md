# .github/copilot-instructions.md
# GitHub Copilot: repository instructions (Python ML research project)

## Goals (priorities)
Help contributors write ML code that is:
- **Clear, simple, and clean** (avoid over-engineering).
- **Reusable and easy to modify**, especially for users who are not expert coders.
- **Consistently named and formatted**.
- **Research-friendly**: reproducible experiments, transparent assumptions, readable outputs.

Default to the **smallest change** that achieves the goal. Prefer clarity over cleverness.

---

## Current repo shape (minimal, OK for early-stage projects)
Keep the early structure small. Use this as a starting point:

- `data/`                     # local training/eval data (not committed); include README.md with expected layout
- `logs/`                     # captured stdout/stderr logs (e.g., `tee logs/train.log`)
- `scripts/`                  # thin entrypoints (train/eval/predict); no core logic here
- `src/<project_name>/`       # library code (installable package)
  - `datasets.py`             # dataset loading, parsing, simple preprocessing
  - `features.py`             # feature extraction (sklearn) / tensor prep
  - `models/`                 # torch models + sklearn wrappers if needed
  - `training.py`             # training loop(s), loss, optimizer, schedulers
  - `evaluation.py`           # metrics, evaluation routines
  - `io.py`                   # save/load checkpoints, artifacts, predictions
  - `utils.py`                # small utilities (keep tight scope)
- `tests/`                    # smoke tests + a few unit tests for critical pieces
- `pyproject.toml`            # packaging as a library
- `README.md`                 # quickstart + how to run experiments
- `configs/`                  # YAML experiment configs (optional but recommended)

Rules:
- **Core logic lives in `src/`**. `scripts/` should mostly parse args + config and call into the library.
- **Do not put core logic in notebooks** (if you add `notebooks/`, keep it exploratory and disposable).
- Avoid deep nesting and too many micro-files; group by function until growth forces separation.

---

## Future structure (only if/when the project grows)
If the codebase grows (multiple datasets/models, many experiments, shared components), migrate towards:

- `src/<project_name>/`
  - `data/`        # dataset loading, preprocessing, transforms
  - `models/`      # model definitions
  - `training/`    # loops, losses, optimizers, schedulers
  - `evaluation/`  # metrics and pipelines
  - `utils/`       # shared helpers

**Do not restructure prematurely.** Introduce extra folders only when it reduces confusion.

---

## Stack assumptions (for this repo)
- Primary: **scikit-learn + PyTorch** (hybrid workflows are common here).
- Configs: **YAML + argparse**.
- Tracking: **none for now**; consider adding **Weights & Biases (wandb)** later.
- Packaging: repo is an **installable library** (importable from `src/`).

---

## CLI + configs (YAML + argparse)
### Single source of truth
- Config defaults should live in YAML (human-readable, explicit).
- CLI should allow **overrides** without introducing hidden defaults in code.

Preferred pattern:
- `--config configs/exp.yaml` required (or has a clear default).
- Optional overrides like `--seed 123`, `--lr 1e-3`, etc.

### Config conventions
- Use **clear key names** (no cryptic abbreviations).
- Include units/meaning in comments when helpful.
- Keep configs small; prefer composition by copy-pasting a base config early on (no Hydra unless needed).

Example keys (illustrative):
- `data.name`, `data.path`, `data.split`
- `model.name`, `model.hidden_dim`
- `train.epochs`, `train.batch_size`, `train.lr`, `train.seed`
- `eval.metrics`

When adding new config keys:
- Add a sensible default.
- Update the config template and README example.

---

## Library boundaries (sklearn + torch)
### Preferred architecture
- Keep **dataset/feature extraction** and **model training** separate.
- Use sklearn where it shines (preprocessing, classical baselines).
- Use torch for deep models and GPU training.

Good patterns:
- `features.py` provides:
  - `fit_transform_train(...)` / `transform_eval(...)` for sklearn pipelines
  - Optional `to_tensor(...)` utilities for torch input
- `models/` may include:
  - `torch.nn.Module` implementations
  - Optional sklearn-compatible wrappers (only if it improves usability)

Avoid:
- Complex registries, plugin systems, factories.
- Multiple competing training entrypoints.
- Magical behavior at import time.

---

## Logging (simple now, wandb later)
### Now (recommended)
- Use Python `logging` (plus optional tqdm).
- Always log:
  - dataset name/split + data path
  - seed
  - model name + parameter count (for torch models)
  - key hyperparameters
  - metrics per epoch / per evaluation run
  - output artifact paths (checkpoints, predictions)

### Capturing logs to `logs/`
Provide examples in README such as:
- `python -m scripts.train --config configs/exp.yaml 2>&1 | tee logs/train.log`

### wandb (future)
Do **not** add wandb by default yet unless actively used.
If you add a hook, keep it optional and off by default:
- `--use_wandb` flag
- If wandb is missing, fail gracefully with an informative error.

---

## Reproducibility expectations
- Set and log seeds (`random`, `numpy`, and `torch`).
- Make device choice explicit (`cpu`/`cuda`).
- Save:
  - config used for the run (copy into output dir)
  - trained model checkpoint (if applicable)
  - evaluation outputs (metrics JSON/CSV)

Avoid hidden randomness (e.g., unseeded data splits).

---

## Testing (smoke tests preferred)
- Use `pytest`.
- Include at least:
  1) A **smoke training** test on a tiny toy dataset (CPU, very fast).
  2) A smoke eval/predict test that loads artifacts produced by the smoke train.
- Keep tests < ~1 minute total.
- Avoid heavyweight integration tests unless they stay tiny and deterministic.

---

## Naming & style
- Use **informative names**:
  - `train_one_epoch`, `build_dataloader`, `fit_preprocessor`, `compute_metrics`
- Avoid abbreviations unless standard (`lr`, `cfg`, `num_epochs`).
- Prefer `pathlib.Path` over `os.path`.
- Prefer dataclasses for simple configuration containers (optional).
- Keep functions reasonably short; split only when it improves readability.

### Type hints
- Add type hints where they improve understanding (especially public APIs).
- For tensors/arrays, mention shapes in docstrings.

### Docstrings
Public functions/classes must include:
- what it does
- args/returns with types + shapes (for tensors/arrays)
- key assumptions / side effects

---

## When uncertain
If there are multiple plausible choices (e.g., how to structure sklearn+torch interoperability, artifact formats, training loop style):
1. **Ask before implementing**.
2. Provide **2–3 options with trade-offs**.
3. Recommend one option, defaulting to simplicity and user-friendliness.
