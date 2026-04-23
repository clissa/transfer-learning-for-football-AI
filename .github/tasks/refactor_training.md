# Refactor Training Handoff

## Goal of this note

This note captures the intended scope of the next training refactor for the
XGBoost `scores` workflow.

The immediate goal is not domain adaptation.

The immediate goal is to maximize **in-domain fit** on **La Liga seasons** with
a cleaner and faster training pipeline, then prepare a better tuning workflow
that can search sensible hyperparameter ranges on top of the new feature space.

Later, once the in-domain training path is stable, the project can return to
cross-domain evaluation and transfer learning for non-EU target leagues.

## High-level decisions

### 1. Keep the current source focus

For now, leave the source conceptually as-is:

- the source domain is the La Liga training domain
- train and validation are in-domain splits within that source domain
- the current objective is better fit inside source, not robustness to shift

This should be made explicit in configs, comments, artifact naming, and tuning
logic so there is no confusion about what the optimization target is.

### 2. Defer domain adaptation

Target data should eventually correspond to **non-EU leagues** for downstream
domain-adaptation experiments.

But that is **not** the objective of this refactor.

For now:

- the tuning objective should be in-domain validation performance only
- non-EU target data should not drive model selection
- transfer learning and shift handling should be postponed to a later pass

### 3. Persist engineered training features

The new `scores` feature engineering logic should no longer be recomputed inside
training on every run.

It should be persisted as a separate dataset artifact under:

- `data/feat_engineered_vaep_data`

This artifact should represent the train-ready engineered feature table for the
`scores` path.

## Current problems to address

### 1. Training-time feature engineering is too expensive

The current logic in `src/football_ai/training.py`:

- decodes one-hot categorical blocks into categorical columns
- derives geometry features
- derives possession continuity features

all during training.

That work is deterministic and belongs in a data-preparation stage, not inside
every train / eval / tuning run.

### 2. There is a regression in `scripts/train_xgboost.py`

`y_proba_val` is used during threshold sweeping before it is defined.

This must be fixed as part of the refactor.

### 3. `load_xy_competition_season_split(...)` does too much eagerly

The current implementation:

1. loads the full dataset
2. prepares engineered features for all rows
3. materializes all splits
4. only then trains on the source train/validation split

This is wasteful for large HDF5 datasets.

The loader should instead:

1. resolve split membership from lightweight raw metadata
2. split source into train/validation first
3. materialize source train/validation immediately
4. materialize non-source splits lazily only when they are actually needed

### 4. Precision is higher than necessary in several training paths

The tuning script already shows the intended direction:

- dense numeric features as `float32`
- boolean / indicator features as `uint8`
- categorical codes as `int32`
- labels as `uint8`

That casting policy should be applied consistently across all XGBoost-related
training scripts.

## Refactor scope

## 1. Persist engineered features into `data/feat_engineered_vaep_data`

Create a dedicated engineered-data stage for the `scores` training workflow.

Recommended shape:

1. read a VAEP dataset from `data/vaep_data/*.h5`
2. apply the current `scores` feature engineering logic
3. persist the result to `data/feat_engineered_vaep_data/*.h5`

The persisted schema should include:

- compact categorical features replacing one-hot action blocks
- geometry features
- possession continuity features
- the metadata needed for downstream splitting and evaluation
- labels (`scores`, `concedes`)

Recommended implementation direction:

- move feature engineering logic out of the training-only path
- reuse it from a dedicated script or data helper
- keep raw VAEP data and engineered VAEP data as separate artifacts

Recommended files to add:

- a dedicated script such as `scripts/create_feat_engineered_vaep.py`
- a matching config such as `configs/create_feat_engineered_vaep.yaml`

`scripts/create_vaep_features.py` should continue to represent the raw VAEP
generation step, while the new step should represent train-ready engineered
data.

## 2. Fix the regression in `scripts/train_xgboost.py`

After `model.fit(...)`, compute validation probabilities immediately before the
threshold sweep.

The threshold sweep should use:

- `y_val`
- `y_proba_val`

where `y_proba_val` has already been computed from the trained model.

This should also be covered by a smoke or regression test.

## 3. Refactor `load_xy_competition_season_split(...)`

Change the loader so that it does not eagerly prepare and store every split.

Desired behavior:

1. read raw dataset
2. resolve competition-season membership
3. isolate source rows
4. split source matches into train/validation
5. build `X_train` and `X_val` first
6. delay `calib`, `test_*`, and `target` feature materialization until needed

Recommended contract changes:

- either expose lazy split materialization methods
- or separate source loading from evaluation-split loading

The main requirement is that training no longer depends on building engineered
feature matrices for every non-source row up front.

## 4. Standardize lower-precision dtypes across training scripts

Use the casting policy already present in `scripts/tune_xgboost.py` as
the baseline.

Apply that policy consistently to:

- `scripts/train_xgboost.py`
- `scripts/xgboost_fewshots.py`
- `scripts/eval_xgboost_target.py`
- the updated tuning script
- shared helpers in `src/football_ai/training.py`

Desired dtype policy:

- continuous numeric columns → `float32`
- booleans / binary engineered indicators → `uint8`
- categorical integer codes → `int32`
- labels → `uint8`

This policy should be centralized in shared helpers instead of reimplemented in
each script.

## 5. Clarify in-domain objective in configs and code comments

The current refactor should explicitly state:

- source is La Liga
- validation is an in-domain held-out match split
- model selection is based on in-domain validation only
- cross-domain performance is not the current optimization target

Artifact naming should also reflect this where useful, for example by avoiding
language that implies the current best model is already optimized for domain
shift.

## 6. Prepare the next tuning script redesign

The final goal of this refactor is an updated tuning workflow that can search
sensible hyperparameter ranges on top of the persisted engineered feature set.

This should be an **implementation plan**, not just informal notes.

The tuning workflow should:

1. read from `data/feat_engineered_vaep_data`
2. load only the in-domain source data needed for tuning
3. split train/validation by `game_id`
4. cast features using the shared low-precision policy
5. fit with early stopping
6. rank trials primarily on in-domain validation quality
7. log enough diagnostics to support later transfer-learning work

## Class imbalance and feature considerations for tuning

These considerations should be preserved in the implementation plan and used to
inform the updated tuning script.

### 1. Positive class imbalance is severe

The `scores` target is rare, so the search space must take imbalance seriously.

Implications:

- prefer `aucpr` as the main ranking metric
- keep `logloss` and calibration-oriented metrics as secondary diagnostics
- search `scale_pos_weight` explicitly instead of fixing it once
- keep `max_delta_step` in the candidate space
- treat threshold tuning as post-fit evaluation, not the early-stopping target

### 2. The new feature space is smaller but more structured

The engineered `scores` feature space:

- compresses large one-hot blocks into compact categorical columns
- adds geometry features
- adds possession continuity features

Implications:

- dimensionality is lower, but correlations may still be high
- very deep trees are not automatically justified
- regularization and sampling parameters are likely important

### 3. Season distribution is uneven

Some seasons contribute far more rows than others.

Implications:

- aggregate validation may hide season-specific weaknesses
- the tuning workflow should log at least some per-season diagnostics
- this does not change the optimization target, but it should inform analysis

### 4. Native categorical handling should not be mixed into the first cleanup by default

If the runtime environment for native XGBoost categorical handling is not fully
stable and pinned, the first refactor should prefer a deterministic encoded
categorical path.

Native categorical support can be tested later as a separate experiment if
desired.

### 5. Fit-time custom metrics should be used carefully

If the early-stopping metric is a built-in metric such as `aucpr` or `logloss`,
then custom Python metrics like `f1`, `precision`, and `recall` should not be
allowed to add avoidable per-iteration overhead unless there is a clear reason.

They can be computed after fit for reporting.

## Sensible first hyperparameter ranges

The updated tuning script should start with practical ranges that match the new
feature space and imbalance characteristics.

Recommended first-pass ranges:

1. `learning_rate`: `0.01` to `0.15`
2. `n_estimators`: `1500` to `5000` with early stopping
3. `grow_policy`: `depthwise` or `lossguide`
4. `max_depth`: `4` to `10` when using `depthwise`
5. `max_leaves`: `31` to `511` when using `lossguide`
6. `min_child_weight`: `1` to `64`
7. `gamma`: `0` to `10`
8. `subsample`: `0.5` to `1.0`
9. `colsample_bytree`: `0.4` to `1.0`
10. `reg_alpha`: `1e-3` to `5`
11. `reg_lambda`: `0.5` to `20`
12. `scale_pos_weight`: broad range around empirical imbalance, e.g. `10` to `120`
13. `max_delta_step`: `0` to `10`
14. `max_bin`: keep fixed first, or search a narrow range later if memory allows

## Expected deliverables from the refactor

1. A persisted engineered-data pipeline under `data/feat_engineered_vaep_data`
2. A fixed `train_xgboost.py` threshold-sweep path
3. A refactored source-first / lazy-rest split loader
4. Shared low-precision feature casting helpers used by all XGBoost scripts
5. Clear in-domain-only objective language in configs and comments
6. A concrete implementation plan for an updated tuning script

## Recommended implementation order

1. Fix the `y_proba_val` regression
2. Extract and centralize feature engineering helpers
3. Add engineered-data persistence under `data/feat_engineered_vaep_data`
4. Add shared dtype-normalization helpers
5. Refactor `load_xy_competition_season_split(...)` to source-first / lazy-rest
6. Update `train_xgboost.py` to consume the engineered dataset
7. Update the remaining XGBoost scripts to use the same feature schema and dtype policy
8. Write the updated tuning-script implementation plan and config search ranges

## Test expectations

Add or update tests that cover:

1. the threshold-sweep regression
2. feature-engineered dataset creation on a tiny sample
3. dtype normalization for XGBoost feature frames
4. lazy split materialization behavior
5. consistency between persisted engineered features and the expected training schema
