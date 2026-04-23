# Updated Tuning Script Follow-up Plan

## Scope

This follow-up is for the next rewrite of the XGBoost tuning workflow after the
engineered `scores` dataset refactor.

The objective remains explicit:

- source domain: `La Liga`
- training split: source train by `game_id`
- model selection: source validation only
- non-source leagues: diagnostics only, not trial ranking

## Inputs and split contract

1. Read from `data/feat_engineered_vaep_data/major_leagues_vaep.h5`.
2. Use `key_candidates = ["feat_engineered_vaep_data", "vaep_data"]`.
3. Reuse `load_xy_competition_season_split(...)`.
4. Materialize only `source_train` and `source_val` inside the objective.
5. Keep `calib`, `test_*`, and `target` for post-study reporting only.

## Trial objective

1. Fit `XGBClassifier` with `objective="binary:logistic"`.
2. Use `eval_metric="aucpr"` for early stopping.
3. Rank trials by source-validation `pr_auc`.
4. Record source-validation `logloss`, `brier`, `roc_auc`, and threshold-swept `f1` as secondary diagnostics.
5. Treat threshold selection as post-fit evaluation, never as the stopping metric.

## Feature and dtype assumptions

1. Use the persisted compact categorical `scores` schema only.
2. Reuse `normalize_xgb_feature_frame(...)` and `normalize_xgb_labels(...)`.
3. Keep the encoded categorical path for now; do not mix in native XGBoost categorical handling in the first tuning rewrite.

## First search space

Use early stopping with a large `n_estimators` ceiling and search the following:

1. `learning_rate`: `0.01` to `0.15` on a log scale.
2. `n_estimators`: fixed at `1500` to `5000` depending on runtime budget, with early stopping.
3. `grow_policy`: `["depthwise", "lossguide"]`.
4. `max_depth`: `4` to `10` when `grow_policy="depthwise"`.
5. `max_leaves`: `31` to `511` when `grow_policy="lossguide"`.
6. `min_child_weight`: `1` to `64` on a log scale.
7. `gamma`: `0` to `10`.
8. `subsample`: `0.5` to `1.0`.
9. `colsample_bytree`: `0.4` to `1.0`.
10. `reg_alpha`: `1e-3` to `5` on a log scale.
11. `reg_lambda`: `0.5` to `20` on a log scale.
12. `scale_pos_weight`: `10` to `120`, centered around the empirical source-train imbalance.
13. `max_delta_step`: `0` to `10`.
14. `max_bin`: keep fixed initially; only open a narrow search later if memory allows.

## Study outputs

Each trial should save:

1. source-validation metrics
2. selected validation threshold
3. best iteration / early-stopping round
4. positive rate in train and validation
5. per-season validation summary for La Liga
6. the exact feature column list and data file used

After the study finishes, run the best model once on:

- `source_train`
- `source_val`
- each `test_*` split
- `target`

Those non-source evaluations should be exported for later transfer-learning work, but they should not change the chosen trial.
