# Refactor Input Features Handoff

## Goal of this note

This note captures the intended scope of the next XGBoost feature refactor for the `scores` target.

The goal is not to keep extending the current hard-coded whitelist with extra columns.

The goal is to reshape the feature space in three specific ways:

1. Substitute one-hot categorical blocks with single multi-category features and let XGBoost handle those unordered categories properly.
2. Add explicit geometry features derived from start and end coordinates.
3. Add possession-continuity and transition features relevant to predicting `scores`.

## Current setup

The current XGBoost training flow uses a curated VAEP feature whitelist from `src/football_ai/training.py`.

That whitelist currently includes:

- one-hot action types for `a0`, `a1`, `a2`
- one-hot results for `a0`, `a1`, `a2`
- one-hot body parts for `a0`, `a1`, `a2`
- start and end coordinates
- movement and delta features
- score-state features
- timing features

## Planned refactor direction

### 1. Replace one-hot blocks with single categorical features

For each action-state slot (`a0`, `a1`, `a2`):

1. Replace `actiontype_*` one-hot columns with one feature such as `actiontype_a0`.
2. Replace `result_*` one-hot columns with one feature such as `result_a0`.
3. Replace `bodypart_*` one-hot columns with one feature such as `bodypart_a0`.

The intent is to reduce dimensionality and avoid representing mutually exclusive categories as many separate columns.

### 2. Add explicit geometry features

Recommended first geometry batch:

1. `start_dist_to_goal`, `end_dist_to_goal`
2. `start_angle_to_goal`, `end_angle_to_goal`
3. `in_final_third`, `start_in_box`, `end_in_box`
4. `dist_to_goal_delta`

These should be added on top of start/end `x` and `y`, not as a replacement in the first ablation.

### 3. Add possession-continuity and transition features

Recommended first possession-continuity batch:

1. `same_team_a01`
2. `same_team_a12`
3. `same_team_a02`
4. `turnover_a01`
5. `turnover_a12`
6. `possession_chain_len`

Recommended first transition batch:

1. turnover indicators between recent actions
2. features marking whether the current action follows a possession change

These are intended specifically for the `scores` target, where recent attacking continuity and transitions may add signal beyond raw action history alone.

## Minimal mental model to carry forward

This refactor is not "add more features to the existing whitelist".

It is:

- compress categorical representation
- add explicit geometry
- add possession / transition context for `scores`
