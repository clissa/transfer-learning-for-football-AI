# Refactor Splitting Handoff

## Goal of this note

This document captures the current train/validation/test splitting behavior reconstructed from the codebase and local datasets, so a later refactoring chat can start with the same context.

## Current splitting behavior

The main XGBoost training script is `scripts/train_xgboost.py`.

It does not perform a temporal split by season.

Instead, it:

1. Resolves three league groups from defaults or YAML config:
   - `source_competitions`
   - `calib_competitions`
   - `target_competitions`
2. Loads one merged HDF5 table from `data/vaep_data/*.h5`
3. Filters rows by `competition_name`
4. Splits only the `source` subset into `train` and `validation`
5. Keeps `calib` untouched
6. Keeps `target` untouched for zero-shot evaluation

Implementation is in `src/football_ai/training.py::load_xy_source_calib_target_split(...)`.

## What train / val / calib / target actually mean

- `train`:
  Rows from `source_competitions`, restricted to matches (`game_id`) sampled into the training fold.
- `validation`:
  Rows from the same `source_competitions`, restricted to matches sampled into the validation fold.
- `calib`:
  All rows from `calib_competitions`. Reserved, not used for model fitting in the current script.
- `target`:
  All rows from `target_competitions`. Used as the held-out zero-shot test set.

Important consequence:

- Train and validation use the same leagues.
- Train and validation also use the same seasons for those leagues.
- The split boundary is match-level, not season-level and not league-level.

## Exact mechanics of the source split

Inside `load_xy_source_calib_target_split(...)`:

- `df_source`, `df_calib`, and `df_target` are built with `df["competition_name"].isin(...)`
- `df_source` is reduced to one row per `game_id`
- `train_test_split(...)` is applied to those matches with:
  - `test_size=validation_frac`
  - `stratify=source_matches["competition_name"]`
  - `random_state=random_state`
- The resulting `train_game_ids` and `val_game_ids` are used to recover all action rows for each match

So the current validation set is:

- 20% of source matches by default
- stratified by league
- mixed across whatever seasons exist inside each source league

## Important limitation

There is no explicit season-aware splitting.

If a source competition contains multiple seasons, actions from older and newer seasons can both land in train and validation, as long as they belong to different matches.

## Config-dependent behavior

There are three relevant ways `train_xgboost.py` can be run:

### 1. No config file

Hardcoded defaults in `scripts/train_xgboost.py`:

- Source:
  - `Premier League`
  - `La Liga`
- Calib:
  - `Serie A`
  - `Ligue 1`
- Target:
  - `Champions League`
  - `UEFA Europa League`

### 2. `configs/train_xgboost.yaml`

Configured split:

- Source:
  - `Premier League`
  - `La Liga`
  - `1. Bundeliga`
- Calib:
  - `Champions League`
  - `UEFA Europa League`
- Target:
  - `Serie A`
  - `Ligue 1`

### 3. `configs/train_xgboost_full.yaml`

Configured split:

- Source:
  - `Premier League`
  - `La Liga`
  - `Copa del Rey`
  - `1. Bundeliga`
  - `Serie A`
  - `Ligue 1`
  - `Champions League`
  - `UEFA Europa League`
- Calib:
  - `FA Women's Super League`
  - `Women's World Cup`
  - `UEFA Women's Euro`
- Target:
  - `FIFA World Cup`
  - `UEFA Euro`
  - `FIFA U20 World Cup`
  - `Copa America`
  - `Liga Profesional`
  - `African Cup of Nations`
  - `Indian Super league`
  - `Major League Soccer`
  - `NWSL`
  - `North American League`

## Bug / inconsistency already identified

Both XGBoost YAML configs contain a typo:

- `1. Bundeliga`

But the dataset uses:

- `1. Bundesliga`

Effect:

- Bundesliga is intended to be part of source in those configs
- but it is not actually selected by the `isin(...)` filter
- so the effective source set is smaller than the config suggests

This is easy to miss because the split helper currently returns `sorted(source_set)`, `sorted(calib_set)`, and `sorted(target_set)` from the requested config values, not from the competitions actually found in the filtered data.

That means logs can report Bundesliga as "used" even when no rows were selected for it.

## Season coverage reconstructed from local data

The seasons below come from the local `competitions` tables in:

- `data/spadl_full_data/major_leagues.h5`
- `data/spadl_full_data/all_leagues.h5`

These are the practical season sets that feed the corresponding VAEP files.

### Coverage for `configs/train_xgboost.yaml`

Data file:

- `data/vaep_data/major_leagues_vaep.h5`

Actual source seasons:

- `Premier League`
  - `2003/2004`
  - `2015/2016`
- `La Liga`
  - `1973/1974`
  - `2004/2005`
  - `2005/2006`
  - `2006/2007`
  - `2007/2008`
  - `2008/2009`
  - `2009/2010`
  - `2010/2011`
  - `2011/2012`
  - `2012/2013`
  - `2013/2014`
  - `2014/2015`
  - `2015/2016`
  - `2016/2017`
  - `2017/2018`
  - `2018/2019`
  - `2019/2020`
  - `2020/2021`
- `1. Bundesliga`
  - `2015/2016`
  - `2023/2024`
  - intended in config, but currently dropped because of the typo

Actual calib seasons:

- `Champions League`
  - `1970/1971`
  - `1971/1972`
  - `1972/1973`
  - `1999/2000`
  - `2003/2004`
  - `2004/2005`
  - `2006/2007`
  - `2008/2009`
  - `2009/2010`
  - `2010/2011`
  - `2011/2012`
  - `2012/2013`
  - `2013/2014`
  - `2014/2015`
  - `2015/2016`
  - `2016/2017`
  - `2017/2018`
  - `2018/2019`
- `UEFA Europa League`
  - `1988/1989`

Actual target seasons:

- `Serie A`
  - `1986/1987`
  - `2015/2016`
- `Ligue 1`
  - `2015/2016`
  - `2021/2022`
  - `2022/2023`

### Coverage for `configs/train_xgboost_full.yaml`

Data file:

- `data/vaep_data/all_leagues_vaep.h5`

Actual source seasons:

- `Premier League`
  - `2003/2004`
  - `2015/2016`
- `La Liga`
  - `1973/1974`
  - `2004/2005`
  - `2005/2006`
  - `2006/2007`
  - `2007/2008`
  - `2008/2009`
  - `2009/2010`
  - `2010/2011`
  - `2011/2012`
  - `2012/2013`
  - `2013/2014`
  - `2014/2015`
  - `2015/2016`
  - `2016/2017`
  - `2017/2018`
  - `2018/2019`
  - `2019/2020`
  - `2020/2021`
- `Copa del Rey`
  - `1977/1978`
  - `1982/1983`
  - `1983/1984`
- `1. Bundesliga`
  - `2015/2016`
  - `2023/2024`
  - intended in config, but currently dropped because of the typo
- `Serie A`
  - `1986/1987`
  - `2015/2016`
- `Ligue 1`
  - `2015/2016`
  - `2021/2022`
  - `2022/2023`
- `Champions League`
  - `1970/1971`
  - `1971/1972`
  - `1972/1973`
  - `1999/2000`
  - `2003/2004`
  - `2004/2005`
  - `2006/2007`
  - `2008/2009`
  - `2009/2010`
  - `2010/2011`
  - `2011/2012`
  - `2012/2013`
  - `2013/2014`
  - `2014/2015`
  - `2015/2016`
  - `2016/2017`
  - `2017/2018`
  - `2018/2019`
- `UEFA Europa League`
  - `1988/1989`

Actual calib seasons:

- `FA Women's Super League`
  - `2018/2019`
  - `2019/2020`
  - `2020/2021`
- `Women's World Cup`
  - `2019`
  - `2023`
- `UEFA Women's Euro`
  - `2022`
  - `2025`

Actual target seasons:

- `FIFA World Cup`
  - `1958`
  - `1962`
  - `1970`
  - `1974`
  - `1986`
  - `1990`
  - `2018`
  - `2022`
- `UEFA Euro`
  - `2020`
  - `2024`
- `FIFA U20 World Cup`
  - `1979`
- `Copa America`
  - `2024`
- `Liga Profesional`
  - `1981`
  - `1997/1998`
- `African Cup of Nations`
  - `2023`
- `Indian Super league`
  - `2021/2022`
- `Major League Soccer`
  - `2023`
- `NWSL`
  - `2018`
- `North American League`
  - `1977`

## Refactor-relevant pain points

These are the main issues worth addressing in a splitting refactor:

1. Split semantics are league-based plus match-based, not season-based.
2. Train and validation can mix seasons from the same competition.
3. Requested competition names are not validated strictly enough.
4. Typos in config silently reduce the effective dataset.
5. Returned and logged competition lists reflect requested config names, not actual selected leagues in the data.
6. Calibration is loaded but not used by the current training flow.
7. There is no single abstraction describing split policy independently of the training script.

## Likely refactor directions

Possible improvements to discuss in the next chat:

1. Make split policy explicit:
   - by competition
   - by season
   - by competition-season
   - by match within competition
2. Validate configured competition names against the dataset and fail on missing names.
3. Return both:
   - requested split names
   - resolved split names actually present in data
4. Optionally log exact competition-season coverage per split.
5. Decide whether `calib` should remain separate, become optional, or be folded into a more general split configuration.
6. Fix the Bundesliga typo in XGBoost configs as part of cleanup.

## Minimal mental model to carry forward

The current `train_xgboost.py` split is:

- first: league filtering into source / calib / target
- then: match-level random split only inside source
- not: season holdout
- not: competition-season holdout
- not: chronological split

That is the baseline behavior any refactor should either preserve deliberately or replace explicitly.

## Todo List

- [ ] Splitting follow-up: revisit split policy semantics captured in [refactor_splitting.md](/eos/home-l/luclissa/work/transfer-learning-for-football-AI/.github/tasks/refactor_splitting.md)
- [ ] Input-feature follow-up: review [refactor_input_features.md](/eos/home-l/luclissa/work/transfer-learning-for-football-AI/.github/tasks/refactor_input_features.md) for the XGBoost feature refactor plan, including one-hot decoding, dimensionality reduction, and candidate higher-signal football features
