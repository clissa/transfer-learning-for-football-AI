from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.tune_xgboost as tune_script
from football_ai.config import load_config
from football_ai.training import ResolvedSplitFrame
from football_ai import tuning


def _make_split_frame(
    name: str,
    *,
    index: list[int],
    labels: list[int],
    scores: list[float],
    competition: str = "La Liga",
    season: str = "2015/2016",
) -> ResolvedSplitFrame:
    X = pd.DataFrame({"score_hint": scores}, index=index, dtype=np.float32)
    y = pd.Series(labels, index=index, name="scores", dtype=np.uint8)
    df = pd.DataFrame(
        {
            "game_id": index,
            "action_id": list(range(len(index))),
            "competition_name": [competition] * len(index),
            "season_name": [season] * len(index),
            "season_id": [101] * len(index),
            "scores": labels,
            "concedes": [0] * len(index),
        },
        index=index,
    )
    coverage = df[["competition_name", "season_name"]].drop_duplicates().reset_index(drop=True)
    return ResolvedSplitFrame(name=name, df=df, X=X, y=y, competition_seasons=coverage)


class _FakeResolvedSplit:
    def __init__(self) -> None:
        self.source_train = _make_split_frame(
            "source_train",
            index=[1, 2, 3, 4],
            labels=[0, 1, 0, 1],
            scores=[0.05, 0.95, 0.15, 0.85],
        )
        self.source_val = _make_split_frame(
            "source_val",
            index=[10, 11, 12, 13],
            labels=[0, 1, 0, 1],
            scores=[0.1, 0.9, 0.2, 0.8],
        )
        self.feature_cols = ["score_hint"]
        self.split_config = {}
        self.materialized: set[str] = set()

    @property
    def calib(self) -> ResolvedSplitFrame:
        self.materialized.add("calib")
        return _make_split_frame(
            "calib",
            index=[20, 21],
            labels=[0, 1],
            scores=[0.3, 0.7],
            competition="Champions League",
        )

    @property
    def target(self) -> ResolvedSplitFrame:
        self.materialized.add("target")
        return _make_split_frame(
            "target",
            index=[30, 31],
            labels=[1, 0],
            scores=[0.75, 0.25],
            competition="Serie A",
        )

    @property
    def test_names(self) -> list[str]:
        return ["league_shift"]

    def get_test_split(self, name: str) -> ResolvedSplitFrame:
        self.materialized.add(f"test_{name}")
        return _make_split_frame(
            f"test_{name}",
            index=[40, 41],
            labels=[0, 1],
            scores=[0.35, 0.65],
            competition="Premier League",
        )

    def iter_test_splits(self) -> list[tuple[str, ResolvedSplitFrame]]:
        return [(name, self.get_test_split(name)) for name in self.test_names]

    def named_splits(self, include_lazy: bool = False) -> dict[str, ResolvedSplitFrame]:
        splits = {
            "source_train": self.source_train,
            "source_val": self.source_val,
        }
        if include_lazy:
            splits["calib"] = self.calib
            for name, split in self.iter_test_splits():
                splits[f"test_{name}"] = split
            splits["target"] = self.target
        return splits

    def is_materialized(self, name: str) -> bool:
        if name in {"source_train", "source_val"}:
            return True
        return name in self.materialized


class _FakeTrial:
    def __init__(self, number: int = 0, grow_policy: str = "depthwise") -> None:
        self.number = number
        self.grow_policy = grow_policy
        self.params: dict[str, Any] = {}
        self.user_attrs: dict[str, Any] = {}
        self.suggested_ints: dict[str, tuple[int, int, bool]] = {}

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        self.suggested_ints[name] = (low, high, log)
        value = low
        if name == "max_leaves":
            value = high
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        value = float(low)
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        value = self.grow_policy if name == "grow_policy" else choices[0]
        self.params[name] = value
        return value

    def set_user_attr(self, name: str, value: Any) -> None:
        self.user_attrs[name] = value


class _DummyXGBClassifier:
    fit_calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.best_score = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "_DummyXGBClassifier":
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.fit_kwargs = kwargs
        _DummyXGBClassifier.fit_calls.append({"X_index": list(X.index), "kwargs": kwargs})
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["score_hint"].to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])


class _FakeStudy:
    def __init__(self) -> None:
        self.best_trial: _FakeTrial | None = None
        self.best_value: float | None = None
        self.trials: list[_FakeTrial] = []

    def optimize(self, objective, n_trials: int, timeout: int | None, n_jobs: int) -> None:
        trial = _FakeTrial(number=0)
        value = objective(trial)
        trial.value = value
        self.trials.append(trial)
        self.best_trial = trial
        self.best_value = value

    def trials_dataframe(self, attrs=None) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params_n_estimators": trial.params["n_estimators"],
                    "user_attrs_source_val_roc_auc": trial.user_attrs["source_val_roc_auc"],
                }
                for trial in self.trials
            ]
        )


class _FakeOptuna:
    class samplers:
        class TPESampler:
            def __init__(self, seed: int) -> None:
                self.seed = seed

    def __init__(self) -> None:
        self.study = _FakeStudy()
        self.study_name: str | None = None
        self.direction: str | None = None

    def create_study(self, study_name: str, direction: str, sampler: Any) -> _FakeStudy:
        assert direction in {"maximize", "minimize"}
        self.study_name = study_name
        self.direction = direction
        return self.study


def _base_tuning_cfg(output_root: Path, seed: int | None = 123) -> dict[str, Any]:
    return {
        "data": {
            "file": "unused.h5",
            "key_candidates": ["feat_engineered_vaep_data", "vaep_data"],
            "target_col": "scores",
        },
        "selected_metric": "roc_auc",
        "seed": seed,
        "device": "cpu",
        "split": {
            "validation_frac": 0.5,
            "source": {"competitions": ["La Liga"]},
            "calib": {"competitions": ["Champions League"]},
            "test": {"competitions": ["La Liga"], "year_shift": {"seasons": ["2019-20"]}},
        },
        "optuna": {"n_trials": 1, "timeout_seconds": None},
        "training": {"early_stopping_rounds": 3},
        "model": {
            "base_params": {
                "booster": "gbtree",
                "tree_method": "hist",
                "n_jobs": 1,
                "verbosity": 0,
                "validate_parameters": True,
                "importance_type": "gain",
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "enable_categorical": True,
            }
        },
        "search_space": {
            "n_estimators": {"low": 100, "high": 5000, "log": True},
            "learning_rate": {"low": 0.01, "high": 0.15, "log": True},
            "grow_policy": ["depthwise", "lossguide"],
            "max_depth": {"low": 4, "high": 10},
            "max_leaves": {"low": 31, "high": 511},
            "min_child_weight": {"low": 1, "high": 64, "log": True},
            "gamma": {"low": 0, "high": 10},
            "subsample": {"low": 0.5, "high": 1.0},
            "colsample_bytree": {"low": 0.4, "high": 1.0},
            "reg_alpha": {"low": 0.001, "high": 5, "log": True},
            "reg_lambda": {"low": 0.5, "high": 20, "log": True},
            "scale_pos_weight": {"low": 10, "high": 120},
            "max_delta_step": {"low": 0, "high": 10},
            "max_bin": 256,
        },
        "threshold": {"min": 0.1, "max": 0.9, "steps": 5},
        "output": {"root": str(output_root)},
    }


def test_tune_script_delegates_to_library_with_yaml_and_cli_overrides(monkeypatch, tmp_path):
    cfg = _base_tuning_cfg(tmp_path / "results")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        tune_script,
        "parse_args",
        lambda: Namespace(
            config="configs/tune_xgboost.yaml",
            target_col="concedes",
            data_file="override.h5",
            output_dir=str(tmp_path / "override_results"),
            n_trials=7,
            timeout=11,
            seed=456,
            device="cpu",
        ),
    )
    monkeypatch.setattr(tune_script, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tune_script, "load_config", lambda path: cfg)

    def fake_run_xgboost_tuning(loaded_cfg, *, cli_overrides):
        captured["cfg"] = loaded_cfg
        captured["cli_overrides"] = cli_overrides
        return tuning.XGBoostTuningResult(
            run_dir=tmp_path / "override_results" / "run",
            best_trial_number=3,
            best_score=0.91,
            best_params={},
            manifest_path=tmp_path / "manifest.json",
        )

    monkeypatch.setattr(tune_script, "run_xgboost_tuning", fake_run_xgboost_tuning)

    assert tune_script.main() == 0
    assert captured["cfg"] is cfg
    assert captured["cli_overrides"] == {
        "target_col": "concedes",
        "data_file": "override.h5",
        "output_dir": str(tmp_path / "override_results"),
        "n_trials": 7,
        "timeout_seconds": 11,
        "seed": 456,
        "device": "cpu",
    }


def test_default_tune_config_uses_cpu_device():
    cfg = load_config(tuning.DEFAULT_CONFIG_PATH)
    resolved = tuning.resolve_xgboost_tuning_config(cfg)

    assert resolved["device"] == "cpu"
    assert resolved["model"]["base_params"]["device"] == "cpu"


def test_tune_config_accepts_custom_selected_metric(tmp_path):
    cfg = _base_tuning_cfg(tmp_path / "results")
    cfg["selected_metric"] = "f1"

    resolved = tuning.resolve_xgboost_tuning_config(cfg)

    assert resolved["selected_metric"] == "f1"
    assert resolved["selected_metric_direction"] == "maximize"


def test_tune_config_sets_minimize_direction_for_loss_metric(tmp_path):
    cfg = _base_tuning_cfg(tmp_path / "results")
    cfg["selected_metric"] = "logloss"

    resolved = tuning.resolve_xgboost_tuning_config(cfg)

    assert resolved["selected_metric"] == "logloss"
    assert resolved["selected_metric_direction"] == "minimize"


def test_sample_search_space_uses_requested_n_estimators_range():
    trial = _FakeTrial()
    params = tuning.sample_xgboost_tuning_params(
        trial,
        base_params={"objective": "binary:logistic", "eval_metric": "auc"},
        search_space=_base_tuning_cfg(Path("unused"))["search_space"],
    )

    assert trial.suggested_ints["n_estimators"] == (100, 5000, True)
    assert params["n_estimators"] == 100
    assert params["grow_policy"] == "depthwise"
    assert params["max_depth"] == 4


def test_objective_uses_only_source_splits_and_returns_selected_metric(monkeypatch, tmp_path):
    fake_split = _FakeResolvedSplit()
    fake_split.source_val.X.loc[10, "score_hint"] = np.float32(0.6)
    monkeypatch.setattr(tuning, "XGBClassifier", _DummyXGBClassifier)
    _DummyXGBClassifier.fit_calls = []

    objective = tuning.build_xgboost_tuning_objective(
        resolved_split=fake_split,
        base_params={"objective": "binary:logistic", "eval_metric": "auc"},
        search_space=_base_tuning_cfg(tmp_path)["search_space"],
        run_dir=tmp_path,
        source_train_empirical_spw=1.0,
        selected_metric="f1",
    )
    trial = _FakeTrial()

    value = objective(trial)

    assert np.isclose(value, 0.8)
    assert trial.user_attrs["source_val_roc_auc"] == 1.0
    assert np.isclose(trial.user_attrs["source_val_f1"], 0.8)
    assert not fake_split.is_materialized("calib")
    assert not fake_split.is_materialized("target")
    assert not fake_split.is_materialized("test_league_shift")
    assert _DummyXGBClassifier.fit_calls[-1]["X_index"] == [1, 2, 3, 4]
    assert (tmp_path / "trial_0_source_val_metrics.json").exists()
    assert (tmp_path / "trial_0_params.json").exists()
    assert (tmp_path / "trial_0_la_liga_val_by_season.csv").exists()


def test_run_tuning_writes_timestamped_artifacts_and_post_study_diagnostics(monkeypatch, tmp_path):
    fake_split = _FakeResolvedSplit()
    fake_optuna = _FakeOptuna()
    monkeypatch.setattr(tuning, "optuna", fake_optuna)
    monkeypatch.setattr(tuning, "XGBClassifier", _DummyXGBClassifier)
    monkeypatch.setattr(tuning, "load_xy_competition_season_split", lambda **_kwargs: fake_split)
    monkeypatch.setattr(tuning, "save_model", lambda model, path: Path(path).write_text("model", encoding="utf-8"))
    _DummyXGBClassifier.fit_calls = []

    result = tuning.run_xgboost_tuning(
        _base_tuning_cfg(tmp_path / "results", seed=123),
        run_timestamp=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert result.run_dir == tmp_path / "results" / "run_20260102_030405_seed123"
    expected_files = {
        "effective_config.yaml",
        "run_manifest.json",
        "trials.csv",
        "trial_0_source_val_metrics.json",
        "trial_0_params.json",
        "trial_0_la_liga_val_by_season.csv",
        "best_params.json",
        "best_model.pkl",
        "best_threshold_sweep_source_val.csv",
        "final_diagnostics_by_split.csv",
        "coverage_source_train.csv",
        "coverage_source_val.csv",
        "coverage_calib.csv",
        "coverage_test_league_shift.csv",
        "coverage_target.csv",
        "feature_columns.json",
    }
    assert expected_files.issubset({path.name for path in result.run_dir.iterdir()})
    assert fake_split.is_materialized("calib")
    assert fake_split.is_materialized("test_league_shift")
    assert fake_split.is_materialized("target")

    manifest = pd.read_json(result.manifest_path, typ="series").to_dict()
    assert manifest["selected_metric"] == "roc_auc"
    assert manifest["selected_metric_direction"] == "maximize"
    assert manifest["best_source_val_metric"] == 1.0
    assert manifest["best_source_val_roc_auc"] == 1.0
    assert manifest["seed"] == 123
    assert manifest["best_trial_number"] == 0


def test_run_tuning_uses_configured_selected_metric(monkeypatch, tmp_path):
    fake_split = _FakeResolvedSplit()
    fake_optuna = _FakeOptuna()
    monkeypatch.setattr(tuning, "optuna", fake_optuna)
    monkeypatch.setattr(tuning, "XGBClassifier", _DummyXGBClassifier)
    monkeypatch.setattr(tuning, "load_xy_competition_season_split", lambda **_kwargs: fake_split)
    monkeypatch.setattr(tuning, "save_model", lambda model, path: Path(path).write_text("model", encoding="utf-8"))
    _DummyXGBClassifier.fit_calls = []
    cfg = _base_tuning_cfg(tmp_path / "results", seed=123)
    cfg["selected_metric"] = "f1"

    result = tuning.run_xgboost_tuning(
        cfg,
        run_timestamp=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert result.selected_metric == "f1"
    assert fake_optuna.direction == "maximize"
    assert fake_optuna.study_name == "xgb_f1_scores_20260102_030405_seed123"

    manifest = pd.read_json(result.manifest_path, typ="series").to_dict()
    assert manifest["selected_metric"] == "f1"
    assert manifest["selected_metric_direction"] == "maximize"
    assert manifest["best_source_val_metric"] == 1.0
    assert manifest["best_source_val_f1"] == 1.0


def test_null_seed_is_resolved_and_tracked(monkeypatch, tmp_path):
    monkeypatch.setattr(tuning, "resolve_random_state", lambda *_args: 987654)
    cfg = tuning.resolve_xgboost_tuning_config(_base_tuning_cfg(tmp_path, seed=None))

    assert cfg["seed"] == 987654
    assert cfg["model"]["base_params"]["random_state"] == 987654
