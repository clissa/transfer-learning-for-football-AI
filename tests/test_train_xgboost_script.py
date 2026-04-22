from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd

import scripts.train_xgboost as train_xgboost
from football_ai.training import ResolvedSplitFrame


def _make_split_frame(
    name: str,
    *,
    index: list[int],
    game_ids: list[int],
    labels: list[int],
    competition: str,
) -> ResolvedSplitFrame:
    X = pd.DataFrame({"feat": np.linspace(0.1, 0.9, num=len(index), dtype=np.float32)}, index=index)
    y = pd.Series(labels, index=index, name="scores", dtype=np.uint8)
    df = pd.DataFrame(
        {
            "game_id": game_ids,
            "action_id": list(range(len(index))),
            "competition_name": [competition] * len(index),
            "season_name": ["2015/2016"] * len(index),
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
            index=[1, 2, 3],
            game_ids=[1001, 1002, 1003],
            labels=[0, 1, 0],
            competition="La Liga",
        )
        self.source_val = _make_split_frame(
            "source_val",
            index=[10, 11],
            game_ids=[1004, 1005],
            labels=[0, 1],
            competition="La Liga",
        )
        self._calib = _make_split_frame(
            "calib",
            index=[20],
            game_ids=[2001],
            labels=[0],
            competition="Champions League",
        )
        self._target = _make_split_frame(
            "target",
            index=[30],
            game_ids=[3001],
            labels=[1],
            competition="Serie A",
        )
        self._tests = {
            "league_shift": _make_split_frame(
                "test_league_shift",
                index=[40],
                game_ids=[4001],
                labels=[0],
                competition="Premier League",
            )
        }

    @property
    def calib(self) -> ResolvedSplitFrame:
        return self._calib

    @property
    def target(self) -> ResolvedSplitFrame:
        return self._target

    @property
    def test_names(self) -> list[str]:
        return sorted(self._tests)

    def get_test_split(self, name: str) -> ResolvedSplitFrame:
        return self._tests[name]

    def iter_test_splits(self) -> list[tuple[str, ResolvedSplitFrame]]:
        return [(name, self._tests[name]) for name in self.test_names]

    def named_splits(self, include_lazy: bool = False) -> dict[str, ResolvedSplitFrame]:
        splits = {
            "source_train": self.source_train,
            "source_val": self.source_val,
        }
        if include_lazy:
            splits["calib"] = self.calib
            splits["target"] = self.target
            for name, split in self.iter_test_splits():
                splits[f"test_{name}"] = split
        return splits


class _DummyXGBClassifier:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.best_score = 0.9123

    def fit(self, X, y, **kwargs):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.fit_kwargs = kwargs
        return self


def test_train_xgboost_threshold_sweep_uses_validation_probabilities(monkeypatch, tmp_path):
    fake_split = _FakeResolvedSplit()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        train_xgboost,
        "parse_args",
        lambda: Namespace(
            config=None,
            target_col=None,
            data_file=None,
            output_dir=str(tmp_path / "results"),
            seed=123,
            device="cpu",
        ),
    )
    monkeypatch.setattr(train_xgboost, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_xgboost,
        "load_xy_competition_season_split",
        lambda **_kwargs: fake_split,
    )
    monkeypatch.setattr(train_xgboost, "XGBClassifier", _DummyXGBClassifier)
    monkeypatch.setattr(train_xgboost, "save_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_xgboost, "plot_confusion_matrix", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_xgboost, "print_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_xgboost,
        "evaluate_binary_with_baselines",
        lambda y_proba, y_true, threshold: {
            "rows": float(len(y_true)),
            "positive_rate": float(np.mean(y_true)),
            "f1": float(threshold),
        },
    )
    monkeypatch.setattr(train_xgboost, "MODELS_PATH", tmp_path / "models")

    def fake_get_positive_class_scores(model, X):
        if list(X.index) == [10, 11]:
            scores = np.array([0.2, 0.8], dtype=np.float32)
            captured["validation_scores"] = scores
            return scores
        return np.full(len(X), 0.4, dtype=np.float32)

    def fake_sweep_thresholds_for_f1(y_true, y_score, threshold_min, threshold_max, threshold_steps):
        captured["sweep_y_true"] = list(np.asarray(y_true))
        captured["sweep_y_score"] = list(np.asarray(y_score))
        return pd.DataFrame({"threshold": [0.45], "f1": [1.0]}), 0.45

    monkeypatch.setattr(train_xgboost, "get_positive_class_scores", fake_get_positive_class_scores)
    monkeypatch.setattr(train_xgboost, "sweep_thresholds_for_f1", fake_sweep_thresholds_for_f1)

    assert train_xgboost.main() == 0
    assert captured["sweep_y_true"] == [0, 1]
    assert captured["sweep_y_score"] == [0.2, 0.8]
