"""Smoke tests for the VAEP feature extraction pipeline.

Creates synthetic SPADL-like DataFrames (2 games, ~10 actions each) and
verifies that ``orient_actions``, ``compute_vaep_features``, and
``build_vaep_dataset`` produce correct outputs.

CPU-only, target < 10 seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_ai.features import (
    DEFAULT_FEATURE_FUNCTIONS,
    LABEL_COLS,
    SPADL_ACTION_COLS,
    build_vaep_dataset,
    compute_vaep_features,
    orient_actions,
    save_vaep_dataset,
)


# ─────────────────────────────────────
# Fixtures
# ─────────────────────────────────────

NB_PREV_ACTIONS = 3


def _make_synthetic_actions(n_actions: int = 10, game_id: int = 0) -> pd.DataFrame:
    """Return a synthetic SPADL-like actions DataFrame for one game."""
    rng = np.random.RandomState(42 + game_id)
    return pd.DataFrame(
        {
            "game_id": game_id,
            "action_id": np.arange(n_actions),
            "period_id": 1,
            "time_seconds": np.linspace(0, 45 * 60, n_actions),
            "team_id": rng.choice([100, 200], size=n_actions),
            "player_id": rng.choice([10, 11, 12, 20, 21, 22], size=n_actions),
            "start_x": rng.uniform(0, 105, n_actions),
            "start_y": rng.uniform(0, 68, n_actions),
            "end_x": rng.uniform(0, 105, n_actions),
            "end_y": rng.uniform(0, 68, n_actions),
            "type_id": rng.randint(0, 5, n_actions),  # pass/cross/throw_in/fk
            "result_id": rng.randint(0, 2, n_actions),  # fail/success
            "bodypart_id": rng.randint(0, 3, n_actions),  # foot/head/other
        }
    )


@pytest.fixture
def two_game_actions() -> pd.DataFrame:
    """Two synthetic games with 10 actions each."""
    g0 = _make_synthetic_actions(n_actions=10, game_id=0)
    g1 = _make_synthetic_actions(n_actions=10, game_id=1)
    return pd.concat([g0, g1], ignore_index=True)


@pytest.fixture
def full_data(two_game_actions: pd.DataFrame) -> pd.DataFrame:
    """Synthetic ``full_data`` table: actions + home_team_id + metadata + labels."""
    df = two_game_actions.copy()
    # Add home_team_id (team 100 is always home)
    df["home_team_id"] = 100
    # Add some metadata columns
    df["competition_name"] = "Test League"
    df["season_name"] = "2025/2026"
    df["team_name"] = df["team_id"].map({100: "Team A", 200: "Team B"})
    df["player_name"] = "Player " + df["player_id"].astype(str)
    # Add labels
    rng = np.random.RandomState(99)
    df["scores"] = rng.randint(0, 2, len(df))
    df["concedes"] = rng.randint(0, 2, len(df))
    return df


# ─────────────────────────────────────
# Tests
# ─────────────────────────────────────


class TestOrientActions:
    """Tests for ``orient_actions``."""

    def test_output_shape_preserved(self, full_data: pd.DataFrame) -> None:
        """Row count is preserved after orientation."""
        oriented = orient_actions(full_data, home_team_id_col="home_team_id")
        assert len(oriented) == len(full_data)

    def test_home_team_id_dropped(self, full_data: pd.DataFrame) -> None:
        """The home_team_id column is dropped from the output."""
        oriented = orient_actions(full_data, home_team_id_col="home_team_id")
        assert "home_team_id" not in oriented.columns

    def test_type_name_added(self, full_data: pd.DataFrame) -> None:
        """``type_name`` is added if not present in the input."""
        assert "type_name" not in full_data.columns
        oriented = orient_actions(full_data, home_team_id_col="home_team_id")
        assert "type_name" in oriented.columns
        # No NaN type_names
        assert oriented["type_name"].notna().all()

    def test_coordinates_flipped_for_away_team(self, full_data: pd.DataFrame) -> None:
        """At least some coordinates differ after orientation (flip happened)."""
        oriented = orient_actions(full_data, home_team_id_col="home_team_id")
        # For away-team actions, start_x should have been flipped
        # (field_length - start_x), so they should differ from originals
        # at least for some rows.
        orig = full_data.sort_values(
            ["game_id", "period_id", "time_seconds", "action_id"]
        ).reset_index(drop=True)
        # Check that not all start_x values are identical (some were flipped)
        diff = (orig["start_x"].values - oriented["start_x"].values)
        assert not np.allclose(diff, 0), "Expected some coordinates to be flipped"


class TestComputeVaepFeatures:
    """Tests for ``compute_vaep_features``."""

    def test_output_has_game_id_and_action_id(self, full_data: pd.DataFrame) -> None:
        """Output contains game_id and action_id columns."""
        oriented = orient_actions(full_data, home_team_id_col="home_team_id")
        feats = compute_vaep_features(oriented, nb_prev_actions=NB_PREV_ACTIONS)
        assert "game_id" in feats.columns
        assert "action_id" in feats.columns

    def test_output_row_count_matches(self, full_data: pd.DataFrame) -> None:
        """One feature row per action."""
        oriented = orient_actions(full_data, home_team_id_col="home_team_id")
        feats = compute_vaep_features(oriented, nb_prev_actions=NB_PREV_ACTIONS)
        assert len(feats) == len(oriented)

    def test_feature_columns_have_suffixes(self, full_data: pd.DataFrame) -> None:
        """Feature columns end with ``_a0``, ``_a1``, ``_a2``."""
        oriented = orient_actions(full_data, home_team_id_col="home_team_id")
        feats = compute_vaep_features(oriented, nb_prev_actions=NB_PREV_ACTIONS)
        feat_cols = [c for c in feats.columns if c not in ("game_id", "action_id")]
        for suffix in ("_a0", "_a1", "_a2"):
            matching = [c for c in feat_cols if c.endswith(suffix)]
            assert len(matching) > 0, f"No columns with suffix {suffix}"

    def test_custom_feature_fns(self, full_data: pd.DataFrame) -> None:
        """Passing a subset of feature functions produces fewer columns."""
        import socceraction.vaep.features as vf

        oriented = orient_actions(full_data, home_team_id_col="home_team_id")
        feats_full = compute_vaep_features(oriented, nb_prev_actions=NB_PREV_ACTIONS)
        feats_small = compute_vaep_features(
            oriented,
            nb_prev_actions=NB_PREV_ACTIONS,
            feature_fns=[vf.startlocation, vf.movement],
        )
        assert feats_small.shape[1] < feats_full.shape[1]


class TestBuildVaepDataset:
    """Tests for ``build_vaep_dataset`` (end-to-end orchestrator)."""

    def test_output_has_original_metadata_cols(self, full_data: pd.DataFrame) -> None:
        """Output retains original metadata columns."""
        result = build_vaep_dataset(full_data, nb_prev_actions=NB_PREV_ACTIONS)
        for col in ("competition_name", "season_name", "team_name", "player_name"):
            assert col in result.columns, f"Missing metadata column: {col}"

    def test_output_has_vaep_feature_cols(self, full_data: pd.DataFrame) -> None:
        """Output contains VAEP feature columns with ``_a0`` suffix."""
        result = build_vaep_dataset(full_data, nb_prev_actions=NB_PREV_ACTIONS)
        a0_cols = [c for c in result.columns if c.endswith("_a0")]
        assert len(a0_cols) > 0, "No VAEP feature columns found"

    def test_output_has_labels(self, full_data: pd.DataFrame) -> None:
        """Output retains scores and concedes label columns."""
        result = build_vaep_dataset(full_data, nb_prev_actions=NB_PREV_ACTIONS)
        for col in LABEL_COLS:
            assert col in result.columns, f"Missing label column: {col}"

    def test_output_row_count(self, full_data: pd.DataFrame) -> None:
        """Inner join should keep all rows (synthetic data has matching keys)."""
        result = build_vaep_dataset(full_data, nb_prev_actions=NB_PREV_ACTIONS)
        assert len(result) == len(full_data)

    def test_missing_column_raises(self) -> None:
        """Missing required columns raise ValueError."""
        df = pd.DataFrame({"game_id": [1], "action_id": [0]})
        with pytest.raises(ValueError, match="Missing columns"):
            build_vaep_dataset(df)


class TestSaveVaepDataset:
    """Tests for ``save_vaep_dataset``."""

    def test_roundtrip(self, full_data: pd.DataFrame, tmp_path) -> None:
        """Save and reload a VAEP dataset from HDF5."""
        result = build_vaep_dataset(full_data, nb_prev_actions=NB_PREV_ACTIONS)
        out_path = tmp_path / "test_vaep.h5"
        save_vaep_dataset(result, out_path, key="vaep_data")

        assert out_path.exists()
        loaded = pd.read_hdf(str(out_path), key="vaep_data")
        assert loaded.shape == result.shape
        assert list(loaded.columns) == list(result.columns)
