"""Unit tests for football_ai.config.resolve_random_state."""
from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest

from football_ai.config import resolve_random_state


class TestResolveRandomState:
    """Tests for the resolve_random_state helper."""

    def test_single_value(self) -> None:
        assert resolve_random_state(42) == 42

    def test_first_wins(self) -> None:
        assert resolve_random_state(1, 2, 3) == 1

    def test_skips_none(self) -> None:
        assert resolve_random_state(None, 99) == 99

    def test_skips_multiple_none(self) -> None:
        assert resolve_random_state(None, None, 7) == 7

    def test_zero_is_valid_seed(self) -> None:
        """Zero must NOT be treated as falsy / skipped."""
        assert resolve_random_state(0) == 0
        assert resolve_random_state(0, 99) == 0

    def test_fallback_to_random(self) -> None:
        """When all candidates are None, fall back to random int in [0, 1_000_000)."""
        result = resolve_random_state(None, None)
        assert isinstance(result, int)
        assert 0 <= result < 1_000_000

    def test_fallback_no_args(self) -> None:
        result = resolve_random_state()
        assert isinstance(result, int)
        assert 0 <= result < 1_000_000

    @patch("football_ai.config.random")
    def test_fallback_uses_random_module(self, mock_random: object) -> None:
        """Verify the fallback calls random.randint with the correct range."""
        mock_random.randint.return_value = 42  # type: ignore[attr-defined]
        assert resolve_random_state(None) == 42
        mock_random.randint.assert_called_once_with(0, 999_999)  # type: ignore[attr-defined]

    def test_string_int_candidate_is_cast(self) -> None:
        """int() cast should handle numeric strings gracefully."""
        # resolve_random_state uses int(c), so a string "42" should work.
        assert resolve_random_state("42") == 42  # type: ignore[arg-type]

    def test_returns_int(self) -> None:
        result = resolve_random_state(None, 3.0)  # type: ignore[arg-type]
        assert isinstance(result, int)
        assert result == 3

    def test_string_none_is_skipped(self) -> None:
        """YAML may parse bare None as the string 'None'; must be treated as missing."""
        assert resolve_random_state("None", 99) == 99  # type: ignore[arg-type]
        assert resolve_random_state("none", 99) == 99  # type: ignore[arg-type]
        assert resolve_random_state(" None ", 99) == 99  # type: ignore[arg-type]

    def test_empty_string_is_skipped(self) -> None:
        assert resolve_random_state("", 99) == 99  # type: ignore[arg-type]
