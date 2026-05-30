"""Unit tests for EarlyStopping."""

import pytest

from src.training.early_stopping import EarlyStopping


class TestEarlyStoppingInit:
    """Test EarlyStopping initialization and validation."""

    def test_default_parameters(self):
        es = EarlyStopping()
        assert es.patience == 10
        assert es.min_delta == 0.0
        assert es.mode == "max"
        assert es.best_value is None
        assert es.epochs_without_improvement == 0
        assert es.should_stop is False

    def test_custom_parameters(self):
        es = EarlyStopping(patience=5, min_delta=0.01, mode="min")
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == "min"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be 'max' or 'min'"):
            EarlyStopping(mode="invalid")

    def test_invalid_patience_raises(self):
        with pytest.raises(ValueError, match="patience must be a positive integer"):
            EarlyStopping(patience=0)
        with pytest.raises(ValueError, match="patience must be a positive integer"):
            EarlyStopping(patience=-1)

    def test_negative_min_delta_raises(self):
        with pytest.raises(ValueError, match="min_delta must be non-negative"):
            EarlyStopping(min_delta=-0.1)


class TestEarlyStoppingMaxMode:
    """Test EarlyStopping in max mode (higher is better)."""

    def test_no_stop_while_improving(self):
        es = EarlyStopping(patience=3, mode="max")
        # Strictly increasing values
        for val in [0.1, 0.2, 0.3, 0.4, 0.5]:
            assert es.step(val) is False
        assert es.best_value == 0.5
        assert es.epochs_without_improvement == 0

    def test_stops_after_patience_exhausted(self):
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.9)  # best = 0.9
        es.step(0.8)  # no improvement, counter = 1
        es.step(0.7)  # no improvement, counter = 2
        result = es.step(0.6)  # no improvement, counter = 3 -> stop
        assert result is True
        assert es.should_stop is True
        assert es.epochs_without_improvement == 3
        assert es.best_value == 0.9

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.5)  # best = 0.5
        es.step(0.4)  # counter = 1
        es.step(0.3)  # counter = 2
        es.step(0.6)  # improvement! counter = 0, best = 0.6
        assert es.epochs_without_improvement == 0
        assert es.best_value == 0.6
        assert es.should_stop is False

    def test_equal_value_is_not_improvement(self):
        es = EarlyStopping(patience=2, mode="max", min_delta=0.0)
        es.step(0.5)  # best = 0.5
        es.step(0.5)  # same value, not improvement, counter = 1
        result = es.step(0.5)  # same value, counter = 2 -> stop
        assert result is True

    def test_min_delta_threshold(self):
        es = EarlyStopping(patience=2, mode="max", min_delta=0.1)
        es.step(0.5)  # best = 0.5
        # 0.59 is above 0.5 but not by min_delta=0.1
        es.step(0.59)  # counter = 1
        result = es.step(0.59)  # counter = 2 -> stop
        assert result is True
        assert es.best_value == 0.5

    def test_min_delta_met(self):
        es = EarlyStopping(patience=2, mode="max", min_delta=0.1)
        es.step(0.5)  # best = 0.5
        es.step(0.61)  # improvement (0.61 > 0.5 + 0.1), counter = 0
        assert es.epochs_without_improvement == 0
        assert es.best_value == 0.61


class TestEarlyStoppingMinMode:
    """Test EarlyStopping in min mode (lower is better)."""

    def test_no_stop_while_improving(self):
        es = EarlyStopping(patience=3, mode="min")
        for val in [1.0, 0.8, 0.6, 0.4, 0.2]:
            assert es.step(val) is False
        assert es.best_value == 0.2
        assert es.epochs_without_improvement == 0

    def test_stops_after_patience_exhausted(self):
        es = EarlyStopping(patience=3, mode="min")
        es.step(0.1)  # best = 0.1
        es.step(0.2)  # no improvement, counter = 1
        es.step(0.3)  # no improvement, counter = 2
        result = es.step(0.4)  # no improvement, counter = 3 -> stop
        assert result is True
        assert es.should_stop is True
        assert es.best_value == 0.1

    def test_min_delta_in_min_mode(self):
        es = EarlyStopping(patience=2, mode="min", min_delta=0.1)
        es.step(1.0)  # best = 1.0
        # 0.95 is below 1.0 but not by min_delta=0.1
        es.step(0.95)  # counter = 1
        result = es.step(0.95)  # counter = 2 -> stop
        assert result is True
        assert es.best_value == 1.0

    def test_min_delta_met_in_min_mode(self):
        es = EarlyStopping(patience=2, mode="min", min_delta=0.1)
        es.step(1.0)  # best = 1.0
        es.step(0.89)  # improvement (0.89 < 1.0 - 0.1), counter = 0
        assert es.epochs_without_improvement == 0
        assert es.best_value == 0.89


class TestEarlyStoppingReset:
    """Test the reset functionality."""

    def test_reset_clears_state(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.9)
        es.step(0.5)
        es.step(0.5)  # should stop
        assert es.should_stop is True
        assert es.best_value == 0.9

        es.reset()
        assert es.best_value is None
        assert es.epochs_without_improvement == 0
        assert es.should_stop is False

    def test_reset_allows_reuse(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.9)
        es.step(0.5)
        es.step(0.5)
        assert es.should_stop is True

        es.reset()
        # After reset, behaves like a fresh instance
        es.step(0.3)
        assert es.best_value == 0.3
        assert es.should_stop is False


class TestEarlyStoppingEdgeCases:
    """Test edge cases."""

    def test_patience_one(self):
        es = EarlyStopping(patience=1, mode="max")
        es.step(0.5)  # best = 0.5
        result = es.step(0.4)  # counter = 1 -> stop
        assert result is True

    def test_first_step_never_stops(self):
        es = EarlyStopping(patience=1, mode="max")
        result = es.step(0.5)
        assert result is False
        assert es.best_value == 0.5

    def test_stays_stopped_after_trigger(self):
        """Once stopped, should_stop stays True even with better values."""
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.9)
        es.step(0.5)
        es.step(0.5)  # stops here
        assert es.should_stop is True
        # Even a better value won't un-stop it
        es.step(1.0)
        assert es.should_stop is True

    def test_negative_values(self):
        es = EarlyStopping(patience=2, mode="min")
        es.step(-1.0)  # best = -1.0
        es.step(-2.0)  # improvement, best = -2.0
        assert es.best_value == -2.0
        assert es.epochs_without_improvement == 0

    def test_large_patience(self):
        es = EarlyStopping(patience=100, mode="max")
        es.step(1.0)
        for _ in range(99):
            es.step(0.0)
        assert es.should_stop is False
        es.step(0.0)  # counter = 100 -> stop
        assert es.should_stop is True
