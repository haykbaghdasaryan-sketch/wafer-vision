"""EarlyStopping logic for training termination.

Provides the EarlyStopping class that monitors a metric and halts training
when the metric plateaus (no improvement for a configurable number of epochs).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to halt training when a monitored metric plateaus.

    Tracks the best metric value seen and counts consecutive epochs without
    improvement. After `patience` epochs without improvement >= min_delta,
    `should_stop` becomes True.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
            Default is 10.
        min_delta: Minimum change in the monitored metric to qualify as an
            improvement. Default is 0.0.
        mode: One of "max" or "min". In "max" mode, training stops when the
            metric stops increasing (e.g., accuracy). In "min" mode, training
            stops when the metric stops decreasing (e.g., loss).

    Raises:
        ValueError: If mode is not "max" or "min".
        ValueError: If patience is not a positive integer.
        ValueError: If min_delta is negative.

    Example:
        early_stop = EarlyStopping(patience=10, min_delta=0.001, mode="max")
        for epoch in range(max_epochs):
            val_acc = train_one_epoch(...)
            if early_stop.step(val_acc):
                print("Early stopping triggered")
                break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(
                f"mode must be 'max' or 'min', got '{mode}'"
            )
        if patience < 1:
            raise ValueError(
                f"patience must be a positive integer, got {patience}"
            )
        if min_delta < 0:
            raise ValueError(
                f"min_delta must be non-negative, got {min_delta}"
            )

        self._patience = patience
        self._min_delta = min_delta
        self._mode = mode
        self._best_value: Optional[float] = None
        self._epochs_without_improvement: int = 0
        self._should_stop: bool = False

    @property
    def patience(self) -> int:
        """Number of epochs to wait for improvement."""
        return self._patience

    @property
    def min_delta(self) -> float:
        """Minimum change to qualify as improvement."""
        return self._min_delta

    @property
    def mode(self) -> str:
        """Optimization mode: 'max' or 'min'."""
        return self._mode

    @property
    def best_value(self) -> Optional[float]:
        """The best metric value observed so far, or None if no steps taken."""
        return self._best_value

    @property
    def epochs_without_improvement(self) -> int:
        """Number of consecutive epochs without improvement."""
        return self._epochs_without_improvement

    @property
    def should_stop(self) -> bool:
        """Whether patience has been exhausted and training should stop."""
        return self._should_stop

    def _is_improvement(self, current: float) -> bool:
        """Check whether the current value is an improvement over the best.

        Args:
            current: The current metric value.

        Returns:
            True if current value improves upon best_value by at least min_delta.
        """
        if self._best_value is None:
            return True

        if self._mode == "max":
            return current > self._best_value + self._min_delta
        else:  # mode == "min"
            return current < self._best_value - self._min_delta

    def step(self, value: float) -> bool:
        """Record a new metric value and check if training should stop.

        Args:
            value: The current epoch's metric value.

        Returns:
            True if training should stop (patience exhausted), False otherwise.
        """
        if self._is_improvement(value):
            self._best_value = value
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._epochs_without_improvement >= self._patience:
            self._should_stop = True
            logger.info(
                "Early stopping triggered: no improvement in %d epochs. "
                "Best value: %.6f, last value: %.6f, mode: %s, min_delta: %.6f",
                self._patience,
                self._best_value if self._best_value is not None else float("nan"),
                value,
                self._mode,
                self._min_delta,
            )

        return self._should_stop

    def reset(self) -> None:
        """Reset internal state to allow reuse.

        Clears best_value, epochs_without_improvement, and should_stop.
        """
        self._best_value = None
        self._epochs_without_improvement = 0
        self._should_stop = False
