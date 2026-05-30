"""Complete exception hierarchy for WaferVision."""

from datetime import datetime
from typing import Any, Optional
import uuid


class WaferVisionError(Exception):
    """Base exception for all WaferVision errors.

    Attributes:
        message: Human-readable error description.
        run_id: Correlation ID for log tracing.
        timestamp: When the error occurred.
        is_user_error: Whether this is a user-fixable error vs system error.
    """

    def __init__(
        self,
        message: str,
        *,
        run_id: Optional[str] = None,
        is_user_error: bool = True,
    ) -> None:
        self.message = message
        self.run_id = run_id or str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.is_user_error = is_user_error
        super().__init__(message)


# --- Data Errors ---


class DataError(WaferVisionError):
    """Base for all data-related errors."""

    def __init__(self, message: str, path: Optional[str] = None, **kwargs: Any) -> None:
        self.path = path
        super().__init__(message, **kwargs)


class DataFileNotFoundError(DataError):
    """Raised when dataset file is missing.

    Raises:
        DataFileNotFoundError: With message template
            "Dataset file not found at: {path}. Download it using 'make download'."
    """

    def __init__(self, path: str) -> None:
        self.path = path
        message = f"Dataset file not found at: {path}. Download it using 'make download'."
        super().__init__(message, path=path)


class DataCorruptionError(DataError):
    """Raised when dataset file is corrupted or unsafe to unpickle.

    Attributes:
        byte_offset: Position in file where corruption was detected.
    """

    def __init__(self, path: str, byte_offset: int, cause: str) -> None:
        self.byte_offset = byte_offset
        self.cause = cause
        message = (
            f"Data corruption detected in '{path}' at byte offset {byte_offset}: "
            f"{cause}. Consider re-downloading the dataset."
        )
        super().__init__(message, path=path, is_user_error=False)


class DataValidationError(DataError):
    """Raised when data fails validation checks."""

    def __init__(self, record_index: int, violation: str) -> None:
        self.record_index = record_index
        self.violation = violation
        message = f"Data validation failed at record {record_index}: {violation}"
        super().__init__(message)


# --- Model Errors ---


class ModelError(WaferVisionError):
    """Base for model-related errors."""

    pass


class UnsupportedBackboneError(ModelError):
    """Raised when an unsupported backbone name is requested.

    Message format: "Unsupported backbone: '{name}'. Supported: {SUPPORTED_BACKBONES}"
    """

    def __init__(self, name: str, supported: list[str]) -> None:
        self.name = name
        self.supported = supported
        message = f"Unsupported backbone: '{name}'. Supported: {supported}"
        super().__init__(message)


class EmbeddingNotFoundError(ModelError):
    """Raised when pre-computed embeddings are not available."""

    def __init__(self, model_name: str, training_mode: str) -> None:
        self.model_name = model_name
        self.training_mode = training_mode
        message = (
            f"Embeddings not found for model '{model_name}' with training mode "
            f"'{training_mode}'. Run embedding extraction first."
        )
        super().__init__(message)


class DimensionMismatchError(ModelError):
    """Raised when embedding dimensions don't match between query and database."""

    def __init__(self, expected: int, actual: int, context: str) -> None:
        self.expected = expected
        self.actual = actual
        self.context = context
        message = (
            f"Dimension mismatch in {context}: expected {expected}, got {actual}"
        )
        super().__init__(message)


# --- Training Errors ---


class TrainingError(WaferVisionError):
    """Base for training-related errors."""

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.epoch = epoch
        self.batch_idx = batch_idx
        super().__init__(message, **kwargs)


class TrainingDivergenceError(TrainingError):
    """Raised when loss becomes NaN or Inf."""

    def __init__(self, epoch: int, batch_idx: int, loss_value: float) -> None:
        self.loss_value = loss_value
        message = (
            f"Training divergence detected at epoch {epoch}, batch {batch_idx}: "
            f"loss value = {loss_value}"
        )
        super().__init__(message, epoch=epoch, batch_idx=batch_idx, is_user_error=False)


class CheckpointLoadError(TrainingError):
    """Raised when checkpoint is corrupted or incompatible."""

    def __init__(
        self, path: str, key: str, expected_shape: tuple, actual_shape: tuple
    ) -> None:
        self.path = path
        self.key = key
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        message = (
            f"Checkpoint load error in '{path}': key '{key}' has shape "
            f"{actual_shape}, expected {expected_shape}"
        )
        super().__init__(message, is_user_error=False)


class EmbeddingCollapseWarning(TrainingError):
    """Warning raised when embedding space is collapsing."""

    def __init__(self, epoch: int, std_dev: float) -> None:
        self.std_dev = std_dev
        message = (
            f"Potential embedding collapse detected at epoch {epoch}: "
            f"embedding std_dev = {std_dev:.6f} (threshold: 0.01)"
        )
        super().__init__(message, epoch=epoch)


# --- Config Errors ---


class ConfigError(WaferVisionError):
    """Base for configuration errors."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when a config value is out of valid range."""

    def __init__(self, parameter: str, value: Any, valid_range: str) -> None:
        self.parameter = parameter
        self.value = value
        self.valid_range = valid_range
        message = (
            f"Configuration validation error: parameter '{parameter}' has value "
            f"{value!r}, valid range: {valid_range}"
        )
        super().__init__(message)


class ConfigConflictError(ConfigError):
    """Raised when conflicting configuration options are specified."""

    def __init__(self, param_a: str, param_b: str, explanation: str) -> None:
        self.param_a = param_a
        self.param_b = param_b
        self.explanation = explanation
        message = (
            f"Configuration conflict between '{param_a}' and '{param_b}': "
            f"{explanation}"
        )
        super().__init__(message)


# --- IO Errors ---


class WaferIOError(WaferVisionError):
    """Base for I/O errors."""

    pass


class AtomicWriteError(WaferIOError):
    """Raised when atomic file write fails."""

    def __init__(self, target_path: str, temp_path: str, cause: str) -> None:
        self.target_path = target_path
        self.temp_path = temp_path
        self.cause = cause
        message = (
            f"Atomic write failed for '{target_path}' (temp: '{temp_path}'): {cause}"
        )
        super().__init__(message, is_user_error=False)


class InsufficientDiskError(WaferIOError):
    """Raised when disk space is below required threshold."""

    def __init__(self, required_mb: float, available_mb: float, path: str) -> None:
        self.required_mb = required_mb
        self.available_mb = available_mb
        self.path = path
        message = (
            f"Insufficient disk space at '{path}': "
            f"required {required_mb:.1f} MB, available {available_mb:.1f} MB"
        )
        super().__init__(message)
