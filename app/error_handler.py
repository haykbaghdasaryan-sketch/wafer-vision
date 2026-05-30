"""User-friendly error display without raw tracebacks.

Provides safe_execute() context manager that catches exceptions,
displays user-friendly messages via Streamlit, and logs full
tracebacks to structured logging.
"""

import logging
import traceback
from contextlib import contextmanager
from typing import Generator

import streamlit as st

from src.exceptions import (
    ConfigError,
    DataError,
    ModelError,
    TrainingError,
    WaferIOError,
    WaferVisionError,
)

logger = logging.getLogger(__name__)

# Map exception types to user-friendly category labels and icons
_ERROR_CATEGORIES: dict[type, tuple[str, str]] = {
    DataError: ("Data Error", "📂"),
    ModelError: ("Model Error", "🧠"),
    TrainingError: ("Training Error", "🏋️"),
    ConfigError: ("Configuration Error", "⚙️"),
    WaferIOError: ("File I/O Error", "💾"),
    WaferVisionError: ("Application Error", "⚠️"),
}


def _get_error_category(exc: Exception) -> tuple[str, str]:
    """Get the user-friendly category and icon for an exception."""
    for exc_type, (category, icon) in _ERROR_CATEGORIES.items():
        if isinstance(exc, exc_type):
            return category, icon
    return "Unexpected Error", "❌"


def _get_user_message(exc: Exception) -> str:
    """Extract a user-friendly message from an exception.

    For WaferVisionError subclasses, uses the message attribute.
    For other exceptions, provides a generic message without internals.
    """
    if isinstance(exc, WaferVisionError):
        return exc.message
    # Generic fallback - don't expose internals
    return (
        "An unexpected error occurred. Please try again or check the logs "
        "for more details."
    )


def _get_suggestion(exc: Exception) -> str | None:
    """Provide actionable suggestions for common errors."""
    if isinstance(exc, DataError):
        if "not found" in str(exc).lower():
            return "Make sure the dataset file exists. Run `make download` to fetch it."
        if "corruption" in str(exc).lower():
            return "The data file may be corrupted. Try re-downloading with `make download`."
        return "Check that your data files are valid and accessible."
    if isinstance(exc, ModelError):
        if "embedding" in str(exc).lower() and "not found" in str(exc).lower():
            return "Run `make extract` to generate embeddings for this model."
        return "Verify that the model checkpoint exists and is compatible."
    if isinstance(exc, ConfigError):
        return "Review your configuration files in the configs/ directory."
    if isinstance(exc, WaferIOError):
        return "Check disk space and file permissions."
    return None


@contextmanager
def safe_execute(
    operation: str = "operation",
    show_error: bool = True,
) -> Generator[None, None, None]:
    """Context manager that catches exceptions and displays user-friendly errors.

    Catches all exceptions, logs the full traceback, and optionally
    displays a Streamlit error message without raw tracebacks.

    Args:
        operation: Human-readable description of the operation being performed.
            Used in error messages (e.g., "loading embeddings").
        show_error: If True, display the error in Streamlit. If False,
            only log it (useful for background operations).

    Yields:
        None

    Example:
        with safe_execute("loading embeddings"):
            embeddings = load_embeddings(model_name)
    """
    try:
        yield
    except WaferVisionError as exc:
        category, icon = _get_error_category(exc)
        user_msg = _get_user_message(exc)
        suggestion = _get_suggestion(exc)

        # Log full details
        logger.error(
            "WaferVision error during %s: %s [run_id=%s]",
            operation,
            exc.message,
            exc.run_id,
            exc_info=True,
        )

        if show_error:
            error_text = f"{icon} **{category}** while {operation}\n\n{user_msg}"
            if suggestion:
                error_text += f"\n\n💡 **Suggestion:** {suggestion}"
            st.error(error_text)

    except Exception as exc:
        # Unexpected errors - log full traceback but show generic message
        logger.error(
            "Unexpected error during %s: %s",
            operation,
            str(exc),
            exc_info=True,
        )

        if show_error:
            st.error(
                f"❌ **Unexpected Error** while {operation}\n\n"
                "An unexpected error occurred. Please try again or contact support "
                "if the problem persists.\n\n"
                f"💡 **Suggestion:** Check the application logs for details."
            )
