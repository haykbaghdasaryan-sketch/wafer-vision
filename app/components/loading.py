"""Loading states with descriptive text.

Provides reusable loading indicators with context-specific
messages to keep users informed during long operations.
"""

from contextlib import contextmanager
from typing import Generator, Optional

import streamlit as st


# Default loading messages for common operations
_LOADING_MESSAGES: dict[str, str] = {
    "embeddings": "Loading embeddings... This may take a moment for large datasets.",
    "umap": "Computing UMAP projection... This can take up to 2 minutes for 25K+ points.",
    "tsne": "Computing t-SNE projection... This can take up to 3 minutes for 25K+ points.",
    "gradcam": "Generating Grad-CAM heatmap...",
    "retrieval": "Searching for similar wafer maps...",
    "metrics": "Computing evaluation metrics...",
    "anomaly": "Running anomaly detection...",
    "model": "Loading model checkpoint...",
    "data": "Loading dataset...",
}


def loading_placeholder(
    operation: str = "",
    message: Optional[str] = None,
) -> None:
    """Display a loading placeholder with descriptive text.

    Shows a spinner icon and informative message while content loads.

    Args:
        operation: Operation key (e.g., "embeddings", "umap").
            Used to look up a default message if message is not provided.
        message: Custom loading message. Overrides the default.
    """
    display_msg = message or _LOADING_MESSAGES.get(
        operation, f"Loading {operation}..." if operation else "Loading..."
    )
    st.info(f"⏳ {display_msg}")


@contextmanager
def loading_spinner(
    operation: str = "",
    message: Optional[str] = None,
) -> Generator[None, None, None]:
    """Context manager that shows a Streamlit spinner during execution.

    Args:
        operation: Operation key for default message lookup.
        message: Custom spinner message. Overrides the default.

    Yields:
        None

    Example:
        with loading_spinner("embeddings"):
            embeddings = load_large_file()
    """
    display_msg = message or _LOADING_MESSAGES.get(
        operation, f"Loading {operation}..." if operation else "Processing..."
    )
    with st.spinner(display_msg):
        yield


def progress_bar(
    total: int,
    operation: str = "",
    message: Optional[str] = None,
) -> st.progress:
    """Create a progress bar with descriptive text.

    Args:
        total: Total number of steps.
        operation: Operation key for default message lookup.
        message: Custom progress message.

    Returns:
        Streamlit progress bar widget.
    """
    display_msg = message or _LOADING_MESSAGES.get(
        operation, f"Processing {operation}..." if operation else "Processing..."
    )
    st.caption(display_msg)
    return st.progress(0)
