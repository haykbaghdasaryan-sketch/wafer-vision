"""Wafer map display card component.

Provides reusable cards for displaying wafer maps with metadata,
including single cards and gallery grids.
"""

from typing import Optional

import numpy as np
import streamlit as st


# Wafer map colormap: 0=background (white), 1=normal (blue), 2=defect (red)
_WAFER_COLORS = np.array(
    [
        [255, 255, 255],  # 0: background - white
        [66, 133, 244],  # 1: normal - blue
        [234, 67, 53],  # 2: defect - red
    ],
    dtype=np.uint8,
)


def colorize_wafer(wafer_map: np.ndarray) -> np.ndarray:
    """Convert a wafer map (values 0,1,2) to an RGB image.

    Args:
        wafer_map: 2D numpy array with values in {0, 1, 2}.

    Returns:
        RGB image as (H, W, 3) uint8 array.
    """
    wafer_clipped = np.clip(wafer_map.astype(int), 0, 2)
    return _WAFER_COLORS[wafer_clipped]


def wafer_card(
    wafer_map: np.ndarray,
    label: str = "",
    index: Optional[int] = None,
    distance: Optional[float] = None,
    confidence: Optional[float] = None,
    caption_extra: str = "",
    width: int = 150,
) -> None:
    """Display a single wafer map as a styled card.

    Args:
        wafer_map: 2D numpy array with values in {0, 1, 2}.
        label: Class label to display.
        index: Dataset index of the wafer map.
        distance: Optional distance value to display.
        confidence: Optional confidence score (0-1).
        caption_extra: Additional text for the caption.
        width: Display width in pixels.
    """
    rgb = colorize_wafer(wafer_map)
    st.image(rgb, width=width)

    # Build caption
    parts = []
    if label:
        parts.append(f"**{label}**")
    if index is not None:
        parts.append(f"idx: {index}")
    if distance is not None:
        parts.append(f"dist: {distance:.4f}")
    if confidence is not None:
        parts.append(f"conf: {confidence:.2%}")
    if caption_extra:
        parts.append(caption_extra)

    if parts:
        st.caption(" | ".join(parts))


def wafer_gallery(
    wafer_maps: list[np.ndarray],
    labels: Optional[list[str]] = None,
    indices: Optional[list[int]] = None,
    distances: Optional[list[float]] = None,
    columns: int = 5,
    width: int = 120,
) -> None:
    """Display a grid gallery of wafer map cards.

    Args:
        wafer_maps: List of 2D wafer map arrays.
        labels: Optional list of class labels.
        indices: Optional list of dataset indices.
        distances: Optional list of distance values.
        columns: Number of columns in the grid.
        width: Display width per card in pixels.
    """
    if not wafer_maps:
        st.info("No wafer maps to display.")
        return

    cols = st.columns(columns)
    for i, wmap in enumerate(wafer_maps):
        with cols[i % columns]:
            wafer_card(
                wafer_map=wmap,
                label=labels[i] if labels else "",
                index=indices[i] if indices else None,
                distance=distances[i] if distances else None,
                width=width,
            )
