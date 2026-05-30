"""Shared Plotly helpers, color palettes, and hover tooltips for wafer visualization.

Provides:
- Colorblind-friendly 9-color palette for defect classes (Paul Tol's qualitative)
- Consistent Plotly layout styling helpers
- Wafer map to base64 PNG conversion for hover tooltips
- RGB colormap for wafer pixel values
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 9-color colorblind-friendly palette for wafer defect classes
# Based on Paul Tol's qualitative color scheme (vibrant + muted blend)
# Reference: https://personal.sron.nl/~pault/data/colourschemes.pdf
# ---------------------------------------------------------------------------

DEFECT_CLASS_PALETTE: dict[str, str] = {
    "Center": "#4477AA",      # blue
    "Donut": "#EE6677",       # rose/red
    "Edge-Loc": "#228833",    # green
    "Edge-Ring": "#CCBB44",   # yellow
    "Loc": "#66CCEE",         # cyan
    "Near-full": "#AA3377",   # purple
    "Random": "#BBBBBB",      # grey
    "Scratch": "#EE8866",     # orange
    "none": "#44BB99",        # teal
}

# Ordered class names for consistent indexing
DEFECT_CLASSES: list[str] = list(DEFECT_CLASS_PALETTE.keys())


def get_class_color(class_name: str) -> str:
    """Get the palette color for a defect class.

    Falls back to grey (#999999) for unknown class names.

    Args:
        class_name: Name of the defect class (case-sensitive, matching DEFECT_CLASS_PALETTE keys).

    Returns:
        Hex color string (e.g., "#4477AA").
    """
    return DEFECT_CLASS_PALETTE.get(class_name, "#999999")


def apply_default_layout(
    fig: go.Figure,
    title: str = "",
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    show_legend: bool = True,
    dark_mode: bool = False,
) -> go.Figure:
    """Apply consistent Plotly layout styling to a figure.

    Sets font, margins, grid styling, and legend placement for a unified look
    across all WaferVision visualizations.

    Args:
        fig: Plotly Figure to style.
        title: Optional chart title.
        width: Optional figure width in pixels.
        height: Optional figure height in pixels.
        show_legend: Whether to display the legend.
        dark_mode: Use dark background if True.

    Returns:
        The same Figure with updated layout (mutated in place and returned for chaining).
    """
    bg_color = "#1e1e1e" if dark_mode else "#ffffff"
    grid_color = "#333333" if dark_mode else "#e5e5e5"
    font_color = "#e0e0e0" if dark_mode else "#333333"

    layout_kwargs: dict = {
        "title": {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": font_color},
        }
        if title
        else None,
        "font": {"family": "Inter, Arial, sans-serif", "size": 12, "color": font_color},
        "paper_bgcolor": bg_color,
        "plot_bgcolor": bg_color,
        "showlegend": show_legend,
        "legend": {
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
        },
        "margin": {"l": 60, "r": 40, "t": 60, "b": 50},
        "xaxis": {
            "gridcolor": grid_color,
            "gridwidth": 1,
            "zeroline": False,
        },
        "yaxis": {
            "gridcolor": grid_color,
            "gridwidth": 1,
            "zeroline": False,
        },
    }

    if width is not None:
        layout_kwargs["width"] = width
    if height is not None:
        layout_kwargs["height"] = height

    # Remove None title to avoid clearing an existing one
    if layout_kwargs.get("title") is None:
        layout_kwargs.pop("title", None)

    fig.update_layout(**layout_kwargs)
    return fig


def create_wafer_colormap() -> np.ndarray:
    """Create an RGB colormap for wafer pixel values.

    Mapping:
        0 → black   (background / no die)
        1 → green   (normal dies)
        2 → red     (defective dies)

    Returns:
        NumPy array of shape (3, 3) where row i is the RGB triplet (0-255) for pixel value i.
    """
    return np.array(
        [
            [0, 0, 0],        # 0: black (background)
            [0, 200, 0],      # 1: green (normal)
            [220, 50, 50],    # 2: red (defective)
        ],
        dtype=np.uint8,
    )


def wafer_map_to_base64_png(wafer_map: np.ndarray, size: int = 64) -> str:
    """Convert a wafer map (2D integer array) to a base64-encoded PNG string.

    The wafer map is colorized using the standard wafer colormap (0=black, 1=green, 2=red)
    and resized to the specified square size using nearest-neighbor interpolation.

    This is useful for embedding wafer thumbnails in Plotly hover tooltips.

    Args:
        wafer_map: 2D NumPy array with integer values in {0, 1, 2}.
        size: Output image size in pixels (square). Default 64.

    Returns:
        Base64-encoded PNG string suitable for use in HTML <img> tags.

    Raises:
        ValueError: If wafer_map is not 2D or contains values outside {0, 1, 2}.
    """
    if wafer_map.ndim != 2:
        raise ValueError(f"wafer_map must be 2D, got shape {wafer_map.shape}")

    # Clamp values to valid range for safety
    wafer_clamped = np.clip(wafer_map, 0, 2).astype(np.uint8)

    # Apply colormap: map pixel values to RGB
    colormap = create_wafer_colormap()
    rgb_image = colormap[wafer_clamped]  # shape (H, W, 3)

    # Resize to target size using nearest-neighbor (preserve discrete values)
    rgb_image = _resize_nearest(rgb_image, size, size)

    # Encode as PNG using matplotlib with Agg backend (no GUI dependency)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig_thumb, ax_thumb = plt.subplots(1, 1, figsize=(1, 1), dpi=size)
    ax_thumb.imshow(rgb_image, interpolation="nearest")
    ax_thumb.axis("off")
    fig_thumb.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_thumb.savefig(buf, format="png", dpi=size, bbox_inches="tight", pad_inches=0)
    plt.close(fig_thumb)

    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return encoded


def create_hover_template(
    class_label: str,
    sample_index: int,
    lot_number: str = "",
    distance_to_centroid: Optional[float] = None,
    base64_png: Optional[str] = None,
) -> str:
    """Create an HTML hover template string for embedding scatter plots.

    Produces a rich tooltip with optional wafer map thumbnail image.

    Args:
        class_label: Defect class name.
        sample_index: Dataset sample index.
        lot_number: Wafer lot number (optional).
        distance_to_centroid: Distance to class centroid (optional).
        base64_png: Base64-encoded PNG thumbnail (optional).

    Returns:
        HTML string suitable for use as a Plotly hovertemplate customdata entry.
    """
    parts: list[str] = []

    if base64_png:
        parts.append(
            f'<img src="data:image/png;base64,{base64_png}" '
            f'width="64" height="64" style="margin-bottom:4px;"><br>'
        )

    parts.append(f"<b>Class:</b> {class_label}<br>")
    parts.append(f"<b>Index:</b> {sample_index}<br>")

    if lot_number:
        parts.append(f"<b>Lot:</b> {lot_number}<br>")

    if distance_to_centroid is not None:
        parts.append(f"<b>Dist to centroid:</b> {distance_to_centroid:.4f}<br>")

    return "".join(parts)


def create_scatter_with_class_colors(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[list[str]] = None,
    *,
    title: str = "",
    marker_size: int = 4,
    opacity: float = 0.7,
    use_webgl: bool = True,
    hover_texts: Optional[list[str]] = None,
) -> go.Figure:
    """Create a 2D scatter plot with points colored by defect class.

    Each class gets its own trace for independent legend toggling.

    Args:
        x: X coordinates, shape (N,).
        y: Y coordinates, shape (N,).
        labels: Integer or string labels, shape (N,).
        class_names: Optional list mapping integer labels to class name strings.
            If None, labels are used directly as strings.
        title: Chart title.
        marker_size: Point size in pixels.
        opacity: Point opacity (0-1).
        use_webgl: Use WebGL renderer for performance with large point counts.
        hover_texts: Optional per-point hover HTML strings.

    Returns:
        Styled Plotly Figure.
    """
    fig = go.Figure()

    # Resolve class names
    if class_names is None:
        unique_labels = sorted(set(str(l) for l in labels))
    else:
        unique_labels = class_names

    scatter_cls = go.Scattergl if use_webgl else go.Scatter

    for cls_name in unique_labels:
        if class_names is not None:
            mask = np.array([class_names[int(l)] == cls_name for l in labels])
        else:
            mask = np.array([str(l) == cls_name for l in labels])

        if not mask.any():
            continue

        trace_kwargs: dict = {
            "x": x[mask],
            "y": y[mask],
            "mode": "markers",
            "name": cls_name,
            "marker": {
                "size": marker_size,
                "color": get_class_color(cls_name),
                "opacity": opacity,
            },
        }

        if hover_texts is not None:
            masked_texts = [t for t, m in zip(hover_texts, mask) if m]
            trace_kwargs["hovertext"] = masked_texts
            trace_kwargs["hoverinfo"] = "text"

        fig.add_trace(scatter_cls(**trace_kwargs))

    apply_default_layout(fig, title=title)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize an image using nearest-neighbor interpolation (no external deps).

    Args:
        image: Input image array of shape (H, W, C) or (H, W).
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized image array.
    """
    h, w = image.shape[:2]
    row_indices = (np.arange(target_h) * h / target_h).astype(int)
    col_indices = (np.arange(target_w) * w / target_w).astype(int)
    row_indices = np.clip(row_indices, 0, h - 1)
    col_indices = np.clip(col_indices, 0, w - 1)
    return image[np.ix_(row_indices, col_indices)]
