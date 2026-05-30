"""Metric tables and radar charts component.

Provides reusable components for displaying evaluation metrics
as cards, tables, and radar/polar charts.
"""

from typing import Optional

import numpy as np
import streamlit as st


def metric_card(
    label: str,
    value: float | str,
    delta: Optional[float | str] = None,
    delta_color: str = "normal",
    help_text: str = "",
) -> None:
    """Display a single metric as a Streamlit metric card.

    Args:
        label: Metric name.
        value: Metric value (numeric or string).
        delta: Optional delta value showing change.
        delta_color: Color for delta ("normal", "inverse", "off").
        help_text: Optional tooltip text.
    """
    st.metric(
        label=label,
        value=f"{value:.4f}" if isinstance(value, float) else value,
        delta=f"{delta:+.4f}" if isinstance(delta, float) else delta,
        delta_color=delta_color,
        help=help_text or None,
    )


def metric_table(
    metrics: dict[str, float | str],
    title: str = "Metrics",
    highlight_best: bool = False,
) -> None:
    """Display metrics as a formatted table.

    Args:
        metrics: Dictionary mapping metric names to values.
        title: Table title.
        highlight_best: If True, highlight the best value (highest).
    """
    if title:
        st.subheader(title)

    if not metrics:
        st.info("No metrics available.")
        return

    # Format as markdown table
    rows = []
    for name, value in metrics.items():
        if isinstance(value, float):
            rows.append(f"| {name} | {value:.4f} |")
        else:
            rows.append(f"| {name} | {value} |")

    table_md = "| Metric | Value |\n|--------|-------|\n" + "\n".join(rows)
    st.markdown(table_md)


def radar_chart(
    models: dict[str, dict[str, float]],
    title: str = "Model Comparison",
    normalize: bool = True,
) -> None:
    """Display a radar/polar chart comparing multiple models.

    Args:
        models: Dict mapping model names to their metric dicts.
            Example: {"resnet50": {"knn@5": 0.85, "recall@1": 0.72}, ...}
        title: Chart title.
        normalize: If True, normalize metrics to [0, 1] range across models.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly is required for radar charts. Install with: pip install plotly")
        return

    if not models:
        st.info("No model data available for comparison.")
        return

    if len(models) > 4:
        st.warning("Radar chart supports up to 4 models. Showing first 4.")
        models = dict(list(models.items())[:4])

    # Get all metric names (union across models)
    all_metrics = sorted(
        set(m for model_metrics in models.values() for m in model_metrics.keys())
    )

    if not all_metrics:
        st.info("No metrics to display.")
        return

    # Normalize if requested
    if normalize:
        min_vals = {}
        max_vals = {}
        for metric in all_metrics:
            values = [
                model_metrics.get(metric, 0.0)
                for model_metrics in models.values()
            ]
            min_vals[metric] = min(values)
            max_vals[metric] = max(values)

    fig = go.Figure()

    for model_name, model_metrics in models.items():
        values = []
        for metric in all_metrics:
            v = model_metrics.get(metric, 0.0)
            if normalize and max_vals[metric] != min_vals[metric]:
                v = (v - min_vals[metric]) / (max_vals[metric] - min_vals[metric])
            elif normalize:
                v = 1.0  # All same value → normalized to 1
            values.append(v)

        # Close the polygon
        values.append(values[0])
        categories = all_metrics + [all_metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=model_name,
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=title,
    )

    st.plotly_chart(fig, use_container_width=True)
