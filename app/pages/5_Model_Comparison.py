import app._path_fix  # noqa: F401
"""Model Comparison page: metrics table, radar chart, side-by-side UMAP."""

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import loading_spinner, metric_table, radar_chart
from app.error_handler import safe_execute

# --- Constants ---
OUTPUTS_DIR = Path("outputs")
METRICS_DIR = OUTPUTS_DIR / "metrics"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "None",
]

MODELS = ["resnet50", "efficientnet_b0", "vit_b16"]
MODES = ["pretrained", "finetune", "metric_triplet", "metric_supcon", "selfsupervised"]

# Key metrics for comparison
KEY_METRICS = [
    "knn_accuracy_k5",
    "recall_at_k5",
    "precision_at_k5",
    "silhouette_score",
    "nmi",
    "mean_average_precision",
    "separability_index",
]


# --- Data Loading ---
@st.cache_data(ttl=300)
def _load_all_metrics() -> Optional[pd.DataFrame]:
    """Load all metrics CSV files into a single DataFrame."""
    if not METRICS_DIR.exists():
        return None

    csv_files = list(METRICS_DIR.glob("*.csv"))
    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(ttl=300)
def _pivot_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot metrics into a comparison table: rows=model, columns=metric."""
    # Create a unique model identifier
    metrics_df = metrics_df.copy()
    metrics_df["model_id"] = metrics_df["model_name"] + " / " + metrics_df["training_mode"]
    metrics_df["metric_value"] = pd.to_numeric(metrics_df["metric_value"], errors="coerce")

    # Pivot: model_id as rows, metric_name as columns
    pivot = metrics_df.pivot_table(
        index="model_id",
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    )

    # Only keep key metrics that exist
    available = [m for m in KEY_METRICS if m in pivot.columns]
    if available:
        pivot = pivot[available]

    return pivot.reset_index()


@st.cache_resource(ttl=600)
def _load_embeddings_for_model(model: str, mode: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load embeddings for a specific model/mode."""
    emb_dir = EMBEDDINGS_DIR / f"{model}_{mode}"
    emb_path = emb_dir / "embeddings.npy"
    labels_path = emb_dir / "labels.npy"

    if emb_path.exists() and labels_path.exists():
        return np.load(emb_path), np.load(labels_path)

    # Try alternative naming
    if EMBEDDINGS_DIR.exists():
        for d in EMBEDDINGS_DIR.iterdir():
            if model in d.name and mode in d.name:
                ep = d / "embeddings.npy"
                lp = d / "labels.npy"
                if ep.exists() and lp.exists():
                    return np.load(ep), np.load(lp)
    return None


# --- Page Layout ---
st.title("📈 Model Comparison")
st.markdown("Compare embedding quality across models and training modes.")

# --- Load Metrics ---
with safe_execute("loading metrics"):
    metrics_df = _load_all_metrics()

if metrics_df is None or metrics_df.empty:
    st.info(
        "📊 No evaluation metrics found. Run model evaluation with `make evaluate` "
        "to populate comparison data.\n\n"
        "At least 2 models need to be evaluated for meaningful comparison."
    )
    st.stop()

# Check if we have enough models
unique_models = metrics_df["model_name"].nunique()
unique_combos = (metrics_df["model_name"] + "_" + metrics_df["training_mode"]).nunique()

if unique_combos < 2:
    st.warning(
        "⚠️ Only 1 model/mode combination found. Model comparison requires at least 2.\n\n"
        "Train additional models with different backbones or training modes, "
        "then run `make evaluate` to generate comparison data."
    )

# --- Sortable Metrics Table ---
st.markdown("---")
st.subheader("📋 Metrics Table")

with safe_execute("building comparison table"):
    pivot_df = _pivot_metrics(metrics_df)

    if pivot_df.empty:
        st.info("No metrics available to display.")
    else:
        # Highlight best values (green)
        def _highlight_best(s):
            """Highlight the maximum value in each column green."""
            if s.dtype in [np.float64, np.float32, float]:
                is_max = s == s.max()
                return ["background-color: #c6efce" if v else "" for v in is_max]
            return [""] * len(s)

        numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns.tolist()
        styled = pivot_df.style.apply(_highlight_best, subset=numeric_cols)
        styled = styled.format({col: "{:.4f}" for col in numeric_cols})

        st.dataframe(styled, use_container_width=True, height=400)

        # Identify winning model
        if "knn_accuracy_k5" in pivot_df.columns:
            best_idx = pivot_df["knn_accuracy_k5"].idxmax()
            best_model = pivot_df.loc[best_idx, "model_id"]
            best_score = pivot_df.loc[best_idx, "knn_accuracy_k5"]
            st.success(f"🏆 **Best Model (KNN@5):** {best_model} — {best_score:.4f}")

# --- Radar Chart ---
st.markdown("---")
st.subheader("📊 Radar Chart")

with safe_execute("generating radar chart"):
    # Build model dict for radar chart
    models_for_radar: dict[str, dict[str, float]] = {}

    for _, row in pivot_df.iterrows():
        model_id = row["model_id"]
        metrics_dict = {}
        for col in numeric_cols:
            if pd.notna(row[col]):
                metrics_dict[col] = float(row[col])
        if metrics_dict:
            models_for_radar[model_id] = metrics_dict

    if len(models_for_radar) > 4:
        st.info("Showing top 4 models by KNN@5 accuracy on radar chart.")
        # Sort by knn_accuracy_k5 and take top 4
        sorted_models = sorted(
            models_for_radar.items(),
            key=lambda x: x[1].get("knn_accuracy_k5", 0),
            reverse=True,
        )[:4]
        models_for_radar = dict(sorted_models)

    if models_for_radar:
        radar_chart(
            models=models_for_radar,
            title="Model Comparison (normalized to [0, 1])",
            normalize=True,
        )
    else:
        st.info("Not enough data for radar chart visualization.")

# --- Side-by-Side UMAP ---
st.markdown("---")
st.subheader("🗺️ Side-by-Side UMAP")

with safe_execute("loading UMAP projections"):
    # Let user pick two models to compare
    available_combos = list(pivot_df["model_id"].values) if not pivot_df.empty else []

    if len(available_combos) < 2:
        st.info("Need at least 2 models for side-by-side UMAP comparison.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            model_a = st.selectbox("Model A", options=available_combos, index=0, key="compare_model_a")
        with col2:
            default_b = 1 if len(available_combos) > 1 else 0
            model_b = st.selectbox("Model B", options=available_combos, index=default_b, key="compare_model_b")

        if st.button("Generate UMAP Comparison", key="compare_umap_btn"):
            # Parse model_id back to model_name and mode
            def _parse_model_id(model_id: str) -> tuple[str, str]:
                parts = model_id.split(" / ")
                return parts[0], parts[1] if len(parts) > 1 else "pretrained"

            model_a_name, mode_a = _parse_model_id(model_a)
            model_b_name, mode_b = _parse_model_id(model_b)

            emb_a = _load_embeddings_for_model(model_a_name, mode_a)
            emb_b = _load_embeddings_for_model(model_b_name, mode_b)

            if emb_a is None:
                st.warning(f"No embeddings found for {model_a}")
            elif emb_b is None:
                st.warning(f"No embeddings found for {model_b}")
            else:
                from src.visualization.umap_viz import UMAPVisualizer

                with loading_spinner("umap"):
                    viz = UMAPVisualizer(n_components=2, cache_dir=str(OUTPUTS_DIR / "cache" / "umap"))

                    proj_a = viz.fit_transform(emb_a[0], emb_a[1])
                    proj_b = viz.fit_transform(emb_b[0], emb_b[1])

                col1, col2 = st.columns(2)
                with col1:
                    fig_a = viz.create_figure(proj_a, emb_a[1][:len(proj_a)], CLASS_NAMES, title=model_a)
                    fig_a.update_layout(height=400)
                    st.plotly_chart(fig_a, use_container_width=True)
                with col2:
                    fig_b = viz.create_figure(proj_b, emb_b[1][:len(proj_b)], CLASS_NAMES, title=model_b)
                    fig_b.update_layout(height=400)
                    st.plotly_chart(fig_b, use_container_width=True)

# --- CSV Export ---
st.markdown("---")
st.subheader("📥 Export")

with safe_execute("preparing export"):
    if not pivot_df.empty:
        csv_buffer = io.StringIO()
        pivot_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="⬇️ Download Metrics CSV",
            data=csv_bytes,
            file_name="wafer_vision_model_comparison.csv",
            mime="text/csv",
            key="export_csv_btn",
        )
    else:
        st.info("No data available for export.")
