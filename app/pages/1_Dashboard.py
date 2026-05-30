"""Dashboard page: dataset statistics, best model, UMAP overview."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components import loading_spinner, metric_card
from app.error_handler import safe_execute

# --- Constants ---
OUTPUTS_DIR = Path("outputs")
METRICS_DIR = OUTPUTS_DIR / "metrics"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
DATA_DIR = Path("data") / "processed"

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "None",
]


# --- Data Loading ---
@st.cache_data(ttl=300)
def _load_dataset_stats() -> Optional[dict]:
    """Load dataset statistics from preprocessed splits."""
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        import json
        with open(manifest_path, "r") as f:
            return json.load(f)

    # Try loading from split files directly
    stats = {"total": 0, "labeled": 0, "unlabeled": 0, "per_class": {}}
    for split in ["train", "val", "test"]:
        labels_path = DATA_DIR / f"{split}_labels.pt"
        if labels_path.exists():
            import torch
            labels = torch.load(labels_path, map_location="cpu", weights_only=True)
            labels_np = labels.numpy() if hasattr(labels, "numpy") else np.array(labels)
            stats["total"] += len(labels_np)
            for i, name in enumerate(CLASS_NAMES):
                count = int((labels_np == i).sum())
                stats["per_class"][name] = stats["per_class"].get(name, 0) + count
    return stats if stats["total"] > 0 else None


@st.cache_data(ttl=300)
def _load_metrics_csv() -> Optional[pd.DataFrame]:
    """Load all metrics CSV files from outputs/metrics/."""
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
def _find_best_model(metrics_df: pd.DataFrame) -> Optional[dict]:
    """Find the model with the highest KNN@5 accuracy."""
    knn5 = metrics_df[metrics_df["metric_name"] == "knn_accuracy_k5"]
    if knn5.empty:
        return None

    knn5 = knn5.copy()
    knn5["metric_value"] = pd.to_numeric(knn5["metric_value"], errors="coerce")
    best_row = knn5.loc[knn5["metric_value"].idxmax()]
    return {
        "model_name": best_row["model_name"],
        "training_mode": best_row["training_mode"],
        "knn_accuracy_k5": float(best_row["metric_value"]),
    }


@st.cache_resource(ttl=600)
def _load_umap_projection() -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load pre-computed UMAP projection for the best model."""
    # Look for cached UMAP projections
    cache_dir = OUTPUTS_DIR / "cache" / "umap"
    if not cache_dir.exists():
        return None

    npy_files = list(cache_dir.glob("umap_*.npy"))
    meta_files = list(cache_dir.glob("*_meta.json"))

    if not npy_files:
        return None

    # Load the first available projection
    projection = np.load(npy_files[0])

    # Try loading corresponding labels
    for emb_dir in EMBEDDINGS_DIR.iterdir() if EMBEDDINGS_DIR.exists() else []:
        labels_path = emb_dir / "labels.npy"
        if labels_path.exists():
            labels = np.load(labels_path)
            if len(labels) >= projection.shape[0]:
                return projection, labels[:projection.shape[0]]

    return None


# --- Page Layout ---
st.title("📊 Dashboard")
st.markdown("Overview of dataset, models, and embedding quality.")
st.markdown("---")

# --- Dataset Statistics Section ---
st.subheader("Dataset Statistics")

with safe_execute("loading dataset statistics"):
    stats = _load_dataset_stats()

    if stats is None:
        st.info(
            "📂 No preprocessed data found. Run `make preprocess` to generate "
            "dataset splits and view statistics here."
        )
    else:
        # Summary metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{stats.get('total', 0):,}")
        with col2:
            labeled = stats.get("labeled", sum(stats.get("per_class", {}).values()))
            st.metric("Labeled", f"{labeled:,}")
        with col3:
            unlabeled = stats.get("unlabeled", 0)
            st.metric("Unlabeled", f"{unlabeled:,}")

        # Per-class distribution bar chart
        per_class = stats.get("per_class", {})
        if per_class:
            st.markdown("#### Per-Class Distribution")
            df_class = pd.DataFrame(
                {"Class": list(per_class.keys()), "Count": list(per_class.values())}
            )
            df_class = df_class.sort_values("Count", ascending=False)

            fig = px.bar(
                df_class,
                x="Class",
                y="Count",
                color="Class",
                title="Samples per Defect Class",
                template="plotly_white",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

            # Also show as table
            with st.expander("View as table"):
                st.dataframe(df_class.reset_index(drop=True), use_container_width=True)

st.markdown("---")

# --- Best Model Section ---
st.subheader("🏆 Best Model")

with safe_execute("loading model metrics"):
    metrics_df = _load_metrics_csv()

    if metrics_df is None:
        st.info(
            "🧠 No model metrics found. Train models and run evaluation with "
            "`make train` and `make evaluate` to see results here."
        )
    else:
        best = _find_best_model(metrics_df)
        if best is None:
            st.warning("No KNN@5 metrics found in evaluation results.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", best["model_name"])
            with col2:
                st.metric("Training Mode", best["training_mode"])
            with col3:
                metric_card(
                    label="KNN@5 Accuracy",
                    value=best["knn_accuracy_k5"],
                    help_text="K-Nearest Neighbor accuracy with K=5",
                )

            # Show additional metrics for best model
            best_model_metrics = metrics_df[
                (metrics_df["model_name"] == best["model_name"])
                & (metrics_df["training_mode"] == best["training_mode"])
            ]
            if not best_model_metrics.empty:
                with st.expander("All metrics for best model"):
                    display_df = best_model_metrics[["metric_name", "metric_value"]].copy()
                    display_df.columns = ["Metric", "Value"]
                    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

st.markdown("---")

# --- UMAP Overview Section ---
st.subheader("🗺️ Embedding Space Overview")

with safe_execute("loading UMAP projection"):
    umap_data = _load_umap_projection()

    if umap_data is None:
        st.info(
            "🔍 No pre-computed UMAP projection available. Extract embeddings with "
            "`make extract` and the visualization will appear here automatically."
        )
    else:
        projection, labels = umap_data
        is_3d = projection.shape[1] == 3

        if is_3d:
            fig = go.Figure()
            for i, name in enumerate(CLASS_NAMES):
                mask = labels == i
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter3d(
                    x=projection[mask, 0],
                    y=projection[mask, 1],
                    z=projection[mask, 2],
                    mode="markers",
                    marker=dict(size=2, opacity=0.6),
                    name=name,
                ))
            fig.update_layout(
                title="UMAP 3D Projection (Best Model)",
                template="plotly_white",
                height=500,
            )
        else:
            fig = go.Figure()
            for i, name in enumerate(CLASS_NAMES):
                mask = labels == i
                if not mask.any():
                    continue
                fig.add_trace(go.Scattergl(
                    x=projection[mask, 0],
                    y=projection[mask, 1],
                    mode="markers",
                    marker=dict(size=3, opacity=0.6),
                    name=name,
                ))
            fig.update_layout(
                title="UMAP 2D Projection (Best Model)",
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                template="plotly_white",
                height=500,
            )

        st.plotly_chart(fig, use_container_width=True)
