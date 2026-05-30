"""Embedding Explorer page: interactive 3D UMAP/t-SNE scatter."""

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.components import loading_spinner, wafer_card, wafer_gallery
from app.error_handler import safe_execute

# --- Constants ---
OUTPUTS_DIR = Path("outputs")
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
DATA_DIR = Path("data") / "processed"

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "None",
]

CLASS_COLORS = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB", "#EE8866", "#44BB99",
]

MODELS = ["resnet50", "efficientnet_b0", "vit_b16"]
MODES = ["pretrained", "finetune", "metric_triplet", "metric_supcon", "selfsupervised"]


# --- Data Loading ---
@st.cache_resource(ttl=600)
def _load_embeddings(model: str, mode: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load embeddings and labels for a given model/mode combination."""
    emb_dir = EMBEDDINGS_DIR / f"{model}_{mode}"
    emb_path = emb_dir / "embeddings.npy"
    labels_path = emb_dir / "labels.npy"

    if not emb_path.exists():
        # Try alternative naming
        for d in EMBEDDINGS_DIR.iterdir() if EMBEDDINGS_DIR.exists() else []:
            if model in d.name and mode in d.name:
                ep = d / "embeddings.npy"
                lp = d / "labels.npy"
                if ep.exists() and lp.exists():
                    return np.load(ep), np.load(lp)
        return None

    if not labels_path.exists():
        return None

    return np.load(emb_path), np.load(labels_path)


@st.cache_resource(ttl=600)
def _compute_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str,
    n_components: int,
) -> np.ndarray:
    """Compute dimensionality reduction projection."""
    if method == "umap":
        from src.visualization.umap_viz import UMAPVisualizer
        viz = UMAPVisualizer(
            n_components=n_components,
            cache_dir=str(OUTPUTS_DIR / "cache" / "umap"),
        )
        return viz.fit_transform(embeddings, labels)
    else:
        from src.visualization.tsne_viz import TSNEVisualizer
        viz = TSNEVisualizer(n_components=n_components)
        return viz.fit_transform(embeddings)


@st.cache_data(ttl=600)
def _load_wafer_maps() -> Optional[np.ndarray]:
    """Load raw wafer maps from preprocessed data."""
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}_data.pt"
        if path.exists():
            import torch
            data = torch.load(path, map_location="cpu", weights_only=True)
            if hasattr(data, "numpy"):
                return data.numpy()
            return np.array(data)
    return None


# --- Page Layout ---
st.title("🔍 Embedding Explorer")
st.markdown("Interactive visualization of learned embedding spaces.")

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Explorer Settings")

    selected_model = st.selectbox(
        "Model",
        options=MODELS,
        index=MODELS.index(st.session_state.get("selected_model", "resnet50")),
        key="explorer_model",
    )
    selected_mode = st.selectbox(
        "Training Mode",
        options=MODES,
        index=MODES.index(st.session_state.get("selected_mode", "pretrained")),
        key="explorer_mode",
    )

    st.markdown("---")

    method = st.radio(
        "Reduction Method",
        options=["umap", "tsne"],
        format_func=lambda x: "UMAP" if x == "umap" else "t-SNE",
        key="explorer_method",
    )

    n_dims = st.radio(
        "Dimensions",
        options=[2, 3],
        format_func=lambda x: f"{x}D",
        horizontal=True,
        key="explorer_dims",
    )

    st.markdown("---")
    st.markdown("### Class Filter")
    selected_classes = []
    for i, name in enumerate(CLASS_NAMES):
        if st.checkbox(name, value=True, key=f"class_filter_{i}"):
            selected_classes.append(i)

    st.markdown("---")
    search_index = st.number_input(
        "Search by Index",
        min_value=0,
        value=0,
        step=1,
        key="explorer_search_idx",
    )

# --- Main Content ---
with safe_execute("loading embeddings"):
    data = _load_embeddings(selected_model, selected_mode)

    if data is None:
        st.info(
            f"📂 No embeddings found for **{selected_model}** / **{selected_mode}**.\n\n"
            "Run `make extract` to generate embeddings, then return here to explore them."
        )
        st.stop()

    embeddings, labels = data
    st.caption(f"Loaded {len(embeddings):,} embeddings (dim={embeddings.shape[1]})")

# --- Compute Projection ---
with safe_execute("computing projection"):
    with loading_spinner(method):
        projection = _compute_projection(
            embeddings, labels, method, n_dims
        )

# --- Apply Class Filter ---
if selected_classes and len(selected_classes) < len(CLASS_NAMES):
    mask = np.isin(labels, selected_classes)
    filtered_proj = projection[mask]
    filtered_labels = labels[mask]
    st.caption(f"Showing {len(filtered_labels):,} points ({len(selected_classes)} classes)")
else:
    filtered_proj = projection
    filtered_labels = labels

# --- Create Scatter Plot ---
with safe_execute("creating visualization"):
    is_3d = n_dims == 3
    fig = go.Figure()

    for i in selected_classes:
        cls_mask = filtered_labels == i
        if not cls_mask.any():
            continue
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
        color = CLASS_COLORS[i % len(CLASS_COLORS)]

        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=filtered_proj[cls_mask, 0],
                y=filtered_proj[cls_mask, 1],
                z=filtered_proj[cls_mask, 2],
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.7),
                name=name,
                hovertemplate=(
                    f"Class: {name}<br>"
                    "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}"
                    "<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Scattergl(
                x=filtered_proj[cls_mask, 0],
                y=filtered_proj[cls_mask, 1],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.7),
                name=name,
                hovertemplate=(
                    f"Class: {name}<br>"
                    "x: %{x:.2f}<br>y: %{y:.2f}"
                    "<extra></extra>"
                ),
            ))

    method_name = "UMAP" if method == "umap" else "t-SNE"
    fig.update_layout(
        title=f"{method_name} {'3D' if is_3d else '2D'} — {selected_model} ({selected_mode})",
        template="plotly_white",
        height=600,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
    )

    if not is_3d:
        fig.update_layout(
            xaxis_title=f"{method_name} 1",
            yaxis_title=f"{method_name} 2",
        )

    st.plotly_chart(fig, use_container_width=True, key="embedding_scatter")

# --- Point Click / Search by Index ---
st.markdown("---")
st.subheader("🔎 Point Inspector")

with safe_execute("inspecting point"):
    idx = int(search_index)
    if idx >= len(embeddings):
        st.warning(f"Index {idx} is out of range (max: {len(embeddings) - 1}).")
    else:
        query_label = int(labels[idx])
        st.markdown(
            f"**Index:** {idx} | **Class:** {CLASS_NAMES[query_label] if query_label < len(CLASS_NAMES) else query_label}"
        )

        # Show nearest neighbors
        from src.evaluation.retrieval import RetrievalEngine

        engine = RetrievalEngine(
            embeddings=embeddings,
            labels=labels,
            distance_metric="euclidean",
        )
        result = engine.query_by_index(idx, k=5)

        if len(result.indices) > 0:
            st.markdown("**5 Nearest Neighbors:**")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(result.indices):
                    neighbor_idx = int(result.indices[i])
                    neighbor_label = int(result.labels[i])
                    neighbor_dist = float(result.distances[i])
                    cls_name = CLASS_NAMES[neighbor_label] if neighbor_label < len(CLASS_NAMES) else str(neighbor_label)
                    match = "✓" if neighbor_label == query_label else "✗"
                    with col:
                        st.markdown(f"**#{neighbor_idx}**")
                        st.caption(f"{cls_name} {match}\ndist: {neighbor_dist:.4f}")
