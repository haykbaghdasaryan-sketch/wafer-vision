import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

try:
    from app.error_handler import safe_execute
    from app.components import loading_spinner, wafer_card, wafer_gallery, metric_card, metric_table, radar_chart
    from app.rate_limiter import SessionRateLimiter
except (ImportError, ModuleNotFoundError):
    from contextlib import contextmanager
    @contextmanager
    def safe_execute(*a, **kw): yield
    def loading_spinner(*a, **kw): return safe_execute()
    def wafer_card(*a, **kw): pass
    def wafer_gallery(*a, **kw): pass
    def metric_card(*a, **kw): pass
    def metric_table(*a, **kw): pass
    def radar_chart(*a, **kw): pass
    class SessionRateLimiter:
        def __init__(self, **kw): self.timestamps = []
        def check(self, **kw): return True
        def wait_message(self, **kw): return ''

"""Anomaly Detection page: threshold slider, distance histogram, flagged gallery."""

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# --- Constants ---
OUTPUTS_DIR = Path("outputs")
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "None",
]

MODELS = ["resnet50", "efficientnet_b0", "vit_b16"]
MODES = ["pretrained", "finetune", "metric_triplet", "metric_supcon", "selfsupervised"]


# --- Data Loading ---
@st.cache_resource(ttl=600)
def _load_embeddings(model: str, mode: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load embeddings and labels for a model/mode."""
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


@st.cache_data(ttl=300)
def _compute_distances(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distances to nearest centroid for all samples.

    Returns:
        (distances, nearest_classes, centroids)
    """
    from src.anomaly.detector import AnomalyDetector

    detector = AnomalyDetector(embeddings, labels, class_names=CLASS_NAMES)
    distances = detector.score(embeddings)
    # Determine nearest class for each sample
    all_dists = detector._compute_all_distances(embeddings)
    nearest_classes = np.argmin(all_dists, axis=1)
    return distances, nearest_classes, detector.centroids


# --- Page Layout ---
st.title("⚡ Anomaly Detection")
st.markdown(
    "Identify out-of-distribution wafer maps using embedding distance to class centroids."
)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Anomaly Settings")

    selected_model = st.selectbox(
        "Model",
        options=MODELS,
        index=MODELS.index(st.session_state.get("selected_model", "resnet50")),
        key="anomaly_model",
    )
    selected_mode = st.selectbox(
        "Training Mode",
        options=MODES,
        index=MODES.index(st.session_state.get("selected_mode", "finetune")),
        key="anomaly_mode",
    )

# --- Load Embeddings ---
with safe_execute("loading embeddings"):
    data = _load_embeddings(selected_model, selected_mode)

    if data is None:
        st.info(
            f"📂 No embeddings found for **{selected_model}** / **{selected_mode}**.\n\n"
            "Run `make extract` to generate embeddings for anomaly detection."
        )
        st.stop()

    embeddings, labels = data
    st.caption(f"Analyzing {len(embeddings):,} embeddings (dim={embeddings.shape[1]})")

# --- Compute Distances ---
with safe_execute("computing distances"):
    with loading_spinner("anomaly"):
        distances, nearest_classes, centroids = _compute_distances(embeddings, labels)

# --- Threshold Controls ---
st.markdown("---")

col_slider, col_toggle = st.columns([3, 1])

with col_slider:
    threshold_sigma = st.slider(
        "Threshold (σ)",
        min_value=1.0,
        max_value=5.0,
        value=st.session_state.get("anomaly_threshold_sigma", 2.0),
        step=0.1,
        key="anomaly_threshold_slider",
        help="Number of standard deviations above mean. Lower → more anomalies flagged.",
    )

with col_toggle:
    per_class = st.checkbox(
        "Per-class threshold",
        value=st.session_state.get("anomaly_per_class", False),
        key="anomaly_per_class_toggle",
        help="Compute separate threshold for each class instead of global.",
    )

# --- Compute Threshold and Flag ---
with safe_execute("applying threshold"):
    from src.anomaly.threshold import compute_threshold, compute_per_class_thresholds

    if per_class:
        class_thresholds = compute_per_class_thresholds(distances, nearest_classes, n_sigma=threshold_sigma)
        # Flag per-class
        flagged_mask = np.zeros(len(distances), dtype=bool)
        for cls_idx, thresh in class_thresholds.items():
            cls_mask = nearest_classes == cls_idx
            flagged_mask[cls_mask] = distances[cls_mask] > thresh
        global_threshold = np.mean(list(class_thresholds.values()))
    else:
        global_threshold = compute_threshold(distances, n_sigma=threshold_sigma)
        flagged_mask = distances > global_threshold

    n_flagged = int(flagged_mask.sum())
    anomaly_rate = n_flagged / len(distances) * 100

# --- Summary Statistics ---
st.markdown("---")
st.subheader("Summary Statistics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", f"{len(distances):,}")
with col2:
    st.metric("Flagged Anomalies", f"{n_flagged:,}")
with col3:
    st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
with col4:
    st.metric("Threshold (σ)", f"{threshold_sigma:.1f}σ")

# Per-class breakdown
if n_flagged > 0:
    with st.expander("Per-Class Breakdown"):
        class_counts = {}
        for cls_idx in range(len(CLASS_NAMES)):
            cls_flagged = int((flagged_mask & (nearest_classes == cls_idx)).sum())
            cls_total = int((nearest_classes == cls_idx).sum())
            if cls_total > 0:
                class_counts[CLASS_NAMES[cls_idx]] = {
                    "Flagged": cls_flagged,
                    "Total": cls_total,
                    "Rate": f"{cls_flagged / cls_total * 100:.1f}%",
                }

        if class_counts:
            import pandas as pd
            breakdown_df = pd.DataFrame(class_counts).T
            breakdown_df.index.name = "Class"
            st.dataframe(breakdown_df, use_container_width=True)

# --- Distance Histogram ---
st.markdown("---")
st.subheader("📊 Distance Distribution")

with safe_execute("generating histogram"):
    fig = go.Figure()

    # Histogram of distances (50 bins)
    fig.add_trace(go.Histogram(
        x=distances,
        nbinsx=50,
        name="Distance Distribution",
        marker_color="#4477AA",
        opacity=0.7,
    ))

    # Threshold line
    fig.add_vline(
        x=global_threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Threshold ({threshold_sigma:.1f}σ) = {global_threshold:.3f}",
        annotation_position="top right",
    )

    # Anomaly count annotation
    fig.add_annotation(
        x=global_threshold + (distances.max() - global_threshold) * 0.3,
        y=0,
        text=f"Anomalies: {n_flagged}",
        showarrow=False,
        yshift=20,
        font=dict(color="red", size=12),
    )

    fig.update_layout(
        title="Distance to Nearest Centroid",
        xaxis_title="Distance",
        yaxis_title="Count",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Flagged Anomaly Gallery ---
st.markdown("---")
st.subheader("🚨 Flagged Anomalies")

if n_flagged == 0:
    st.success(
        f"✅ No anomalies detected at {threshold_sigma:.1f}σ threshold. "
        "Try lowering the threshold to flag more samples."
    )
else:
    # Sort flagged by distance descending
    flagged_indices = np.where(flagged_mask)[0]
    flagged_distances = distances[flagged_indices]
    sort_order = np.argsort(flagged_distances)[::-1]
    flagged_indices = flagged_indices[sort_order]
    flagged_distances = flagged_distances[sort_order]

    # Pagination
    items_per_page = 20
    total_pages = (n_flagged + items_per_page - 1) // items_per_page
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=max(1, total_pages),
        value=1,
        key="anomaly_page",
    )

    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, n_flagged)

    st.caption(f"Showing {start_idx + 1}–{end_idx} of {n_flagged} anomalies (sorted by distance, descending)")

    # Display as table
    rows = []
    for i in range(start_idx, end_idx):
        sample_idx = int(flagged_indices[i])
        dist = float(flagged_distances[i])
        cls_idx = int(nearest_classes[sample_idx])
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)

        # Confidence: 1 - (threshold / distance)
        sample_threshold = (
            class_thresholds.get(cls_idx, global_threshold) if per_class else global_threshold
        )
        confidence = max(0.0, min(1.0, 1.0 - (sample_threshold / dist))) if dist > 0 else 0.0

        rows.append({
            "Rank": i + 1,
            "Index": sample_idx,
            "Distance": f"{dist:.4f}",
            "Class": cls_name,
            "Confidence": f"{confidence:.2%}",
        })

    if rows:
        import pandas as pd
        anomaly_df = pd.DataFrame(rows)
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)

    # Compact gallery view
    with st.expander("Gallery View (top 10)"):
        n_show = min(10, end_idx - start_idx)
        n_cols = 5
        for row_start in range(0, n_show, n_cols):
            cols = st.columns(n_cols)
            for col_offset, col in enumerate(cols):
                i = start_idx + row_start + col_offset
                if i >= end_idx:
                    break
                with col:
                    sample_idx = int(flagged_indices[i])
                    dist = float(flagged_distances[i])
                    cls_idx = int(nearest_classes[sample_idx])
                    cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
                    st.markdown(f"**#{sample_idx}**")
                    st.caption(f"{cls_name}\ndist: {dist:.3f}")
