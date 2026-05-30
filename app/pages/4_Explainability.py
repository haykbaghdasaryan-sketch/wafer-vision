import app._path_fix  # noqa: F401
"""Explainability page: Grad-CAM heatmaps and attention rollout."""

from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

from app.components import loading_spinner
from app.error_handler import safe_execute

# --- Constants ---
OUTPUTS_DIR = Path("outputs")
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
DATA_DIR = Path("data") / "processed"

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "None",
]

MODELS = ["resnet50", "efficientnet_b0", "vit_b16"]
MODES = ["pretrained", "finetune", "metric_triplet", "metric_supcon", "selfsupervised"]


# --- Data Loading ---
@st.cache_data(ttl=600)
def _load_dataset_samples() -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load preprocessed dataset samples and labels."""
    for split in ["test", "val", "train"]:
        data_path = DATA_DIR / f"{split}_data.pt"
        labels_path = DATA_DIR / f"{split}_labels.pt"
        if data_path.exists() and labels_path.exists():
            import torch
            data = torch.load(data_path, map_location="cpu", weights_only=True)
            labels = torch.load(labels_path, map_location="cpu", weights_only=True)
            if hasattr(data, "numpy"):
                data = data.numpy()
            if hasattr(labels, "numpy"):
                labels = labels.numpy()
            return np.array(data), np.array(labels)
    return None


@st.cache_resource(ttl=600)
def _load_model(model_name: str, mode: str):
    """Load a backbone model from checkpoint."""
    try:
        import torch
        from src.models.factory import get_backbone

        model = get_backbone(model_name, pretrained=(mode == "pretrained"))
        model.eval()

        # Try loading fine-tuned checkpoint
        if mode != "pretrained":
            ckpt_path = CHECKPOINTS_DIR / f"{model_name}_{mode}" / "checkpoint_best.pth"
            if not ckpt_path.exists():
                ckpt_path = CHECKPOINTS_DIR / "checkpoint_best.pth"
            if ckpt_path.exists():
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return model
    except Exception:
        return None


def _compute_heatmap(model, image_tensor) -> Optional[np.ndarray]:
    """Compute Grad-CAM or Attention Rollout heatmap."""
    try:
        import torch
        from src.visualization.gradcam import GradCAMVisualizer

        visualizer = GradCAMVisualizer(model)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        heatmap = visualizer.compute_explanation(image_tensor)
        return heatmap
    except Exception:
        return None


# --- Page Layout ---
st.title("🔬 Explainability")
st.markdown(
    "Visualize what regions of a wafer map the model focuses on using "
    "Grad-CAM (CNN) or Attention Rollout (ViT)."
)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Explainability Settings")

    selected_model = st.selectbox(
        "Model",
        options=MODELS,
        index=MODELS.index(st.session_state.get("selected_model", "resnet50")),
        key="explain_model",
    )
    selected_mode = st.selectbox(
        "Training Mode",
        options=MODES,
        index=MODES.index(st.session_state.get("selected_mode", "pretrained")),
        key="explain_mode",
    )

    st.markdown("---")

    opacity = st.slider(
        "Overlay Opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        key="explain_opacity",
        help="Blend between wafer map (0) and heatmap (1).",
    )

# --- Load Data ---
with safe_execute("loading dataset"):
    dataset = _load_dataset_samples()

    if dataset is None:
        st.info(
            "📂 No preprocessed data found. Run `make preprocess` to generate "
            "dataset splits for explainability analysis."
        )
        st.stop()

    data_samples, data_labels = dataset
    n_samples = len(data_labels)

# --- Wafer Selection ---
st.markdown("---")
st.subheader("Select Wafer Map")

select_tab1, select_tab2, select_tab3 = st.tabs(["🔢 By Index", "🎲 Random", "📋 Class Representative"])

selected_idx: Optional[int] = None

with select_tab1:
    idx_input = st.number_input(
        "Sample Index",
        min_value=0,
        max_value=n_samples - 1,
        value=0,
        step=1,
        key="explain_idx",
    )
    if st.button("Select", key="explain_select_idx"):
        selected_idx = int(idx_input)
        st.session_state["explain_selected_idx"] = selected_idx

with select_tab2:
    if st.button("🎲 Pick Random", key="explain_random"):
        rng = np.random.default_rng()
        selected_idx = int(rng.integers(0, n_samples))
        st.session_state["explain_selected_idx"] = selected_idx

with select_tab3:
    class_choice = st.selectbox(
        "Select Class",
        options=CLASS_NAMES,
        key="explain_class_choice",
    )
    class_idx = CLASS_NAMES.index(class_choice)
    class_mask = data_labels == class_idx
    class_indices = np.where(class_mask)[0]
    if len(class_indices) > 0:
        if st.button(f"Pick representative from '{class_choice}'", key="explain_class_rep"):
            # Pick the sample closest to the class centroid (median index as proxy)
            rep_idx = int(class_indices[len(class_indices) // 2])
            selected_idx = rep_idx
            st.session_state["explain_selected_idx"] = selected_idx
    else:
        st.info(f"No samples found for class '{class_choice}'.")

# Restore from session state
if selected_idx is None:
    selected_idx = st.session_state.get("explain_selected_idx")

if selected_idx is None:
    st.info("👆 Select a wafer map above to generate explanations.")
    st.stop()

# --- Display Selected Wafer ---
sample_label = int(data_labels[selected_idx])
cls_name = CLASS_NAMES[sample_label] if sample_label < len(CLASS_NAMES) else str(sample_label)
st.markdown(f"**Selected:** Index {selected_idx} | Class: {cls_name}")

# --- Generate Explanation ---
st.markdown("---")
st.subheader("Explanation")

# Pretrained mode warning
if selected_mode == "pretrained":
    st.warning(
        "⚠️ **Pretrained Model Notice:** The model has not been fine-tuned on wafer data. "
        "Attention maps may not highlight wafer-specific features."
    )

with safe_execute("loading model"):
    model = _load_model(selected_model, selected_mode)

if model is None:
    st.error(
        f"Could not load model **{selected_model}** ({selected_mode}). "
        "Make sure the checkpoint exists in `outputs/checkpoints/`."
    )
    st.stop()

# Prepare image tensor
with safe_execute("computing explanation"):
    import torch

    sample = data_samples[selected_idx]
    # Ensure proper tensor shape (3, H, W) and float32
    if sample.ndim == 2:
        # Raw wafer map → convert to 3-channel float
        sample_float = sample.astype(np.float32) / 2.0  # normalize {0,1,2} to [0,1]
        sample_3ch = np.stack([sample_float] * 3, axis=0)
    elif sample.ndim == 3 and sample.shape[0] == 3:
        sample_3ch = sample.astype(np.float32)
    elif sample.ndim == 3 and sample.shape[-1] == 3:
        sample_3ch = sample.astype(np.float32).transpose(2, 0, 1)
    else:
        sample_3ch = sample.astype(np.float32)
        if sample_3ch.ndim == 2:
            sample_3ch = np.stack([sample_3ch] * 3, axis=0)

    image_tensor = torch.from_numpy(sample_3ch).unsqueeze(0)

    # Show progress for potentially long computation
    progress_placeholder = st.empty()
    progress_placeholder.info("⏳ Computing heatmap...")

    heatmap = _compute_heatmap(model, image_tensor)
    progress_placeholder.empty()

    if heatmap is None:
        st.error("Failed to compute heatmap. The model may not support this operation.")
        st.stop()

    # --- Display Three Images ---
    col1, col2, col3 = st.columns(3)

    # Get spatial wafer map for overlay
    if sample.ndim == 2:
        wafer_2d = sample
    elif sample.ndim == 3 and sample.shape[0] == 3:
        wafer_2d = sample[0]  # Take first channel as proxy
    else:
        wafer_2d = sample[:, :, 0] if sample.ndim == 3 else sample

    # Colorize wafer map
    from app.components.wafer_card import colorize_wafer

    with col1:
        st.markdown("**Original (Colorized)**")
        # Create a viewable version
        wafer_int = np.clip(wafer_2d, 0, 2).astype(int)
        rgb_original = colorize_wafer(wafer_int)
        st.image(rgb_original, width=200)

    with col2:
        st.markdown("**Heatmap (Jet)**")
        import cv2
        # Resize heatmap to match wafer dimensions
        h, w = wafer_2d.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, width=200)

    with col3:
        st.markdown(f"**Overlay (α={opacity:.1f})**")
        from src.visualization.gradcam import GradCAMVisualizer
        overlay = GradCAMVisualizer.create_overlay(wafer_int, heatmap, alpha=opacity)
        st.image(overlay, width=200)

# --- Compare All Models Grid ---
st.markdown("---")
st.subheader("Compare All Models")

compare_all = st.checkbox("Show comparison across all models", key="explain_compare_all")

if compare_all:
    with safe_execute("comparing models"):
        progress = st.progress(0)
        total_models = len(MODELS)
        results = {}

        for i, m_name in enumerate(MODELS):
            progress.progress((i + 1) / total_models, text=f"Computing {m_name}...")
            m = _load_model(m_name, selected_mode)
            if m is not None:
                hm = _compute_heatmap(m, image_tensor)
                if hm is not None:
                    results[m_name] = hm

        progress.empty()

        if results:
            cols = st.columns(len(results))
            for col, (m_name, hm) in zip(cols, results.items()):
                with col:
                    st.markdown(f"**{m_name}**")
                    overlay_img = GradCAMVisualizer.create_overlay(wafer_int, hm, alpha=opacity)
                    st.image(overlay_img, width=180)
        else:
            st.warning("Could not load any models for comparison.")
