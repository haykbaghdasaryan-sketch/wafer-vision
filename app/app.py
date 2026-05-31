"""Streamlit entry point + session state management.

Main application file for WaferVision. Configures the page, initializes
session state, renders the navigation sidebar, and shows the welcome
banner for first-time visitors.
"""

import sys
import importlib.util
from pathlib import Path

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
_app_dir = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Direct file-based import to avoid namespace conflicts
def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_onboarding = _import_from_file("onboarding", Path(_app_dir) / "onboarding.py")
show_onboarding = _onboarding.show_onboarding

import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="WaferVision",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Session State Initialization ---
def _init_session_state() -> None:
    """Initialize session state with default values if not already set."""
    defaults: dict = {
        # Navigation tracking
        "visited_pages": set(),
        # Model selection
        "selected_model": "resnet50",
        "selected_mode": "finetune",
        # Embedding explorer state
        "dim_reduction_method": "umap",
        "dim_reduction_dims": 3,
        "class_filter": None,
        # Retrieval state
        "retrieval_k": 10,
        "distance_metric": "euclidean",
        # Anomaly detection state
        "anomaly_threshold_sigma": 2.0,
        "anomaly_per_class": False,
        # Onboarding
        "onboarding_dismissed": False,
        # Rate limiter (initialized lazily in rate_limiter module)
        "rate_limiter": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# --- Navigation Sidebar ---
def _render_sidebar() -> None:
    """Render the navigation sidebar with page links and model selector."""
    with st.sidebar:
        st.title("🔬 WaferVision")
        st.markdown("---")

        st.subheader("Navigation")
        pages = [
            ("📊", "Dashboard"),
            ("🔍", "Embedding Explorer"),
            ("🎯", "Retrieval"),
            ("🔬", "Explainability"),
            ("📈", "Model Comparison"),
            ("⚡", "Anomaly Detection"),
        ]
        for icon, name in pages:
            st.page_link(
                f"pages/{pages.index((icon, name)) + 1}_{name.replace(' ', '_')}.py",
                label=f"{icon} {name}",
            )

        st.markdown("---")
        st.subheader("Model Selection")

        st.session_state["selected_model"] = st.selectbox(
            "Backbone",
            options=["resnet50", "efficientnet_b0", "vit_b16"],
            index=["resnet50", "efficientnet_b0", "vit_b16"].index(
                st.session_state["selected_model"]
            ),
            key="sidebar_model_select",
        )

        st.session_state["selected_mode"] = st.selectbox(
            "Training Mode",
            options=["pretrained", "finetune", "metric_triplet", "metric_supcon", "selfsupervised"],
            index=["pretrained", "finetune", "metric_triplet", "metric_supcon", "selfsupervised"].index(
                st.session_state["selected_mode"]
            ),
            key="sidebar_mode_select",
        )

        st.markdown("---")
        st.caption("WaferVision v0.1.0 | Research Platform")


_render_sidebar()


# --- Main Content ---
st.session_state["visited_pages"].add("home")

# Show onboarding for first-time visitors
show_onboarding()

# Home page content when no sub-page is selected
if st.session_state.get("onboarding_dismissed", False) or True:
    st.title("🔬 WaferVision")
    st.markdown(
        """
        **Intelligent Wafer Defect Embedding & Retrieval System**

        Use the sidebar to navigate between pages, or select a page below:
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Dashboard")
        st.markdown("Dataset overview and best model summary")
        st.markdown("### 🔬 Explainability")
        st.markdown("Grad-CAM heatmaps and attention maps")
    with col2:
        st.markdown("### 🔍 Embedding Explorer")
        st.markdown("Interactive 3D embedding visualization")
        st.markdown("### 📈 Model Comparison")
        st.markdown("Side-by-side model evaluation")
    with col3:
        st.markdown("### 🎯 Retrieval")
        st.markdown("Find similar wafer maps")
        st.markdown("### ⚡ Anomaly Detection")
        st.markdown("Identify out-of-distribution samples")
