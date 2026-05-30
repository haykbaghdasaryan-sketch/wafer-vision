"""First-visit tour component for new users.

Shows a dismissible welcome card with a quick feature overview
on the user's first visit. State is stored in session_state.
"""

import streamlit as st


def show_onboarding() -> None:
    """Display the onboarding welcome card for first-time visitors.

    Shows a dismissible card explaining key features of WaferVision.
    Once dismissed, it won't appear again in the current session.
    The dismissed state is stored in `st.session_state['onboarding_dismissed']`.
    """
    if st.session_state.get("onboarding_dismissed", False):
        return

    with st.container():
        st.markdown(
            """
            ## 👋 Welcome to WaferVision!

            WaferVision is a research-grade platform for intelligent wafer defect analysis.
            Here's what you can explore:

            | Page | Description |
            |------|-------------|
            | 📊 **Dashboard** | Dataset overview, best model summary, and training history |
            | 🔍 **Embedding Explorer** | Interactive 3D UMAP/t-SNE visualization of embeddings |
            | 🎯 **Retrieval** | Find similar wafer maps by uploading an image or selecting a sample |
            | 🔬 **Explainability** | Grad-CAM heatmaps showing what the model focuses on |
            | 📈 **Model Comparison** | Side-by-side metrics and radar charts for all models |
            | ⚡ **Anomaly Detection** | Identify out-of-distribution wafer maps with threshold tuning |
            """
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Got it!", key="dismiss_onboarding", type="primary"):
                st.session_state["onboarding_dismissed"] = True
                st.rerun()
        with col2:
            st.caption("This message won't appear again this session.")
