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

"""Retrieval page: query input, top-K gallery, model comparison."""

import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st


# --- Constants ---
OUTPUTS_DIR = Path("outputs")
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
DATA_DIR = Path("data") / "processed"

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "None",
]

MODELS = ["resnet50", "efficientnet_b0", "vit_b16"]
MODES = ["pretrained", "finetune", "metric_triplet", "metric_supcon", "selfsupervised"]

MAX_UPLOAD_SIZE_MB = 10.0
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# --- Session State ---
def _get_rate_limiter() -> SessionRateLimiter:
    """Get or create the session rate limiter."""
    if st.session_state.get("rate_limiter") is None:
        st.session_state["rate_limiter"] = SessionRateLimiter(
            max_requests=10, window_seconds=60
        )
    return st.session_state["rate_limiter"]


# --- Data Loading ---
@st.cache_resource(ttl=600)
def _load_embeddings(model: str, mode: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load embeddings and labels for a model/mode."""
    emb_dir = EMBEDDINGS_DIR / f"{model}_{mode}"
    emb_path = emb_dir / "embeddings.npy"
    labels_path = emb_dir / "labels.npy"

    if not emb_path.exists():
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


def _validate_upload(file_bytes: bytes, filename: str) -> Optional[str]:
    """Validate uploaded file. Returns error message or None if valid."""
    # Check size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_UPLOAD_SIZE_MB} MB."

    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}."

    # Try to decode as image
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        if w > 1024 or h > 1024:
            return f"Image dimensions ({w}×{h}) exceed maximum (1024×1024)."
        if w < 5 or h < 5:
            return f"Image too small ({w}×{h}). Minimum: 5×5 pixels."
    except Exception:
        return "Could not decode file as a valid image."

    return None


def _preprocess_upload(file_bytes: bytes) -> np.ndarray:
    """Preprocess an uploaded image to model input format."""
    from PIL import Image
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((96, 96), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    # Convert to (3, H, W) and normalize
    arr = arr.transpose(2, 0, 1)
    return arr


# --- Page Layout ---
st.title("🎯 Retrieval")
st.markdown("Find similar wafer maps using embedding-based nearest neighbor search.")

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Retrieval Settings")

    selected_model = st.selectbox(
        "Model",
        options=MODELS,
        index=MODELS.index(st.session_state.get("selected_model", "resnet50")),
        key="retrieval_model",
    )
    selected_mode = st.selectbox(
        "Training Mode",
        options=MODES,
        index=MODES.index(st.session_state.get("selected_mode", "finetune")),
        key="retrieval_mode",
    )

    st.markdown("---")

    k_value = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=50,
        value=st.session_state.get("retrieval_k", 10),
        key="retrieval_k_slider",
    )

    distance_metric = st.selectbox(
        "Distance Metric",
        options=["euclidean", "cosine"],
        index=0,
        key="retrieval_distance",
    )

# --- Load Embeddings ---
with safe_execute("loading embeddings"):
    data = _load_embeddings(selected_model, selected_mode)

    if data is None:
        st.info(
            f"📂 No embeddings found for **{selected_model}** / **{selected_mode}**.\n\n"
            "Run `make extract` to generate embeddings first."
        )
        st.stop()

    embeddings, labels = data
    st.caption(f"Database: {len(embeddings):,} embeddings (dim={embeddings.shape[1]})")

# --- Query Input ---
st.markdown("---")
st.subheader("Query Input")

query_tab1, query_tab2, query_tab3 = st.tabs(["📤 Upload Image", "🎲 Random Sample", "🔢 Dataset Index"])

query_embedding: Optional[np.ndarray] = None
query_index: Optional[int] = None
query_label: Optional[int] = None

with query_tab1:
    uploaded_file = st.file_uploader(
        "Upload a wafer map image (PNG/JPEG, ≤ 10 MB)",
        type=["png", "jpg", "jpeg"],
        key="retrieval_upload",
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        error_msg = _validate_upload(file_bytes, uploaded_file.name)
        if error_msg:
            st.error(f"⚠️ {error_msg}")
        else:
            st.image(file_bytes, caption="Uploaded query image", width=150)
            st.info(
                "Note: For uploaded images, a model forward pass is needed to compute "
                "the embedding. This requires the model checkpoint to be loaded."
            )
            # In practice, we'd run the image through the model here
            # For now, show placeholder message
            st.warning(
                "Image-to-embedding inference is available when model checkpoints are loaded. "
                "Use 'Dataset Index' or 'Random Sample' for pre-computed embedding retrieval."
            )

with query_tab2:
    if st.button("🎲 Pick Random Sample", key="retrieval_random"):
        rng = np.random.default_rng()
        query_index = int(rng.integers(0, len(embeddings)))
        query_label = int(labels[query_index])
        st.session_state["retrieval_query_idx"] = query_index

    # Show previously selected random sample
    if "retrieval_query_idx" in st.session_state and query_index is None:
        query_index = st.session_state["retrieval_query_idx"]
        query_label = int(labels[query_index])

    if query_index is not None:
        st.success(
            f"Selected sample **#{query_index}** — "
            f"Class: **{CLASS_NAMES[query_label] if query_label < len(CLASS_NAMES) else query_label}**"
        )

with query_tab3:
    idx_input = st.number_input(
        "Dataset Index",
        min_value=0,
        max_value=len(embeddings) - 1,
        value=0,
        step=1,
        key="retrieval_idx_input",
    )
    if st.button("🔍 Search by Index", key="retrieval_search_btn"):
        query_index = int(idx_input)
        query_label = int(labels[query_index])
        st.session_state["retrieval_query_idx"] = query_index

    if query_index is None and "retrieval_query_idx" in st.session_state:
        query_index = st.session_state.get("retrieval_query_idx")
        if query_index is not None:
            query_label = int(labels[query_index])

# --- Perform Retrieval ---
if query_index is not None:
    st.markdown("---")
    st.subheader("Retrieval Results")

    # Rate limiting
    rate_limiter = _get_rate_limiter()
    if not rate_limiter.check():
        wait_msg = rate_limiter.wait_message()
        st.warning(f"⏳ {wait_msg}")
        st.stop()

    with safe_execute("performing retrieval"):
        from src.evaluation.retrieval import RetrievalEngine

        t_start = time.perf_counter()

        engine = RetrievalEngine(
            embeddings=embeddings,
            labels=labels,
            distance_metric=distance_metric,
        )
        result = engine.query_by_index(query_index, k=k_value)

        latency_ms = (time.perf_counter() - t_start) * 1000

        if len(result.indices) == 0:
            st.warning("No results found.")
        else:
            # Precision@K
            matches = (result.labels == query_label).sum()
            precision = matches / len(result.labels)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision@K", f"{precision:.2%}")
            with col2:
                st.metric("Retrieval Latency", f"{latency_ms:.1f} ms")
            with col3:
                st.metric("Results", f"{len(result.indices)}")

            # Results gallery
            st.markdown("#### Top-K Neighbors")
            n_cols = min(5, len(result.indices))
            rows_needed = (len(result.indices) + n_cols - 1) // n_cols

            for row in range(rows_needed):
                cols = st.columns(n_cols)
                for col_idx, col in enumerate(cols):
                    i = row * n_cols + col_idx
                    if i >= len(result.indices):
                        break
                    with col:
                        neighbor_idx = int(result.indices[i])
                        neighbor_label = int(result.labels[i])
                        neighbor_dist = float(result.distances[i])
                        cls_name = (
                            CLASS_NAMES[neighbor_label]
                            if neighbor_label < len(CLASS_NAMES)
                            else str(neighbor_label)
                        )
                        is_match = neighbor_label == query_label
                        match_icon = "✓" if is_match else "✗"
                        match_color = "green" if is_match else "red"

                        st.markdown(f"**#{i+1}** — idx: {neighbor_idx}")
                        st.markdown(
                            f":{match_color}[{match_icon}] {cls_name} "
                            f"| dist: {neighbor_dist:.4f}"
                        )
