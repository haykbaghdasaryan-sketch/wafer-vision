"""UMAPVisualizer with 2D/3D support, disk caching, and interactive Plotly visualization."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from umap import UMAP

logger = logging.getLogger(__name__)

# Colorblind-friendly palette for 9 wafer defect classes
CLASS_COLORS = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
    "#EE8866",  # orange
    "#44BB99",  # teal
]

DEFAULT_CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "None",
]


class UMAPVisualizer:
    """UMAP dimensionality reduction visualizer with caching and interactive Plotly output.

    Supports 2D and 3D projections, stratified subsampling for large datasets,
    disk caching based on parameter hashing, and class filtering without recomputation.

    Args:
        n_components: Number of output dimensions (2 or 3).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        metric: Distance metric for UMAP ('euclidean' or 'cosine').
        random_state: Random seed for reproducibility.
        cache_dir: Directory for caching projections. None disables caching.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        if n_components not in (2, 3):
            raise ValueError(f"n_components must be 2 or 3, got {n_components}")
        if n_neighbors < 2:
            raise ValueError(f"n_neighbors must be >= 2, got {n_neighbors}")
        if not (0.0 <= min_dist <= 1.0):
            raise ValueError(f"min_dist must be in [0.0, 1.0], got {min_dist}")

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_params_hash(self, embeddings: np.ndarray) -> str:
        """Compute a hash based on UMAP parameters and embedding shape for cache key."""
        params_str = (
            f"n_components={self.n_components},"
            f"n_neighbors={self.n_neighbors},"
            f"min_dist={self.min_dist},"
            f"metric={self.metric},"
            f"random_state={self.random_state},"
            f"shape={embeddings.shape},"
            f"dtype={embeddings.dtype}"
        )
        return hashlib.md5(params_str.encode()).hexdigest()

    def _get_cache_path(self, params_hash: str) -> tuple[Path, Path]:
        """Get the paths for cached projection .npy and metadata .json files."""
        assert self.cache_dir is not None
        npy_path = self.cache_dir / f"umap_{params_hash}.npy"
        meta_path = self.cache_dir / f"umap_{params_hash}_meta.json"
        return npy_path, meta_path

    def _load_cache(self, params_hash: str) -> Optional[np.ndarray]:
        """Attempt to load cached projection from disk."""
        if self.cache_dir is None:
            return None

        npy_path, meta_path = self._get_cache_path(params_hash)

        if npy_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                # Validate metadata matches current params
                if (
                    meta.get("n_components") == self.n_components
                    and meta.get("n_neighbors") == self.n_neighbors
                    and meta.get("min_dist") == self.min_dist
                    and meta.get("metric") == self.metric
                    and meta.get("random_state") == self.random_state
                ):
                    projection = np.load(npy_path)
                    logger.info(f"Loaded cached UMAP projection from {npy_path}")
                    return projection
            except (json.JSONDecodeError, ValueError, OSError) as e:
                logger.warning(f"Failed to load cache: {e}")

        return None

    def _save_cache(self, params_hash: str, projection: np.ndarray) -> None:
        """Save projection to disk cache."""
        if self.cache_dir is None:
            return

        npy_path, meta_path = self._get_cache_path(params_hash)

        metadata = {
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "random_state": self.random_state,
            "shape": list(projection.shape),
        }

        np.save(npy_path, projection)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved UMAP projection to cache: {npy_path}")

    def _stratified_subsample(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray],
        subsample_n: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Stratified subsampling preserving class proportions.

        Returns:
            Tuple of (subsampled_embeddings, subsampled_labels, indices).
            indices maps back to original array positions.
        """
        n_samples = embeddings.shape[0]

        if n_samples <= subsample_n:
            return embeddings, labels, np.arange(n_samples)

        rng = np.random.default_rng(self.random_state)

        if labels is None:
            # Without labels, do uniform random sampling
            indices = rng.choice(n_samples, size=subsample_n, replace=False)
            indices.sort()
            return embeddings[indices], None, indices

        # Stratified sampling: allocate proportionally per class
        unique_classes = np.unique(labels)
        indices_list = []

        for cls in unique_classes:
            cls_mask = labels == cls
            cls_indices = np.where(cls_mask)[0]
            n_cls = len(cls_indices)
            # Proportional allocation
            n_select = max(1, int(round(subsample_n * n_cls / n_samples)))
            n_select = min(n_select, n_cls)
            selected = rng.choice(cls_indices, size=n_select, replace=False)
            indices_list.append(selected)

        indices = np.concatenate(indices_list)

        # If we have too many or too few due to rounding, adjust
        if len(indices) > subsample_n:
            indices = rng.choice(indices, size=subsample_n, replace=False)
        elif len(indices) < subsample_n:
            # Add more samples from underrepresented pool
            remaining = np.setdiff1d(np.arange(n_samples), indices)
            extra = rng.choice(remaining, size=subsample_n - len(indices), replace=False)
            indices = np.concatenate([indices, extra])

        indices.sort()
        sub_labels = labels[indices] if labels is not None else None
        return embeddings[indices], sub_labels, indices

    def fit_transform(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        subsample_n: int = 50000,
    ) -> np.ndarray:
        """Compute UMAP projection, with optional subsampling and caching.

        Args:
            embeddings: Input embeddings of shape (N, D).
            labels: Optional class labels of shape (N,).
            subsample_n: Max points to project. Stratified subsampling applied if N > subsample_n.

        Returns:
            Projections of shape (M, n_components) where M = min(N, subsample_n).
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

        # Subsample if needed
        sub_embeddings, sub_labels, indices = self._stratified_subsample(
            embeddings, labels, subsample_n
        )

        if sub_embeddings.shape[0] < embeddings.shape[0]:
            logger.info(
                f"Subsampled from {embeddings.shape[0]} to {sub_embeddings.shape[0]} points "
                f"(stratified)"
            )

        # Check cache
        params_hash = self._compute_params_hash(sub_embeddings)
        cached = self._load_cache(params_hash)
        if cached is not None and cached.shape[0] == sub_embeddings.shape[0]:
            return cached

        # Compute UMAP
        logger.info(
            f"Computing UMAP projection: {sub_embeddings.shape[0]} points, "
            f"D={sub_embeddings.shape[1]}, n_components={self.n_components}"
        )

        reducer = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )

        projection = reducer.fit_transform(sub_embeddings)
        projection = projection.astype(np.float32)

        # Cache result
        self._save_cache(params_hash, projection)

        return projection

    def create_figure(
        self,
        projections: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[list[str]] = None,
        anomaly_mask: Optional[np.ndarray] = None,
        title: str = "UMAP",
    ) -> go.Figure:
        """Create interactive Plotly scatter plot with WebGL, class coloring, and anomaly overlay.

        Args:
            projections: Projection coordinates, shape (N, 2) or (N, 3).
            labels: Class labels for each point, shape (N,).
            class_names: Optional list of class name strings (indexed by label value).
            anomaly_mask: Optional boolean mask of shape (N,) indicating anomalies.
            title: Plot title.

        Returns:
            Plotly Figure with WebGL rendering.
        """
        if projections.ndim != 2:
            raise ValueError(f"projections must be 2D array, got shape {projections.shape}")
        if projections.shape[1] not in (2, 3):
            raise ValueError(
                f"projections must have 2 or 3 columns, got {projections.shape[1]}"
            )
        if len(labels) != projections.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} != projections rows {projections.shape[0]}"
            )

        if class_names is None:
            class_names = DEFAULT_CLASS_NAMES

        is_3d = projections.shape[1] == 3
        unique_labels = np.unique(labels)

        fig = go.Figure()

        for cls_idx in unique_labels:
            mask = labels == cls_idx
            cls_name = (
                class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
            )
            color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]

            if is_3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=projections[mask, 0],
                        y=projections[mask, 1],
                        z=projections[mask, 2],
                        mode="markers",
                        marker=dict(size=2, color=color, opacity=0.7),
                        name=cls_name,
                        hovertemplate=(
                            f"Class: {cls_name}<br>"
                            "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=projections[mask, 0],
                        y=projections[mask, 1],
                        mode="markers",
                        marker=dict(size=3, color=color, opacity=0.7),
                        name=cls_name,
                        hovertemplate=(
                            f"Class: {cls_name}<br>"
                            "x: %{x:.2f}<br>y: %{y:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

        # Anomaly overlay with star markers
        if anomaly_mask is not None and np.any(anomaly_mask):
            anomaly_points = projections[anomaly_mask]
            if is_3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=anomaly_points[:, 0],
                        y=anomaly_points[:, 1],
                        z=anomaly_points[:, 2],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color="rgba(255, 0, 0, 0.8)",
                            symbol="diamond",
                            line=dict(width=1, color="red"),
                        ),
                        name="Anomaly",
                        hovertemplate=(
                            "Anomaly<br>"
                            "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=anomaly_points[:, 0],
                        y=anomaly_points[:, 1],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="rgba(255, 0, 0, 0.8)",
                            symbol="star",
                            line=dict(width=1, color="red"),
                        ),
                        name="Anomaly",
                        hovertemplate=(
                            "Anomaly<br>"
                            "x: %{x:.2f}<br>y: %{y:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title=title,
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        if not is_3d:
            fig.update_layout(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
            )
        else:
            fig.update_layout(
                scene=dict(
                    xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2",
                    zaxis_title="UMAP 3",
                ),
            )

        return fig

    def filter_classes(
        self,
        projections: np.ndarray,
        labels: np.ndarray,
        keep_classes: list[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter projections to keep only specified classes without recomputation.

        Args:
            projections: Pre-computed projections, shape (N, n_components).
            labels: Class labels, shape (N,).
            keep_classes: List/array of class indices to keep.

        Returns:
            Tuple of (filtered_projections, filtered_labels).
        """
        if len(labels) != projections.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} != projections rows {projections.shape[0]}"
            )

        keep_set = set(int(c) for c in keep_classes)
        mask = np.array([int(lbl) in keep_set for lbl in labels], dtype=bool)

        return projections[mask], labels[mask]
