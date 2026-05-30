"""TSNEVisualizer with PCA initialization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class TSNEVisualizer:
    """t-SNE visualization with PCA initialization.

    Applies PCA to reduce dimensionality before running t-SNE for speed.
    Configurable perplexity in range [5, 100].

    Args:
        perplexity: t-SNE perplexity parameter (5-100, default 30).
        n_components: Number of output dimensions (default 2).
        random_state: Random seed for reproducibility.
        pca_init_dims: Number of PCA dimensions before t-SNE (default 50).
    """

    def __init__(
        self,
        perplexity: float = 30.0,
        n_components: int = 2,
        random_state: int = 42,
        pca_init_dims: int = 50,
    ) -> None:
        if not (5 <= perplexity <= 100):
            raise ValueError(
                f"Perplexity must be in [5, 100], got {perplexity}"
            )
        self.perplexity = perplexity
        self.n_components = n_components
        self.random_state = random_state
        self.pca_init_dims = pca_init_dims

    def fit_transform(
        self, embeddings: np.ndarray, subsample_n: int = 25000
    ) -> np.ndarray:
        """Reduce with PCA pre-processing for speed, then t-SNE.

        Steps:
            1. Subsample if N > subsample_n (random, preserving order of selected)
            2. PCA to pca_init_dims dimensions
            3. t-SNE to n_components dimensions

        Args:
            embeddings: Input array of shape (N, D).
            subsample_n: Maximum number of samples. If N > subsample_n,
                randomly subsample to this count.

        Returns:
            Projections of shape (M, n_components) where M = min(N, subsample_n).
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got shape {embeddings.shape}"
            )

        n_samples, n_features = embeddings.shape

        # Subsample if needed
        if n_samples > subsample_n:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=subsample_n, replace=False)
            indices.sort()
            embeddings = embeddings[indices]
            n_samples = subsample_n

        # PCA reduction (only if input dims > pca_init_dims)
        pca_dims = min(self.pca_init_dims, n_features, n_samples)
        if n_features > pca_dims:
            pca = PCA(n_components=pca_dims, random_state=self.random_state)
            embeddings = pca.fit_transform(embeddings)

        # t-SNE
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=min(self.perplexity, (n_samples - 1) / 3.0),
            init="pca",
            random_state=self.random_state,
            learning_rate="auto",
        )
        projections = tsne.fit_transform(embeddings)

        return projections

    def create_figure(
        self,
        projections: np.ndarray,
        labels: np.ndarray,
        class_names: list[str] | None = None,
        title: str = "t-SNE",
    ) -> go.Figure:
        """Create a Plotly scatter plot with class coloring.

        Args:
            projections: 2D array of shape (N, 2).
            labels: Integer labels of shape (N,).
            class_names: Optional list mapping label indices to names.
            title: Plot title.

        Returns:
            Plotly Figure with colored scatter plot.
        """
        # Colorblind-friendly palette (9 classes)
        palette = [
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#009E73",  # bluish green
            "#F0E442",  # yellow
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#CC79A7",  # reddish purple
            "#999999",  # grey
            "#000000",  # black
        ]

        unique_labels = np.unique(labels)
        fig = go.Figure()

        for label in unique_labels:
            mask = labels == label
            name = (
                class_names[label]
                if class_names is not None and label < len(class_names)
                else f"Class {label}"
            )
            color = palette[int(label) % len(palette)]

            fig.add_trace(
                go.Scattergl(
                    x=projections[mask, 0],
                    y=projections[mask, 1],
                    mode="markers",
                    name=name,
                    marker=dict(size=4, color=color, opacity=0.7),
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "x: %{x:.2f}<br>"
                        "y: %{y:.2f}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            template="plotly_white",
            legend=dict(itemsizing="constant"),
            width=800,
            height=600,
        )

        return fig
