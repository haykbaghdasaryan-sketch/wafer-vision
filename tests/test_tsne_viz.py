"""Unit tests for TSNEVisualizer."""

import numpy as np
import pytest
import plotly.graph_objects as go

from src.visualization.tsne_viz import TSNEVisualizer


class TestTSNEVisualizerInit:
    """Test TSNEVisualizer initialization and parameter validation."""

    def test_default_parameters(self):
        viz = TSNEVisualizer()
        assert viz.perplexity == 30.0
        assert viz.n_components == 2
        assert viz.random_state == 42
        assert viz.pca_init_dims == 50

    def test_custom_parameters(self):
        viz = TSNEVisualizer(perplexity=50.0, n_components=2, random_state=7, pca_init_dims=30)
        assert viz.perplexity == 50.0
        assert viz.random_state == 7
        assert viz.pca_init_dims == 30

    def test_perplexity_lower_bound(self):
        viz = TSNEVisualizer(perplexity=5)
        assert viz.perplexity == 5

    def test_perplexity_upper_bound(self):
        viz = TSNEVisualizer(perplexity=100)
        assert viz.perplexity == 100

    def test_perplexity_too_low_raises(self):
        with pytest.raises(ValueError, match="Perplexity must be in"):
            TSNEVisualizer(perplexity=4)

    def test_perplexity_too_high_raises(self):
        with pytest.raises(ValueError, match="Perplexity must be in"):
            TSNEVisualizer(perplexity=101)


class TestTSNEFitTransform:
    """Test fit_transform produces correct output shapes."""

    def test_output_shape_small(self):
        viz = TSNEVisualizer(perplexity=5)
        embeddings = np.random.default_rng(0).standard_normal((50, 128))
        result = viz.fit_transform(embeddings)
        assert result.shape == (50, 2)

    def test_output_shape_high_dim(self):
        """PCA should reduce from 2048 to 50 before t-SNE."""
        viz = TSNEVisualizer(perplexity=5)
        embeddings = np.random.default_rng(0).standard_normal((100, 2048))
        result = viz.fit_transform(embeddings)
        assert result.shape == (100, 2)

    def test_subsampling(self):
        """When N > subsample_n, output should have subsample_n rows."""
        viz = TSNEVisualizer(perplexity=5)
        embeddings = np.random.default_rng(0).standard_normal((200, 64))
        result = viz.fit_transform(embeddings, subsample_n=50)
        assert result.shape == (50, 2)

    def test_no_subsampling_when_below_limit(self):
        viz = TSNEVisualizer(perplexity=5)
        embeddings = np.random.default_rng(0).standard_normal((30, 64))
        result = viz.fit_transform(embeddings, subsample_n=100)
        assert result.shape == (30, 2)

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        viz = TSNEVisualizer(perplexity=5, random_state=123)
        embeddings = np.random.default_rng(0).standard_normal((50, 64))
        r1 = viz.fit_transform(embeddings)
        r2 = viz.fit_transform(embeddings)
        np.testing.assert_array_equal(r1, r2)

    def test_invalid_input_shape(self):
        viz = TSNEVisualizer(perplexity=5)
        with pytest.raises(ValueError, match="Expected 2D array"):
            viz.fit_transform(np.ones((10,)))

    def test_skips_pca_when_dims_already_small(self):
        """If input dims <= pca_init_dims, PCA step should still work."""
        viz = TSNEVisualizer(perplexity=5, pca_init_dims=50)
        embeddings = np.random.default_rng(0).standard_normal((50, 30))
        result = viz.fit_transform(embeddings)
        assert result.shape == (50, 2)


class TestTSNECreateFigure:
    """Test Plotly figure creation."""

    def test_returns_figure(self):
        viz = TSNEVisualizer()
        projections = np.random.default_rng(0).standard_normal((20, 2))
        labels = np.array([0] * 10 + [1] * 10)
        fig = viz.create_figure(projections, labels)
        assert isinstance(fig, go.Figure)

    def test_trace_count_matches_classes(self):
        viz = TSNEVisualizer()
        projections = np.random.default_rng(0).standard_normal((30, 2))
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        fig = viz.create_figure(projections, labels)
        assert len(fig.data) == 3

    def test_class_names_in_legend(self):
        viz = TSNEVisualizer()
        projections = np.random.default_rng(0).standard_normal((20, 2))
        labels = np.array([0] * 10 + [1] * 10)
        class_names = ["Center", "Donut"]
        fig = viz.create_figure(projections, labels, class_names=class_names)
        trace_names = [t.name for t in fig.data]
        assert "Center" in trace_names
        assert "Donut" in trace_names

    def test_custom_title(self):
        viz = TSNEVisualizer()
        projections = np.random.default_rng(0).standard_normal((20, 2))
        labels = np.zeros(20, dtype=int)
        fig = viz.create_figure(projections, labels, title="My t-SNE")
        assert fig.layout.title.text == "My t-SNE"

    def test_default_class_name_fallback(self):
        viz = TSNEVisualizer()
        projections = np.random.default_rng(0).standard_normal((10, 2))
        labels = np.array([3] * 10)
        fig = viz.create_figure(projections, labels)
        assert fig.data[0].name == "Class 3"
