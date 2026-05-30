"""Unit tests for UMAPVisualizer: shape, caching, and filtering."""

import json
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from src.visualization.umap_viz import UMAPVisualizer


@pytest.fixture
def small_embeddings():
    """Generate small embeddings for fast UMAP tests."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((200, 64)).astype(np.float32)
    labels = np.repeat(np.arange(5), 40)
    return embeddings, labels


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    return tmp_path / "umap_cache"


class TestUMAPVisualizerInit:
    """Test initialization and parameter validation."""

    def test_valid_2d_init(self):
        viz = UMAPVisualizer(n_components=2)
        assert viz.n_components == 2

    def test_valid_3d_init(self):
        viz = UMAPVisualizer(n_components=3)
        assert viz.n_components == 3

    def test_invalid_n_components_raises(self):
        with pytest.raises(ValueError, match="n_components must be 2 or 3"):
            UMAPVisualizer(n_components=4)

    def test_invalid_n_neighbors_raises(self):
        with pytest.raises(ValueError, match="n_neighbors must be >= 2"):
            UMAPVisualizer(n_neighbors=1)

    def test_invalid_min_dist_raises(self):
        with pytest.raises(ValueError, match="min_dist must be in"):
            UMAPVisualizer(min_dist=1.5)

    def test_cache_dir_created(self, tmp_path):
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()
        UMAPVisualizer(cache_dir=cache_dir)
        assert cache_dir.exists()


class TestFitTransform:
    """Test fit_transform output shapes and subsampling."""

    def test_output_shape_2d(self, small_embeddings):
        embeddings, labels = small_embeddings
        viz = UMAPVisualizer(n_components=2, n_neighbors=10)
        result = viz.fit_transform(embeddings, labels)
        assert result.shape == (200, 2)
        assert result.dtype == np.float32

    def test_output_shape_3d(self, small_embeddings):
        embeddings, labels = small_embeddings
        viz = UMAPVisualizer(n_components=3, n_neighbors=10)
        result = viz.fit_transform(embeddings, labels)
        assert result.shape == (200, 3)
        assert result.dtype == np.float32

    def test_subsample_limits_output(self):
        """When N > subsample_n, output is subsampled."""
        rng = np.random.default_rng(123)
        embeddings = rng.standard_normal((500, 32)).astype(np.float32)
        labels = np.repeat(np.arange(5), 100)
        viz = UMAPVisualizer(n_components=2, n_neighbors=5)
        result = viz.fit_transform(embeddings, labels, subsample_n=100)
        assert result.shape[0] == 100
        assert result.shape[1] == 2

    def test_no_subsample_when_small(self, small_embeddings):
        """When N <= subsample_n, all points are kept."""
        embeddings, labels = small_embeddings
        viz = UMAPVisualizer(n_components=2, n_neighbors=10)
        result = viz.fit_transform(embeddings, labels, subsample_n=50000)
        assert result.shape[0] == 200

    def test_fit_transform_without_labels(self, small_embeddings):
        embeddings, _ = small_embeddings
        viz = UMAPVisualizer(n_components=2, n_neighbors=10)
        result = viz.fit_transform(embeddings, labels=None)
        assert result.shape == (200, 2)

    def test_invalid_embeddings_ndim_raises(self):
        viz = UMAPVisualizer(n_components=2)
        with pytest.raises(ValueError, match="embeddings must be 2D"):
            viz.fit_transform(np.zeros((10,)))


class TestCaching:
    """Test disk caching behavior."""

    def test_cache_saves_and_loads(self, small_embeddings, cache_dir):
        embeddings, labels = small_embeddings
        viz = UMAPVisualizer(n_components=2, n_neighbors=10, cache_dir=cache_dir)

        # First call computes and saves
        result1 = viz.fit_transform(embeddings, labels)

        # Cache files should exist
        npy_files = list(cache_dir.glob("*.npy"))
        json_files = list(cache_dir.glob("*_meta.json"))
        assert len(npy_files) == 1
        assert len(json_files) == 1

        # Second call should load from cache
        result2 = viz.fit_transform(embeddings, labels)
        np.testing.assert_array_equal(result1, result2)

    def test_cache_invalidated_on_param_change(self, small_embeddings, cache_dir):
        embeddings, labels = small_embeddings

        viz1 = UMAPVisualizer(n_components=2, n_neighbors=10, cache_dir=cache_dir)
        result1 = viz1.fit_transform(embeddings, labels)

        # Different params should produce new cache entry
        viz2 = UMAPVisualizer(n_components=2, n_neighbors=20, cache_dir=cache_dir)
        result2 = viz2.fit_transform(embeddings, labels)

        # Should have 2 cache entries now
        npy_files = list(cache_dir.glob("*.npy"))
        assert len(npy_files) == 2

    def test_no_cache_when_cache_dir_none(self, small_embeddings):
        embeddings, labels = small_embeddings
        viz = UMAPVisualizer(n_components=2, n_neighbors=10, cache_dir=None)
        result = viz.fit_transform(embeddings, labels)
        assert result.shape == (200, 2)

    def test_cache_metadata_content(self, small_embeddings, cache_dir):
        embeddings, labels = small_embeddings
        viz = UMAPVisualizer(
            n_components=2, n_neighbors=10, min_dist=0.1,
            metric="euclidean", cache_dir=cache_dir,
        )
        viz.fit_transform(embeddings, labels)

        meta_files = list(cache_dir.glob("*_meta.json"))
        assert len(meta_files) == 1
        with open(meta_files[0]) as f:
            meta = json.load(f)

        assert meta["n_components"] == 2
        assert meta["n_neighbors"] == 10
        assert meta["min_dist"] == 0.1
        assert meta["metric"] == "euclidean"
        assert meta["shape"] == [200, 2]


class TestCreateFigure:
    """Test Plotly figure creation."""

    def test_2d_figure_creation(self):
        projections = np.random.randn(100, 2).astype(np.float32)
        labels = np.repeat(np.arange(5), 20)
        viz = UMAPVisualizer(n_components=2)
        fig = viz.create_figure(projections, labels)
        assert isinstance(fig, go.Figure)
        # One trace per class
        assert len(fig.data) == 5

    def test_3d_figure_creation(self):
        projections = np.random.randn(99, 3).astype(np.float32)
        labels = np.repeat(np.arange(3), 33)
        viz = UMAPVisualizer(n_components=3)
        fig = viz.create_figure(projections, labels)
        assert isinstance(fig, go.Figure)

    def test_figure_with_anomaly_overlay(self):
        projections = np.random.randn(100, 2).astype(np.float32)
        labels = np.repeat(np.arange(5), 20)
        anomaly_mask = np.zeros(100, dtype=bool)
        anomaly_mask[:10] = True
        viz = UMAPVisualizer(n_components=2)
        fig = viz.create_figure(projections, labels, anomaly_mask=anomaly_mask)
        # 5 class traces + 1 anomaly trace
        assert len(fig.data) == 6

    def test_figure_with_custom_class_names(self):
        projections = np.random.randn(60, 2).astype(np.float32)
        labels = np.repeat(np.arange(3), 20)
        class_names = ["Alpha", "Beta", "Gamma"]
        viz = UMAPVisualizer(n_components=2)
        fig = viz.create_figure(projections, labels, class_names=class_names)
        trace_names = [trace.name for trace in fig.data]
        assert "Alpha" in trace_names
        assert "Beta" in trace_names
        assert "Gamma" in trace_names

    def test_figure_invalid_projections_raises(self):
        viz = UMAPVisualizer(n_components=2)
        with pytest.raises(ValueError, match="projections must be 2D"):
            viz.create_figure(np.zeros(10), np.zeros(10))

    def test_figure_mismatched_labels_raises(self):
        viz = UMAPVisualizer(n_components=2)
        with pytest.raises(ValueError, match="labels length"):
            viz.create_figure(np.zeros((10, 2)), np.zeros(5))

    def test_no_anomaly_trace_when_mask_all_false(self):
        projections = np.random.randn(50, 2).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        anomaly_mask = np.zeros(50, dtype=bool)
        viz = UMAPVisualizer(n_components=2)
        fig = viz.create_figure(projections, labels, anomaly_mask=anomaly_mask)
        # Only class traces, no anomaly trace
        assert len(fig.data) == 5


class TestFilterClasses:
    """Test class filtering without recomputation."""

    def test_filter_keeps_selected_classes(self):
        projections = np.random.randn(100, 2).astype(np.float32)
        labels = np.repeat(np.arange(5), 20)
        viz = UMAPVisualizer(n_components=2)

        filtered_proj, filtered_labels = viz.filter_classes(
            projections, labels, keep_classes=[0, 2, 4]
        )

        assert set(np.unique(filtered_labels)) == {0, 2, 4}
        assert filtered_proj.shape[0] == 60  # 20 per class * 3 classes
        assert filtered_proj.shape[1] == 2

    def test_filter_removes_unselected_classes(self):
        projections = np.random.randn(100, 2).astype(np.float32)
        labels = np.repeat(np.arange(5), 20)
        viz = UMAPVisualizer(n_components=2)

        filtered_proj, filtered_labels = viz.filter_classes(
            projections, labels, keep_classes=[1, 3]
        )

        assert 0 not in filtered_labels
        assert 2 not in filtered_labels
        assert 4 not in filtered_labels
        assert filtered_proj.shape[0] == 40

    def test_filter_preserves_coordinates(self):
        """Filtered coordinates match original values (no recomputation)."""
        rng = np.random.default_rng(99)
        projections = rng.standard_normal((100, 2)).astype(np.float32)
        labels = np.repeat(np.arange(5), 20)
        viz = UMAPVisualizer(n_components=2)

        filtered_proj, _ = viz.filter_classes(
            projections, labels, keep_classes=[0]
        )

        # Original points for class 0 are indices 0-19
        np.testing.assert_array_equal(filtered_proj, projections[:20])

    def test_filter_empty_keep_classes(self):
        projections = np.random.randn(50, 2).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        viz = UMAPVisualizer(n_components=2)

        filtered_proj, filtered_labels = viz.filter_classes(
            projections, labels, keep_classes=[]
        )

        assert filtered_proj.shape[0] == 0
        assert filtered_labels.shape[0] == 0

    def test_filter_all_classes_returns_all(self):
        projections = np.random.randn(50, 2).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        viz = UMAPVisualizer(n_components=2)

        filtered_proj, filtered_labels = viz.filter_classes(
            projections, labels, keep_classes=[0, 1, 2, 3, 4]
        )

        assert filtered_proj.shape[0] == 50
        np.testing.assert_array_equal(filtered_proj, projections)

    def test_filter_mismatched_lengths_raises(self):
        viz = UMAPVisualizer(n_components=2)
        with pytest.raises(ValueError, match="labels length"):
            viz.filter_classes(np.zeros((10, 2)), np.zeros(5), keep_classes=[0])


class TestStratifiedSubsampling:
    """Test the internal stratified subsampling logic."""

    def test_subsample_preserves_class_proportions(self):
        rng = np.random.default_rng(42)
        # 5 classes with different sizes
        labels = np.array([0]*200 + [1]*100 + [2]*50 + [3]*300 + [4]*150)
        embeddings = rng.standard_normal((800, 32)).astype(np.float32)
        viz = UMAPVisualizer(n_components=2, n_neighbors=5)

        sub_emb, sub_labels, indices = viz._stratified_subsample(
            embeddings, labels, subsample_n=200
        )

        assert sub_emb.shape[0] == 200
        assert sub_labels.shape[0] == 200

        # Check proportions are roughly maintained
        original_proportions = {
            cls: np.sum(labels == cls) / len(labels)
            for cls in np.unique(labels)
        }
        sub_proportions = {
            cls: np.sum(sub_labels == cls) / len(sub_labels)
            for cls in np.unique(sub_labels)
        }

        for cls in original_proportions:
            # Allow some tolerance for rounding
            assert abs(original_proportions[cls] - sub_proportions[cls]) < 0.05

    def test_no_subsample_when_under_limit(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 16)).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        viz = UMAPVisualizer(n_components=2)

        sub_emb, sub_labels, indices = viz._stratified_subsample(
            embeddings, labels, subsample_n=100
        )

        assert sub_emb.shape[0] == 50
        np.testing.assert_array_equal(indices, np.arange(50))
