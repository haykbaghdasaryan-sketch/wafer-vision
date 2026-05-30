"""Unit tests: known baselines for metrics and CSV round-trip."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import EmbeddingMetrics, MetricsComputer
from src.exceptions import DimensionMismatchError


@pytest.fixture
def clustered_embeddings():
    """Create well-clustered embeddings for predictable metric values."""
    rng = np.random.RandomState(42)
    n_per_class = 50
    n_classes = 3
    dim = 32

    embeddings = []
    labels = []

    for c in range(n_classes):
        # Each class centered at a different location with small spread
        center = rng.randn(dim).astype(np.float32) * 10
        samples = center + rng.randn(n_per_class, dim).astype(np.float32) * 0.1
        embeddings.append(samples)
        labels.extend([c] * n_per_class)

    return np.vstack(embeddings), np.array(labels)


@pytest.fixture
def random_embeddings():
    """Create random embeddings (poor clustering) for testing edge cases."""
    rng = np.random.RandomState(123)
    n_samples = 100
    dim = 64
    n_classes = 5

    embeddings = rng.randn(n_samples, dim).astype(np.float32)
    labels = rng.randint(0, n_classes, size=n_samples)

    return embeddings, labels


class TestMetricsComputerInit:
    """Tests for MetricsComputer initialization."""

    def test_dimension_mismatch_raises(self):
        """Should raise DimensionMismatchError when embeddings and labels differ in length."""
        embeddings = np.random.randn(100, 32).astype(np.float32)
        labels = np.arange(50)  # mismatched length

        with pytest.raises(DimensionMismatchError):
            MetricsComputer(embeddings, labels)

    def test_single_class_raises(self):
        """Should raise ValueError when fewer than 2 classes."""
        embeddings = np.random.randn(100, 32).astype(np.float32)
        labels = np.zeros(100, dtype=np.int64)

        with pytest.raises(ValueError, match="At least 2 classes"):
            MetricsComputer(embeddings, labels)

    def test_invalid_metric_raises(self):
        """Should raise ValueError for invalid distance metric."""
        embeddings = np.random.randn(100, 32).astype(np.float32)
        labels = np.random.randint(0, 3, size=100)

        with pytest.raises(ValueError, match="Invalid distance_metric"):
            MetricsComputer(embeddings, labels, distance_metric="manhattan")

    def test_valid_init_euclidean(self, clustered_embeddings):
        """Should initialize correctly with valid inputs."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels, distance_metric="euclidean")
        assert mc.n_samples == len(labels)
        assert mc.distance_metric == "euclidean"

    def test_valid_init_cosine(self, clustered_embeddings):
        """Should initialize correctly with cosine distance."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels, distance_metric="cosine")
        assert mc.distance_metric == "cosine"
        # Embeddings should be normalized
        norms = np.linalg.norm(mc.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestKNNAccuracy:
    """Tests for KNN Accuracy computation."""

    def test_well_clustered_high_accuracy(self, clustered_embeddings):
        """Well-separated clusters should yield high KNN accuracy."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        knn_acc = mc.compute_knn_accuracy([1, 3, 5])

        # With well-separated clusters, accuracy should be very high
        for k in [1, 3, 5]:
            assert 0.9 <= knn_acc[k] <= 1.0

    def test_accuracy_bounds(self, random_embeddings):
        """KNN accuracy should always be in [0, 1]."""
        embeddings, labels = random_embeddings
        mc = MetricsComputer(embeddings, labels)
        knn_acc = mc.compute_knn_accuracy([1, 3, 5, 10])

        for k, acc in knn_acc.items():
            assert 0.0 <= acc <= 1.0

    def test_custom_k_values(self, clustered_embeddings):
        """Should support custom k values."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        knn_acc = mc.compute_knn_accuracy([2, 7])

        assert 2 in knn_acc
        assert 7 in knn_acc


class TestRecallAtK:
    """Tests for Recall@K computation."""

    def test_well_clustered_high_recall(self, clustered_embeddings):
        """Well-separated clusters should yield high recall."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        recall = mc.compute_recall_at_k([1, 5, 10])

        for k in [1, 5, 10]:
            assert 0.9 <= recall[k] <= 1.0

    def test_recall_bounds(self, random_embeddings):
        """Recall should always be in [0, 1]."""
        embeddings, labels = random_embeddings
        mc = MetricsComputer(embeddings, labels)
        recall = mc.compute_recall_at_k([1, 5, 10])

        for k, r in recall.items():
            assert 0.0 <= r <= 1.0

    def test_recall_monotonic(self, random_embeddings):
        """Recall should be non-decreasing with increasing k."""
        embeddings, labels = random_embeddings
        mc = MetricsComputer(embeddings, labels)
        recall = mc.compute_recall_at_k([1, 5, 10])

        assert recall[1] <= recall[5] <= recall[10]


class TestPrecisionAtK:
    """Tests for Precision@K computation."""

    def test_well_clustered_high_precision(self, clustered_embeddings):
        """Well-separated clusters should yield high precision."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        precision = mc.compute_precision_at_k([1, 5, 10])

        for k in [1, 5, 10]:
            assert 0.9 <= precision[k] <= 1.0

    def test_precision_bounds(self, random_embeddings):
        """Precision should always be in [0, 1]."""
        embeddings, labels = random_embeddings
        mc = MetricsComputer(embeddings, labels)
        precision = mc.compute_precision_at_k([1, 5, 10])

        for k, p in precision.items():
            assert 0.0 <= p <= 1.0


class TestSilhouetteScore:
    """Tests for Silhouette Score computation."""

    def test_well_clustered_positive(self, clustered_embeddings):
        """Well-separated clusters should have positive silhouette."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        sil = mc.compute_silhouette()

        assert sil > 0.5  # Well-clustered should be significantly positive

    def test_silhouette_bounds(self, random_embeddings):
        """Silhouette should be in [-1, 1]."""
        embeddings, labels = random_embeddings
        mc = MetricsComputer(embeddings, labels)
        sil = mc.compute_silhouette()

        assert -1.0 <= sil <= 1.0


class TestNMI:
    """Tests for NMI computation."""

    def test_well_clustered_high_nmi(self, clustered_embeddings):
        """Well-separated clusters should yield high NMI."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        nmi = mc.compute_nmi(n_clusters=3)

        assert nmi > 0.8

    def test_nmi_bounds(self, random_embeddings):
        """NMI should be in [0, 1]."""
        embeddings, labels = random_embeddings
        mc = MetricsComputer(embeddings, labels)
        nmi = mc.compute_nmi()

        assert 0.0 <= nmi <= 1.0


class TestMAP:
    """Tests for Mean Average Precision computation."""

    def test_well_clustered_high_map(self, clustered_embeddings):
        """Well-separated clusters should yield high MAP."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        map_score = mc.compute_map()

        assert map_score > 0.8

    def test_map_bounds(self, random_embeddings):
        """MAP should be in [0, 1]."""
        embeddings, labels = random_embeddings
        mc = MetricsComputer(embeddings, labels)
        map_score = mc.compute_map()

        assert 0.0 <= map_score <= 1.0


class TestSeparability:
    """Tests for inter/intra class distances and separability index."""

    def test_well_clustered_high_separability(self, clustered_embeddings):
        """Well-separated clusters should have high separability index."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        inter, intra, sep = mc.compute_separability()

        assert inter > 0
        assert intra > 0
        assert sep > 10  # Well-separated clusters

    def test_distances_non_negative(self, random_embeddings):
        """Inter and intra class distances should be non-negative."""
        embeddings, labels = random_embeddings
        mc = MetricsComputer(embeddings, labels)
        inter, intra, sep = mc.compute_separability()

        assert inter >= 0
        assert intra >= 0
        assert sep >= 0


class TestPerClassMetrics:
    """Tests for per-class metric breakdown."""

    def test_all_classes_present(self, clustered_embeddings):
        """Per-class metrics should contain entries for all classes."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        per_class = mc.compute_per_class_metrics()

        unique_classes = np.unique(labels)
        for c in unique_classes:
            assert str(c) in per_class

    def test_per_class_structure(self, clustered_embeddings):
        """Each class entry should have the expected keys."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        per_class = mc.compute_per_class_metrics()

        expected_keys = {
            "knn_accuracy_k5", "precision_at_5", "recall_at_5",
            "intra_class_distance", "n_samples"
        }
        for class_label, metrics in per_class.items():
            assert set(metrics.keys()) == expected_keys

    def test_per_class_bounds(self, random_embeddings):
        """Per-class metrics should be in valid ranges."""
        embeddings, labels = random_embeddings
        mc = MetricsComputer(embeddings, labels)
        per_class = mc.compute_per_class_metrics()

        for class_label, metrics in per_class.items():
            assert 0.0 <= metrics["knn_accuracy_k5"] <= 1.0
            assert 0.0 <= metrics["precision_at_5"] <= 1.0
            assert 0.0 <= metrics["recall_at_5"] <= 1.0
            assert metrics["intra_class_distance"] >= 0.0
            assert metrics["n_samples"] > 0


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_ci_structure(self, clustered_embeddings):
        """CI dict should have entries for each k value."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        ci = mc._compute_bootstrap_ci(n_bootstrap=100)

        expected_keys = {"knn_1", "knn_3", "knn_5", "knn_10"}
        assert set(ci.keys()) == expected_keys

    def test_ci_bounds(self, clustered_embeddings):
        """CI lower should be <= upper and both in [0, 1]."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        ci = mc._compute_bootstrap_ci(n_bootstrap=100)

        for key, (lower, upper) in ci.items():
            assert 0.0 <= lower <= upper <= 1.0


class TestComputeAll:
    """Tests for the compute_all method."""

    def test_returns_embedding_metrics(self, clustered_embeddings):
        """compute_all should return a fully populated EmbeddingMetrics."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        result = mc.compute_all(bootstrap_ci=False)

        assert isinstance(result, EmbeddingMetrics)
        assert len(result.knn_accuracy) > 0
        assert len(result.recall_at_k) > 0
        assert len(result.precision_at_k) > 0
        assert result.per_class_metrics is not None

    def test_compute_all_with_ci(self, clustered_embeddings):
        """compute_all with CI should populate confidence_intervals."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)
        result = mc.compute_all(bootstrap_ci=True, n_bootstrap=50)

        assert result.confidence_intervals is not None
        assert len(result.confidence_intervals) > 0


class TestCSVRoundTrip:
    """Tests for CSV save/load round-trip support."""

    def test_save_and_load(self, clustered_embeddings):
        """Metrics should survive CSV save/load round-trip."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "metrics.csv"
            mc.save_csv(csv_path, model_name="resnet50", training_mode="finetune")

            loaded = MetricsComputer.load_csv(csv_path)

        # Verify the loaded metrics match
        original = mc.compute_all(bootstrap_ci=False)

        for k in original.knn_accuracy:
            assert abs(loaded.knn_accuracy[k] - original.knn_accuracy[k]) < 1e-5

        for k in original.recall_at_k:
            assert abs(loaded.recall_at_k[k] - original.recall_at_k[k]) < 1e-5

        for k in original.precision_at_k:
            assert abs(loaded.precision_at_k[k] - original.precision_at_k[k]) < 1e-5

        assert abs(loaded.silhouette_score - original.silhouette_score) < 1e-5
        assert abs(loaded.nmi - original.nmi) < 1e-5
        assert abs(loaded.mean_average_precision - original.mean_average_precision) < 1e-5
        assert abs(loaded.inter_class_distance - original.inter_class_distance) < 1e-5
        assert abs(loaded.intra_class_distance - original.intra_class_distance) < 1e-5
        assert abs(loaded.separability_index - original.separability_index) < 1e-5

    def test_load_nonexistent_raises(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MetricsComputer.load_csv(Path("/nonexistent/metrics.csv"))

    def test_csv_has_expected_columns(self, clustered_embeddings):
        """CSV file should have the expected column headers."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "metrics.csv"
            mc.save_csv(csv_path, model_name="test_model", training_mode="pretrained")

            with open(csv_path, "r") as f:
                import csv as csv_mod
                reader = csv_mod.DictReader(f)
                fieldnames = reader.fieldnames

        expected = {
            "model_name", "training_mode", "metric_name", "metric_value",
            "timestamp", "n_queries", "database_size", "distance_metric",
        }
        assert set(fieldnames) == expected


class TestCosineDistance:
    """Tests for cosine distance mode."""

    def test_cosine_mode_works(self, clustered_embeddings):
        """Cosine distance mode should produce valid metrics."""
        embeddings, labels = clustered_embeddings
        mc = MetricsComputer(embeddings, labels, distance_metric="cosine")
        result = mc.compute_all(bootstrap_ci=False)

        assert isinstance(result, EmbeddingMetrics)
        assert 0.0 <= result.mean_average_precision <= 1.0
        assert -1.0 <= result.silhouette_score <= 1.0
