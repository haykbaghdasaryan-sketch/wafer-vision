"""Unit tests: anomaly detection."""

import numpy as np
import pytest

from src.anomaly.detector import AnomalyDetector, AnomalyResult
from src.anomaly.threshold import compute_per_class_thresholds, compute_threshold


class TestComputeThreshold:
    """Tests for compute_threshold function."""

    def test_basic_threshold(self):
        """Threshold equals mean + n_sigma * std."""
        distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        mean = np.mean(distances)
        std = np.std(distances)
        result = compute_threshold(distances, n_sigma=2.0)
        expected = mean + 2.0 * std
        assert abs(result - expected) < 1e-6

    def test_zero_sigma_returns_mean(self):
        """n_sigma=0 returns just the mean."""
        distances = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
        result = compute_threshold(distances, n_sigma=0.0)
        expected = float(np.mean(distances))
        assert abs(result - expected) < 1e-6

    def test_constant_distances_zero_std(self):
        """All same distances yields threshold = mean (std=0)."""
        distances = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        result = compute_threshold(distances, n_sigma=3.0)
        assert abs(result - 5.0) < 1e-6

    def test_empty_distances_raises(self):
        """Empty distances raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_threshold(np.array([]), n_sigma=2.0)

    def test_invalid_n_sigma_raises(self):
        """n_sigma out of range raises ValueError."""
        distances = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="n_sigma must be in"):
            compute_threshold(distances, n_sigma=-0.1)
        with pytest.raises(ValueError, match="n_sigma must be in"):
            compute_threshold(distances, n_sigma=5.1)

    def test_single_value(self):
        """Single value: std=0, threshold=value."""
        distances = np.array([3.14])
        result = compute_threshold(distances, n_sigma=2.0)
        assert abs(result - 3.14) < 1e-6


class TestComputePerClassThresholds:
    """Tests for compute_per_class_thresholds function."""

    def test_basic_per_class(self):
        """Each class gets its own threshold."""
        distances = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0], dtype=np.float32)
        classes = np.array([0, 0, 0, 1, 1, 1])
        result = compute_per_class_thresholds(distances, classes, n_sigma=1.0)
        assert 0 in result
        assert 1 in result
        # Class 0: mean=2, std=0.8165..., threshold = 2 + 0.8165...
        # Class 1: mean=11, std=0.8165..., threshold = 11 + 0.8165...
        assert result[0] < result[1]

    def test_per_class_values(self):
        """Verify per-class threshold calculation."""
        distances = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
        classes = np.array([0, 0, 1, 1])
        result = compute_per_class_thresholds(distances, classes, n_sigma=2.0)
        # Class 0: mean=2.0, std=1.0, threshold=4.0
        # Class 1: mean=6.0, std=1.0, threshold=8.0
        assert abs(result[0] - 4.0) < 1e-6
        assert abs(result[1] - 8.0) < 1e-6

    def test_empty_raises(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_per_class_thresholds(np.array([]), np.array([]), n_sigma=2.0)

    def test_mismatched_lengths_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_per_class_thresholds(
                np.array([1.0, 2.0]), np.array([0]), n_sigma=2.0
            )


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    @pytest.fixture
    def simple_detector(self):
        """Create a simple detector with well-separated clusters."""
        np.random.seed(42)
        n_per_class = 50
        dim = 16
        embeddings = []
        labels = []
        for c in range(3):
            center = np.zeros(dim, dtype=np.float32)
            center[c] = 10.0  # Each class centered at different axis
            embs = center + np.random.randn(n_per_class, dim).astype(np.float32) * 0.5
            embeddings.append(embs)
            labels.extend([c] * n_per_class)
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        return AnomalyDetector(
            embeddings, labels, class_names=["ClassA", "ClassB", "ClassC"]
        )

    def test_centroid_computation(self, simple_detector):
        """Centroids should be close to the known centers."""
        centroids = simple_detector.centroids
        assert centroids.shape == (3, 16)
        # Class 0 should have high value at index 0
        assert centroids[0, 0] > 8.0
        # Class 1 should have high value at index 1
        assert centroids[1, 1] > 8.0
        # Class 2 should have high value at index 2
        assert centroids[2, 2] > 8.0

    def test_score_shape(self, simple_detector):
        """Score returns correct shape."""
        test = np.random.randn(20, 16).astype(np.float32)
        scores = simple_detector.score(test)
        assert scores.shape == (20,)
        assert np.all(scores >= 0)

    def test_score_near_centroid_is_low(self, simple_detector):
        """Samples near a centroid should have low scores."""
        centroid = simple_detector.centroids[0].copy()
        test = centroid.reshape(1, -1) + np.random.randn(1, 16).astype(np.float32) * 0.01
        score = simple_detector.score(test)
        assert score[0] < 1.0

    def test_score_far_from_centroids_is_high(self, simple_detector):
        """Samples far from all centroids should have high scores."""
        test = np.full((1, 16), 100.0, dtype=np.float32)
        score = simple_detector.score(test)
        assert score[0] > 50.0

    def test_detect_flags_outliers(self, simple_detector):
        """Detect should flag clear outliers."""
        # Generate an obvious outlier
        outlier = np.full((1, 16), 50.0, dtype=np.float32)
        # And some inliers near class 0
        inliers = simple_detector.centroids[0].reshape(1, -1) + np.random.randn(5, 16).astype(np.float32) * 0.1
        test = np.vstack([inliers, outlier])
        results = simple_detector.detect(test, n_std=2.0)
        # The outlier (index 5) should definitely be flagged
        flagged_indices = [r.index for r in results]
        assert 5 in flagged_indices

    def test_detect_confidence_bounds(self, simple_detector):
        """Confidence should always be in [0, 1]."""
        test = np.random.randn(50, 16).astype(np.float32) * 5
        results = simple_detector.detect(test, n_std=1.0)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_detect_sorted_descending(self, simple_detector):
        """Results should be sorted by distance descending."""
        test = np.random.randn(100, 16).astype(np.float32) * 10
        results = simple_detector.detect(test, n_std=1.0)
        if len(results) > 1:
            distances = [r.distance_to_nearest_centroid for r in results]
            for i in range(len(distances) - 1):
                assert distances[i] >= distances[i + 1]

    def test_detect_per_class(self, simple_detector):
        """Per-class thresholding should work."""
        test = np.random.randn(50, 16).astype(np.float32) * 5
        results = simple_detector.detect(test, n_std=2.0, per_class=True)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0
            assert r.nearest_class in [0, 1, 2]

    def test_detect_manual_threshold(self, simple_detector):
        """Manual threshold should be used when provided."""
        test = np.random.randn(20, 16).astype(np.float32)
        # Very high threshold → no anomalies
        results_high = simple_detector.detect(test, threshold=1000.0)
        assert len(results_high) == 0
        # Very low threshold → all anomalies
        results_low = simple_detector.detect(test, threshold=0.0)
        assert len(results_low) == 20

    def test_anomaly_result_fields(self, simple_detector):
        """AnomalyResult should have all expected fields."""
        outlier = np.full((1, 16), 50.0, dtype=np.float32)
        results = simple_detector.detect(outlier, threshold=0.1)
        assert len(results) == 1
        r = results[0]
        assert r.index == 0
        assert r.distance_to_nearest_centroid > 0
        assert r.nearest_class in [0, 1, 2]
        assert r.nearest_class_name in ["ClassA", "ClassB", "ClassC"]
        assert 0.0 <= r.confidence <= 1.0
        assert r.all_centroid_distances.shape == (3,)
        assert r.second_nearest_class in [0, 1, 2]
        assert r.second_nearest_distance >= r.distance_to_nearest_centroid

    def test_get_summary_empty(self, simple_detector):
        """Summary with no results returns zeros."""
        summary = simple_detector.get_summary([], total_evaluated=100)
        assert summary["total_evaluated"] == 100
        assert summary["anomalies_flagged"] == 0
        assert summary["anomaly_rate_pct"] == 0.0

    def test_get_summary_with_results(self, simple_detector):
        """Summary computes correct statistics."""
        test = np.random.randn(50, 16).astype(np.float32) * 10
        results = simple_detector.detect(test, n_std=1.0)
        summary = simple_detector.get_summary(results, total_evaluated=50)
        assert summary["total_evaluated"] == 50
        assert summary["anomalies_flagged"] == len(results)
        assert 0.0 <= summary["anomaly_rate_pct"] <= 100.0
        if len(results) > 0:
            assert summary["mean_distance"] > 0
            assert summary["max_distance"] >= summary["mean_distance"]
            assert summary["most_anomalous_class"] is not None

    def test_detect_lof(self, simple_detector):
        """LOF detection should return valid results."""
        # Generate clear outliers
        outliers = np.full((3, 16), 50.0, dtype=np.float32) + np.random.randn(3, 16).astype(np.float32)
        results = simple_detector.detect_lof(outliers, n_neighbors=10)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0
            assert r.nearest_class in [0, 1, 2]

    def test_auto_threshold_computation(self, simple_detector):
        """Auto threshold matches expected formula."""
        distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        threshold = simple_detector.auto_threshold(distances, n_std=2.0)
        expected = float(np.mean(distances) + 2.0 * np.std(distances))
        assert abs(threshold - expected) < 1e-6


class TestPerformance:
    """Performance tests for anomaly detection."""

    @pytest.mark.slow
    def test_scoring_25k_under_10s(self):
        """Scoring 25K embeddings should complete within 10 seconds on CPU."""
        import time

        np.random.seed(42)
        dim = 2048
        n_train = 5000
        n_test = 25000

        # Create training data with 9 classes
        train_embeddings = np.random.randn(n_train, dim).astype(np.float32)
        train_labels = np.random.randint(0, 9, size=n_train)
        test_embeddings = np.random.randn(n_test, dim).astype(np.float32)

        detector = AnomalyDetector(train_embeddings, train_labels)

        start = time.time()
        results = detector.detect(test_embeddings, n_std=2.0)
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Scoring took {elapsed:.2f}s, expected < 10s"
