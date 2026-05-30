"""AnomalyDetector: centroid-based and LOF methods for OOD detection."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from src.anomaly.threshold import compute_per_class_thresholds, compute_threshold
from src.exceptions import ModelError


@dataclass
class AnomalyResult:
    """Single anomaly detection result."""

    index: int
    distance_to_nearest_centroid: float
    nearest_class: int
    nearest_class_name: str
    confidence: float  # In [0, 1]
    all_centroid_distances: np.ndarray  # Shape (num_classes,)
    second_nearest_class: int
    second_nearest_distance: float


class AnomalyDetector:
    """OOD detection based on embedding space distances to class centroids.

    Methods:
        - Centroid distance: flag samples > threshold from nearest centroid
        - LOF (alternative): Local Outlier Factor comparison baseline
        - Per-class thresholding: separate threshold per class

    Args:
        embeddings: Training embeddings for centroid computation, shape (N, D).
        labels: Training labels, shape (N,).
        class_names: Optional list of human-readable class names.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[list[str]] = None,
    ) -> None:
        self._embeddings = embeddings.astype(np.float32)
        self._labels = labels
        self._num_classes = int(np.max(labels)) + 1
        self._class_names = class_names or [str(i) for i in range(self._num_classes)]
        self._centroids: Optional[np.ndarray] = None
        self.compute_centroids()

    def compute_centroids(self) -> np.ndarray:
        """Compute class centroids as mean of class members.

        Returns:
            Centroid matrix, shape (num_classes, D).
            Centroid for class c = mean(embeddings where label == c).
        """
        dim = self._embeddings.shape[1]
        centroids = np.zeros((self._num_classes, dim), dtype=np.float32)

        for c in range(self._num_classes):
            mask = self._labels == c
            if np.any(mask):
                centroids[c] = np.mean(self._embeddings[mask], axis=0)

        self._centroids = centroids
        return centroids

    @property
    def centroids(self) -> np.ndarray:
        """Return computed centroids, raising if not yet computed."""
        if self._centroids is None:
            raise ModelError(
                "Centroids have not been computed. "
                "Call compute_centroids() or ensure the detector was initialized with training data."
            )
        return self._centroids

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """Compute distance of each test embedding to nearest centroid.

        Args:
            test_embeddings: Shape (M, D).

        Returns:
            Distances array, shape (M,). Each entry is the minimum
            Euclidean distance to any class centroid.

        Raises:
            ModelError: If centroids not computed.
        """
        centroids = self.centroids
        test_embeddings = test_embeddings.astype(np.float32)

        # Compute pairwise distances: (M, num_classes)
        all_distances = self._compute_all_distances(test_embeddings)

        # Min distance to nearest centroid
        return np.min(all_distances, axis=1)

    def _compute_all_distances(self, test_embeddings: np.ndarray) -> np.ndarray:
        """Compute distances from each test embedding to all centroids.

        Args:
            test_embeddings: Shape (M, D).

        Returns:
            Distance matrix, shape (M, num_classes).
        """
        centroids = self.centroids
        # Efficient vectorized Euclidean distance computation
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        test_sq = np.sum(test_embeddings ** 2, axis=1, keepdims=True)  # (M, 1)
        cent_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, C)
        cross = test_embeddings @ centroids.T  # (M, C)
        dist_sq = test_sq + cent_sq - 2 * cross
        # Clamp negative values from floating point errors
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.sqrt(dist_sq)

    def auto_threshold(self, distances: np.ndarray, n_std: float = 2.0) -> float:
        """Compute threshold = mean(distances) + n_std × std(distances).

        Args:
            distances: Array of distances.
            n_std: Number of standard deviations (range [0.0, 5.0]).
                   n_std=0 flags everything above mean.

        Returns:
            Threshold value (float).
        """
        return compute_threshold(distances, n_sigma=n_std)

    def per_class_thresholds(
        self, distances: np.ndarray, assigned_classes: np.ndarray, n_std: float = 2.0
    ) -> dict[int, float]:
        """Compute per-class thresholds.

        For each class c: threshold_c = mean(distances_c) + n_std × std(distances_c).

        Returns:
            Dict mapping class_index → threshold.
        """
        return compute_per_class_thresholds(distances, assigned_classes, n_sigma=n_std)

    def detect(
        self,
        test_embeddings: np.ndarray,
        threshold: Optional[float] = None,
        n_std: float = 2.0,
        per_class: bool = False,
    ) -> list[AnomalyResult]:
        """Flag samples exceeding threshold as anomalies.

        Invariant: flagged iff distance > threshold. Not-flagged iff distance ≤ threshold.
        Confidence = 1 - (threshold / distance), clamped to [0, 1].

        Args:
            test_embeddings: Shape (M, D).
            threshold: Manual threshold. Auto-computed if None.
            n_std: For auto-threshold computation.
            per_class: Use per-class thresholds instead of global.

        Returns:
            List of AnomalyResults, sorted by distance descending.
        """
        test_embeddings = test_embeddings.astype(np.float32)
        all_distances = self._compute_all_distances(test_embeddings)  # (M, num_classes)

        # Nearest and second-nearest classes
        sorted_indices = np.argsort(all_distances, axis=1)
        nearest_classes = sorted_indices[:, 0]
        second_nearest_classes = sorted_indices[:, 1]

        min_distances = all_distances[np.arange(len(all_distances)), nearest_classes]
        second_distances = all_distances[np.arange(len(all_distances)), second_nearest_classes]

        if per_class:
            # Per-class thresholding
            class_thresholds = self.per_class_thresholds(min_distances, nearest_classes, n_std)
            # Determine which samples are flagged
            flagged_mask = np.zeros(len(test_embeddings), dtype=bool)
            sample_thresholds = np.zeros(len(test_embeddings), dtype=np.float32)
            for cls, thresh in class_thresholds.items():
                cls_mask = nearest_classes == cls
                sample_thresholds[cls_mask] = thresh
                flagged_mask[cls_mask] = min_distances[cls_mask] > thresh
        else:
            # Global threshold
            if threshold is None:
                threshold = self.auto_threshold(min_distances, n_std)
            flagged_mask = min_distances > threshold
            sample_thresholds = np.full(len(test_embeddings), threshold, dtype=np.float32)

        # Build results for flagged samples
        results: list[AnomalyResult] = []
        flagged_indices = np.where(flagged_mask)[0]

        for idx in flagged_indices:
            dist = float(min_distances[idx])
            t = float(sample_thresholds[idx])
            # Confidence = 1 - (threshold / distance), clamped to [0, 1]
            if dist > 0:
                confidence = max(0.0, min(1.0, 1.0 - (t / dist)))
            else:
                confidence = 0.0

            result = AnomalyResult(
                index=int(idx),
                distance_to_nearest_centroid=dist,
                nearest_class=int(nearest_classes[idx]),
                nearest_class_name=self._class_names[int(nearest_classes[idx])],
                confidence=confidence,
                all_centroid_distances=all_distances[idx].copy(),
                second_nearest_class=int(second_nearest_classes[idx]),
                second_nearest_distance=float(second_distances[idx]),
            )
            results.append(result)

        # Sort by distance descending
        results.sort(key=lambda r: r.distance_to_nearest_centroid, reverse=True)
        return results

    def detect_lof(
        self, test_embeddings: np.ndarray, n_neighbors: int = 20
    ) -> list[AnomalyResult]:
        """Alternative detection using Local Outlier Factor.

        Uses training embeddings to fit the LOF model, then predicts on test set.
        Samples with LOF score < -1 (outliers) are flagged.

        Args:
            test_embeddings: Shape (M, D).
            n_neighbors: LOF parameter (default 20).

        Returns:
            List of AnomalyResults flagged by LOF, sorted by distance descending.
        """
        test_embeddings = test_embeddings.astype(np.float32)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(self._embeddings)

        # LOF scores: negative means outlier, lower = more anomalous
        lof_scores = lof.decision_function(test_embeddings)
        flagged_mask = lof_scores < 0  # Outliers have negative decision function

        # Compute centroid-based info for flagged samples
        all_distances = self._compute_all_distances(test_embeddings)
        sorted_indices = np.argsort(all_distances, axis=1)
        nearest_classes = sorted_indices[:, 0]
        second_nearest_classes = sorted_indices[:, 1]
        min_distances = all_distances[np.arange(len(all_distances)), nearest_classes]
        second_distances = all_distances[np.arange(len(all_distances)), second_nearest_classes]

        results: list[AnomalyResult] = []
        flagged_indices = np.where(flagged_mask)[0]

        for idx in flagged_indices:
            dist = float(min_distances[idx])
            # For LOF, use abs(lof_score) as a proxy for confidence
            confidence = max(0.0, min(1.0, -float(lof_scores[idx])))

            result = AnomalyResult(
                index=int(idx),
                distance_to_nearest_centroid=dist,
                nearest_class=int(nearest_classes[idx]),
                nearest_class_name=self._class_names[int(nearest_classes[idx])],
                confidence=confidence,
                all_centroid_distances=all_distances[idx].copy(),
                second_nearest_class=int(second_nearest_classes[idx]),
                second_nearest_distance=float(second_distances[idx]),
            )
            results.append(result)

        # Sort by distance descending
        results.sort(key=lambda r: r.distance_to_nearest_centroid, reverse=True)
        return results

    def get_summary(
        self, results: list[AnomalyResult], total_evaluated: Optional[int] = None
    ) -> dict:
        """Compute summary statistics for detection results.

        Args:
            results: List of AnomalyResult from detect() or detect_lof().
            total_evaluated: Total number of samples evaluated. If None,
                inferred as max(index) + 1 from results.

        Returns:
            Dict with: total_evaluated, anomalies_flagged, anomaly_rate_pct,
            mean_distance, median_distance, max_distance,
            most_anomalous_class, per_class_anomaly_counts.
        """
        n_flagged = len(results)

        if total_evaluated is None:
            if n_flagged > 0:
                total_evaluated = max(r.index for r in results) + 1
            else:
                total_evaluated = 0

        if n_flagged == 0:
            return {
                "total_evaluated": total_evaluated,
                "anomalies_flagged": 0,
                "anomaly_rate_pct": 0.0,
                "mean_distance": 0.0,
                "median_distance": 0.0,
                "max_distance": 0.0,
                "most_anomalous_class": None,
                "per_class_anomaly_counts": {},
            }

        distances = np.array([r.distance_to_nearest_centroid for r in results])
        anomaly_rate = (n_flagged / total_evaluated) * 100.0 if total_evaluated > 0 else 0.0

        # Per-class anomaly counts
        per_class_counts: dict[str, int] = {}
        for r in results:
            cls_name = r.nearest_class_name
            per_class_counts[cls_name] = per_class_counts.get(cls_name, 0) + 1

        # Most anomalous class
        most_anomalous_class = max(per_class_counts, key=per_class_counts.get)  # type: ignore[arg-type]

        return {
            "total_evaluated": total_evaluated,
            "anomalies_flagged": n_flagged,
            "anomaly_rate_pct": float(anomaly_rate),
            "mean_distance": float(np.mean(distances)),
            "median_distance": float(np.median(distances)),
            "max_distance": float(np.max(distances)),
            "most_anomalous_class": most_anomalous_class,
            "per_class_anomaly_counts": per_class_counts,
        }
