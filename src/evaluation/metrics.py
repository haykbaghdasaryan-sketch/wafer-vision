"""MetricsComputer with all 7+ embedding quality metrics.

Computes KNN Accuracy, Recall@K, Precision@K, Silhouette Score, NMI,
Mean Average Precision, inter/intra class distances and separability index.
Supports both Euclidean and cosine distance metrics.
Uses exact NN for < 50K samples, FAISS IVF for larger sets.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score as sklearn_silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from src.exceptions import DimensionMismatchError

logger = logging.getLogger(__name__)

# Threshold for switching from exact NN to FAISS IVF
_EXACT_NN_THRESHOLD = 50_000


@dataclass
class EmbeddingMetrics:
    """Structured container for all embedding quality metrics.

    All retrieval metrics are in [0, 1].
    Silhouette in [-1, 1]. Distances are non-negative.
    """

    knn_accuracy: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    silhouette_score: float = 0.0
    nmi: float = 0.0
    mean_average_precision: float = 0.0
    inter_class_distance: float = 0.0
    intra_class_distance: float = 0.0
    separability_index: float = 0.0
    confidence_intervals: Optional[dict[str, tuple[float, float]]] = None
    per_class_metrics: Optional[dict[str, dict]] = None


class MetricsComputer:
    """Computes all embedding quality metrics.

    Supports both Euclidean and cosine distance.
    Uses exact NN for < 50K samples, FAISS IVF for larger sets.

    Args:
        embeddings: Array of shape (N, D).
        labels: Array of shape (N,), integer class labels.
        distance_metric: "euclidean" or "cosine".
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        distance_metric: str = "euclidean",
    ) -> None:
        """Initialize metrics computer.

        Raises:
            DimensionMismatchError: If embeddings.shape[0] != labels.shape[0].
            ValueError: If fewer than 2 classes present or invalid distance metric.
        """
        if embeddings.shape[0] != labels.shape[0]:
            raise DimensionMismatchError(
                expected=embeddings.shape[0],
                actual=labels.shape[0],
                context="embeddings vs labels length mismatch",
            )
        if distance_metric not in ("euclidean", "cosine"):
            raise ValueError(
                f"Invalid distance_metric '{distance_metric}'. Must be 'euclidean' or 'cosine'."
            )

        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            raise ValueError(
                f"At least 2 classes required, got {len(unique_classes)}."
            )

        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.distance_metric = distance_metric
        self.n_samples = embeddings.shape[0]
        self.n_dims = embeddings.shape[1]
        self.unique_classes = unique_classes

        # For cosine distance, L2-normalize embeddings
        if distance_metric == "cosine":
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # avoid division by zero
            self.embeddings = self.embeddings / norms

        # Build nearest neighbor index
        self._nn_index = None
        self._nn_distances = None
        self._nn_indices = None

        logger.info(
            "MetricsComputer initialized: N=%d, D=%d, classes=%d, metric=%s",
            self.n_samples,
            self.n_dims,
            len(self.unique_classes),
            distance_metric,
        )

    def _get_sklearn_metric(self) -> str:
        """Get sklearn-compatible metric string."""
        if self.distance_metric == "cosine":
            return "euclidean"  # already L2-normalized, so euclidean ~ cosine
        return "euclidean"

    def _build_nn_index(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Build or retrieve nearest neighbor index.

        For < 50K samples uses exact brute-force NN.
        For larger sets uses FAISS IVF approximate NN.

        Returns:
            (distances, indices) arrays of shape (N, k).
        """
        max_k = k + 1  # +1 because query itself may be included

        if self.n_samples < _EXACT_NN_THRESHOLD:
            logger.info("Using exact NN (brute-force) for %d samples", self.n_samples)
            nn = NearestNeighbors(
                n_neighbors=max_k,
                metric=self._get_sklearn_metric(),
                algorithm="brute",
            )
            nn.fit(self.embeddings)
            distances, indices = nn.kneighbors(self.embeddings)
        else:
            logger.info("Using FAISS IVF for %d samples", self.n_samples)
            distances, indices = self._faiss_knn(max_k)

        return distances, indices

    def _faiss_knn(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Use FAISS IVF for approximate nearest neighbor search."""
        import faiss

        d = self.n_dims
        # Number of clusters for IVF
        n_clusters = min(int(np.sqrt(self.n_samples)), 256)
        nprobe = min(n_clusters // 4, 32)

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_clusters)
        index.train(self.embeddings)
        index.add(self.embeddings)
        index.nprobe = max(nprobe, 1)

        distances, indices = index.search(self.embeddings, k)
        return distances, indices

    def compute_all(
        self, bootstrap_ci: bool = True, n_bootstrap: int = 1000
    ) -> EmbeddingMetrics:
        """Compute all metrics and return structured result.

        Args:
            bootstrap_ci: Whether to compute 95% confidence intervals.
            n_bootstrap: Number of bootstrap iterations.

        Returns:
            EmbeddingMetrics with all fields populated.
        """
        knn_accuracy = self.compute_knn_accuracy()
        recall_at_k = self.compute_recall_at_k()
        precision_at_k = self.compute_precision_at_k()
        silhouette = self.compute_silhouette()
        nmi = self.compute_nmi()
        map_score = self.compute_map()
        inter_dist, intra_dist, sep_index = self.compute_separability()
        per_class = self.compute_per_class_metrics()

        ci = None
        if bootstrap_ci:
            ci = self._compute_bootstrap_ci(n_bootstrap=n_bootstrap)

        return EmbeddingMetrics(
            knn_accuracy=knn_accuracy,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            silhouette_score=silhouette,
            nmi=nmi,
            mean_average_precision=map_score,
            inter_class_distance=inter_dist,
            intra_class_distance=intra_dist,
            separability_index=sep_index,
            confidence_intervals=ci,
            per_class_metrics=per_class,
        )

    def compute_knn_accuracy(self, k_values: list[int] | None = None) -> dict[int, float]:
        """Compute KNN classification accuracy for given k values.

        Uses leave-one-out: each sample is classified by the majority vote
        of its k nearest neighbors (excluding itself).

        Args:
            k_values: List of k values to evaluate. Defaults to [1, 3, 5, 10].

        Returns:
            Dict mapping k -> accuracy (as fraction in [0, 1]).
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        max_k = max(k_values)
        distances, indices = self._build_nn_index(max_k)

        # Exclude self (first column is self with distance 0)
        neighbor_indices = indices[:, 1: max_k + 1]
        neighbor_labels = self.labels[neighbor_indices]

        results = {}
        for k in k_values:
            k_labels = neighbor_labels[:, :k]
            # Majority vote
            predictions = np.array([
                np.bincount(row, minlength=len(self.unique_classes)).argmax()
                for row in k_labels
            ])
            accuracy = np.mean(predictions == self.labels)
            results[k] = float(accuracy)

        return results

    def compute_recall_at_k(self, k_values: list[int] | None = None) -> dict[int, float]:
        """Compute Recall@K: fraction of queries with at least one correct neighbor in top-K.

        Excludes the query itself from candidates.

        Args:
            k_values: List of k values. Defaults to [1, 5, 10].

        Returns:
            Dict mapping k -> recall (fraction in [0, 1]).
        """
        if k_values is None:
            k_values = [1, 5, 10]

        max_k = max(k_values)
        distances, indices = self._build_nn_index(max_k)

        # Exclude self (first column)
        neighbor_indices = indices[:, 1: max_k + 1]
        neighbor_labels = self.labels[neighbor_indices]

        results = {}
        for k in k_values:
            k_labels = neighbor_labels[:, :k]
            # For each query, check if any of the k neighbors has the same class
            hits = np.any(k_labels == self.labels[:, np.newaxis], axis=1)
            results[k] = float(np.mean(hits))

        return results

    def compute_precision_at_k(self, k_values: list[int] | None = None) -> dict[int, float]:
        """Compute Precision@K: fraction of top-K neighbors belonging to same class.

        Averaged over all queries.

        Args:
            k_values: List of k values. Defaults to [1, 5, 10].

        Returns:
            Dict mapping k -> precision (fraction in [0, 1]).
        """
        if k_values is None:
            k_values = [1, 5, 10]

        max_k = max(k_values)
        distances, indices = self._build_nn_index(max_k)

        # Exclude self
        neighbor_indices = indices[:, 1: max_k + 1]
        neighbor_labels = self.labels[neighbor_indices]

        results = {}
        for k in k_values:
            k_labels = neighbor_labels[:, :k]
            # For each query, count how many of k neighbors have same class
            matches = (k_labels == self.labels[:, np.newaxis]).sum(axis=1)
            precision = matches / k
            results[k] = float(np.mean(precision))

        return results

    def compute_silhouette(self, subsample_n: int = 10000) -> float:
        """Compute Silhouette Score on a subsample for efficiency.

        Args:
            subsample_n: Maximum number of samples to use.

        Returns:
            Mean silhouette score in [-1, 1].
        """
        if self.n_samples <= subsample_n:
            sample_emb = self.embeddings
            sample_labels = self.labels
        else:
            rng = np.random.RandomState(42)
            idx = rng.choice(self.n_samples, size=subsample_n, replace=False)
            sample_emb = self.embeddings[idx]
            sample_labels = self.labels[idx]

        # Need at least 2 unique labels in the subsample
        if len(np.unique(sample_labels)) < 2:
            logger.warning("Silhouette: fewer than 2 classes in subsample, returning 0.0")
            return 0.0

        score = sklearn_silhouette_score(
            sample_emb, sample_labels, metric="euclidean"
        )
        return float(score)

    def compute_nmi(self, n_clusters: int = 9, n_init: int = 10) -> float:
        """Compute NMI between K-means cluster assignments and ground truth.

        Args:
            n_clusters: Number of clusters for K-means (default 9 for 9 classes).
            n_init: Number of random initializations.

        Returns:
            NMI score in [0, 1].
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=42,
            max_iter=300,
        )
        cluster_labels = kmeans.fit_predict(self.embeddings)
        nmi = normalized_mutual_info_score(self.labels, cluster_labels)
        return float(nmi)

    def compute_map(self, max_neighbors: int = 100) -> float:
        """Compute Mean Average Precision over all queries.

        For each query, average precision is the area under the precision-recall
        curve when neighbors are ranked by distance. Uses top-K neighbors
        to avoid O(N^2) memory for large datasets.

        Args:
            max_neighbors: Maximum number of neighbors to consider per query.
                Limits memory usage while providing a good MAP approximation.

        Returns:
            MAP score in [0, 1].
        """
        # Limit neighbors to avoid memory explosion on large datasets
        # MAP@100 is a standard approximation for full MAP
        limit_k = min(max_neighbors, self.n_samples - 1)

        nn = NearestNeighbors(
            n_neighbors=limit_k + 1,  # +1 because self is included
            metric=self._get_sklearn_metric(),
            algorithm="brute",
        )
        nn.fit(self.embeddings)
        _, indices = nn.kneighbors(self.embeddings)
        # Exclude self (first column)
        neighbor_indices = indices[:, 1:]

        neighbor_labels = self.labels[neighbor_indices]
        query_labels = self.labels[:, np.newaxis]

        # relevance matrix: 1 if neighbor has same class as query
        relevance = (neighbor_labels == query_labels).astype(np.float64)

        # Compute average precision per query
        ap_scores = []
        for i in range(self.n_samples):
            rel = relevance[i]
            if rel.sum() == 0:
                ap_scores.append(0.0)
                continue
            # Cumulative sum of relevance
            cum_rel = np.cumsum(rel)
            # Precision at each position
            positions = np.arange(1, len(rel) + 1)
            precisions = cum_rel / positions
            # AP = sum of (precision * relevance) / total relevant
            ap = np.sum(precisions * rel) / rel.sum()
            ap_scores.append(ap)

        return float(np.mean(ap_scores))

    def compute_separability(self) -> tuple[float, float, float]:
        """Compute inter-class and intra-class distances and separability index.

        Inter-class: average pairwise distance between class centroids.
        Intra-class: average distance from samples to their class centroid.
        Separability index: inter / intra ratio.

        Returns:
            Tuple of (inter_class_dist, intra_class_dist, separability_index).
        """
        # Compute class centroids
        centroids = {}
        for c in self.unique_classes:
            mask = self.labels == c
            centroids[c] = self.embeddings[mask].mean(axis=0)

        centroid_array = np.array([centroids[c] for c in self.unique_classes])

        # Inter-class distance: average pairwise distance between centroids
        n_classes = len(self.unique_classes)
        inter_distances = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                dist = np.linalg.norm(centroid_array[i] - centroid_array[j])
                inter_distances.append(dist)
        inter_class_dist = float(np.mean(inter_distances)) if inter_distances else 0.0

        # Intra-class distance: average distance from samples to their centroid
        intra_distances = []
        for c in self.unique_classes:
            mask = self.labels == c
            class_embeddings = self.embeddings[mask]
            centroid = centroids[c]
            dists = np.linalg.norm(class_embeddings - centroid, axis=1)
            intra_distances.extend(dists.tolist())
        intra_class_dist = float(np.mean(intra_distances)) if intra_distances else 0.0

        # Separability index
        if intra_class_dist > 0:
            separability_index = inter_class_dist / intra_class_dist
        else:
            separability_index = float("inf") if inter_class_dist > 0 else 0.0

        return inter_class_dist, intra_class_dist, separability_index

    def compute_per_class_metrics(self) -> dict[str, dict]:
        """Compute per-class breakdown of key metrics.

        Returns:
            Dict mapping class label (as string) to dict with:
            - knn_accuracy_k5: KNN accuracy for k=5 within this class
            - precision_at_5: Precision@5 for queries of this class
            - recall_at_5: Recall@5 for queries of this class
            - intra_class_distance: Average intra-class distance
            - n_samples: Number of samples in this class
        """
        max_k = 10
        distances, indices = self._build_nn_index(max_k)
        neighbor_indices = indices[:, 1: max_k + 1]
        neighbor_labels = self.labels[neighbor_indices]

        # Compute centroids for intra-class distance
        centroids = {}
        for c in self.unique_classes:
            mask = self.labels == c
            centroids[c] = self.embeddings[mask].mean(axis=0)

        per_class = {}
        for c in self.unique_classes:
            mask = self.labels == c
            class_indices = np.where(mask)[0]
            n_class = int(mask.sum())

            # KNN accuracy for k=5 on this class
            k5_labels = neighbor_labels[class_indices, :5]
            k5_preds = np.array([
                np.bincount(row, minlength=len(self.unique_classes)).argmax()
                for row in k5_labels
            ])
            knn_acc = float(np.mean(k5_preds == c))

            # Precision@5 for this class
            k5_all = neighbor_labels[class_indices, :5]
            matches = (k5_all == c).sum(axis=1)
            prec5 = float(np.mean(matches / 5.0))

            # Recall@5 for this class
            hits = np.any(k5_all == c, axis=1)
            rec5 = float(np.mean(hits))

            # Intra-class distance
            class_embeddings = self.embeddings[class_indices]
            centroid = centroids[c]
            intra_dists = np.linalg.norm(class_embeddings - centroid, axis=1)
            intra_dist = float(np.mean(intra_dists))

            per_class[str(c)] = {
                "knn_accuracy_k5": knn_acc,
                "precision_at_5": prec5,
                "recall_at_5": rec5,
                "intra_class_distance": intra_dist,
                "n_samples": n_class,
            }

        return per_class

    def _compute_bootstrap_ci(
        self, n_bootstrap: int = 1000, confidence: float = 0.95
    ) -> dict[str, tuple[float, float]]:
        """Compute 95% bootstrap confidence intervals for KNN accuracy.

        Resamples labels/predictions n_bootstrap times, computes metric each time,
        takes 2.5/97.5 percentiles.

        Args:
            n_bootstrap: Number of bootstrap iterations.
            confidence: Confidence level (default 0.95).

        Returns:
            Dict mapping metric name (e.g., "knn_1", "knn_3") to (lower, upper) CI.
        """
        max_k = 10
        distances, indices = self._build_nn_index(max_k)
        neighbor_indices = indices[:, 1: max_k + 1]
        neighbor_labels = self.labels[neighbor_indices]

        alpha = (1 - confidence) / 2
        rng = np.random.RandomState(42)
        n = self.n_samples

        ci_results = {}

        for k in [1, 3, 5, 10]:
            k_labels = neighbor_labels[:, :k]
            # Pre-compute predictions for all samples
            predictions = np.array([
                np.bincount(row, minlength=len(self.unique_classes)).argmax()
                for row in k_labels
            ])
            correct = (predictions == self.labels).astype(np.float64)

            # Bootstrap
            bootstrap_accs = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                sample_idx = rng.randint(0, n, size=n)
                bootstrap_accs[b] = correct[sample_idx].mean()

            lower = float(np.percentile(bootstrap_accs, alpha * 100))
            upper = float(np.percentile(bootstrap_accs, (1 - alpha) * 100))
            ci_results[f"knn_{k}"] = (lower, upper)

        return ci_results

    def save_csv(
        self,
        path: Path,
        model_name: str = "unknown",
        training_mode: str = "unknown",
    ) -> None:
        """Save metrics to CSV with full context.

        Columns: model_name, training_mode, metric_name, metric_value,
        timestamp, n_queries, database_size, distance_metric.

        Args:
            path: Path to output CSV file.
            model_name: Name of the model that produced embeddings.
            training_mode: Training mode used.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().isoformat()
        rows = []

        # Compute all metrics first
        metrics = self.compute_all(bootstrap_ci=False)

        # KNN accuracy
        for k, v in metrics.knn_accuracy.items():
            rows.append(self._make_row(
                model_name, training_mode, f"knn_accuracy_k{k}", v, timestamp
            ))

        # Recall@K
        for k, v in metrics.recall_at_k.items():
            rows.append(self._make_row(
                model_name, training_mode, f"recall_at_k{k}", v, timestamp
            ))

        # Precision@K
        for k, v in metrics.precision_at_k.items():
            rows.append(self._make_row(
                model_name, training_mode, f"precision_at_k{k}", v, timestamp
            ))

        # Scalar metrics
        rows.append(self._make_row(
            model_name, training_mode, "silhouette_score", metrics.silhouette_score, timestamp
        ))
        rows.append(self._make_row(
            model_name, training_mode, "nmi", metrics.nmi, timestamp
        ))
        rows.append(self._make_row(
            model_name, training_mode, "mean_average_precision",
            metrics.mean_average_precision, timestamp
        ))
        rows.append(self._make_row(
            model_name, training_mode, "inter_class_distance",
            metrics.inter_class_distance, timestamp
        ))
        rows.append(self._make_row(
            model_name, training_mode, "intra_class_distance",
            metrics.intra_class_distance, timestamp
        ))
        rows.append(self._make_row(
            model_name, training_mode, "separability_index",
            metrics.separability_index, timestamp
        ))

        # Write CSV
        fieldnames = [
            "model_name", "training_mode", "metric_name", "metric_value",
            "timestamp", "n_queries", "database_size", "distance_metric",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Metrics saved to %s (%d rows)", path, len(rows))

    def _make_row(
        self,
        model_name: str,
        training_mode: str,
        metric_name: str,
        metric_value: float,
        timestamp: str,
    ) -> dict:
        """Create a single CSV row dict."""
        return {
            "model_name": model_name,
            "training_mode": training_mode,
            "metric_name": metric_name,
            "metric_value": f"{metric_value:.6f}",
            "timestamp": timestamp,
            "n_queries": self.n_samples,
            "database_size": self.n_samples,
            "distance_metric": self.distance_metric,
        }

    @staticmethod
    def load_csv(path: Path) -> EmbeddingMetrics:
        """Load metrics from CSV file and reconstruct EmbeddingMetrics.

        Args:
            path: Path to the CSV file.

        Returns:
            Reconstructed EmbeddingMetrics (round-trip support).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {path}")

        knn_accuracy: dict[int, float] = {}
        recall_at_k: dict[int, float] = {}
        precision_at_k: dict[int, float] = {}
        silhouette = 0.0
        nmi = 0.0
        map_score = 0.0
        inter_class_dist = 0.0
        intra_class_dist = 0.0
        sep_index = 0.0

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["metric_name"]
                value = float(row["metric_value"])

                if name.startswith("knn_accuracy_k"):
                    k = int(name.replace("knn_accuracy_k", ""))
                    knn_accuracy[k] = value
                elif name.startswith("recall_at_k"):
                    k = int(name.replace("recall_at_k", ""))
                    recall_at_k[k] = value
                elif name.startswith("precision_at_k"):
                    k = int(name.replace("precision_at_k", ""))
                    precision_at_k[k] = value
                elif name == "silhouette_score":
                    silhouette = value
                elif name == "nmi":
                    nmi = value
                elif name == "mean_average_precision":
                    map_score = value
                elif name == "inter_class_distance":
                    inter_class_dist = value
                elif name == "intra_class_distance":
                    intra_class_dist = value
                elif name == "separability_index":
                    sep_index = value

        return EmbeddingMetrics(
            knn_accuracy=knn_accuracy,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            silhouette_score=silhouette,
            nmi=nmi,
            mean_average_precision=map_score,
            inter_class_distance=inter_class_dist,
            intra_class_distance=intra_class_dist,
            separability_index=sep_index,
        )
