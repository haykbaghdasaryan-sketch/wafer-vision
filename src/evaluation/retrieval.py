"""RetrievalEngine: brute-force nearest neighbor retrieval in embedding space.

Supports:
    - Query by pre-computed embedding or dataset index
    - Euclidean and cosine distance metrics
    - Self-exclusion (when querying by index)
    - Class-filtered retrieval
    - Batch queries (M queries -> (M, K) indices and distances)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.exceptions import DimensionMismatchError


@dataclass
class RetrievalResult:
    """Result container for retrieval queries.

    For single queries, arrays have shape (K,).
    For batch queries, arrays have shape (M, K).

    Attributes:
        indices: Indices of nearest neighbors in the database.
        distances: Distances to nearest neighbors (ascending order).
        labels: Class labels of the nearest neighbors.
    """

    indices: np.ndarray
    distances: np.ndarray
    labels: np.ndarray


class RetrievalEngine:
    """Nearest neighbor retrieval engine using brute-force distance computation.

    Efficient for datasets up to ~200K embeddings using optimized numpy operations.
    Supports Euclidean and cosine distance metrics.

    Args:
        embeddings: Database embeddings, shape (N, D).
        labels: Database labels, shape (N,).
        distance_metric: One of "euclidean" or "cosine".

    Raises:
        ValueError: If distance_metric is not supported or shapes are invalid.
    """

    SUPPORTED_METRICS = ("euclidean", "cosine")

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        distance_metric: str = "euclidean",
    ) -> None:
        if distance_metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported distance metric: '{distance_metric}'. "
                f"Supported: {list(self.SUPPORTED_METRICS)}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got shape {embeddings.shape}"
            )
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"Labels shape {labels.shape} does not match "
                f"embeddings count {embeddings.shape[0]}"
            )

        self._distance_metric = distance_metric
        self._labels = labels.astype(np.int64)
        self._n_samples = embeddings.shape[0]
        self._embedding_dim = embeddings.shape[1]

        # Store embeddings in contiguous float32 for BLAS efficiency
        if distance_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero for zero vectors
            norms = np.maximum(norms, 1e-10)
            self._embeddings = np.ascontiguousarray(
                (embeddings / norms).astype(np.float32)
            )
            # For cosine, pre-normalized so ||e||^2 = 1 for all
            self._embed_norms_sq = np.ones(self._n_samples, dtype=np.float32)
        else:
            self._embeddings = np.ascontiguousarray(
                embeddings.astype(np.float32)
            )
            # Precompute ||e||^2 for efficient Euclidean distance
            self._embed_norms_sq = np.sum(
                self._embeddings * self._embeddings, axis=1
            )

    @property
    def n_samples(self) -> int:
        """Number of samples in the database."""
        return self._n_samples

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of embeddings."""
        return self._embedding_dim

    @property
    def distance_metric(self) -> str:
        """The distance metric in use."""
        return self._distance_metric

    def _validate_query_dim(self, query: np.ndarray) -> None:
        """Validate query embedding dimensionality matches the database."""
        if query.shape[-1] != self._embedding_dim:
            raise DimensionMismatchError(
                expected=self._embedding_dim,
                actual=query.shape[-1],
                context="retrieval query",
            )

    def _compute_distances(self, query: np.ndarray) -> np.ndarray:
        """Compute distances from a single query to all database embeddings.

        Uses expanded form: ||q-e||^2 = ||q||^2 + ||e||^2 - 2*q.e
        with precomputed ||e||^2 and BLAS-accelerated dot product.

        Args:
            query: Shape (D,) query embedding.

        Returns:
            Shape (N,) array of distances.
        """
        if self._distance_metric == "euclidean":
            # ||q-e||^2 = ||q||^2 + ||e||^2 - 2 * q . e
            query_norm_sq = float(np.dot(query, query))
            # BLAS-accelerated matrix-vector product
            cross_terms = self._embeddings @ query  # (N,)
            sq_distances = self._embed_norms_sq + query_norm_sq - 2.0 * cross_terms
            # Clamp numerical errors
            np.maximum(sq_distances, 0.0, out=sq_distances)
            distances = np.sqrt(sq_distances)
        else:
            # Cosine distance: 1 - cosine_similarity
            # Embeddings are already normalized
            query_norm = np.linalg.norm(query)
            if query_norm < 1e-10:
                # Zero vector → max distance to all
                return np.ones(self._n_samples, dtype=np.float32)
            normalized_query = query / query_norm
            # BLAS-accelerated dot product
            similarities = self._embeddings @ normalized_query  # (N,)
            distances = 1.0 - similarities

        return distances.astype(np.float32)

    def _compute_distances_batch(self, queries: np.ndarray) -> np.ndarray:
        """Compute distances from M queries to all database embeddings.

        Uses BLAS-accelerated matrix multiplication for efficiency.

        Args:
            queries: Shape (M, D) query embeddings.

        Returns:
            Shape (M, N) array of distances.
        """
        if self._distance_metric == "euclidean":
            # ||q - e||^2 = ||q||^2 + ||e||^2 - 2 * q @ e.T
            query_sq = np.sum(queries * queries, axis=1, keepdims=True)  # (M, 1)
            # BLAS-accelerated matrix multiply
            cross = queries @ self._embeddings.T  # (M, N)
            sq_distances = query_sq + self._embed_norms_sq[np.newaxis, :] - 2.0 * cross
            # Clamp negative values from numerical errors
            np.maximum(sq_distances, 0.0, out=sq_distances)
            distances = np.sqrt(sq_distances)
        else:
            # Cosine distance: 1 - cosine_similarity
            # Normalize queries
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normalized_queries = queries / norms
            similarities = normalized_queries @ self._embeddings.T  # (M, N)
            distances = 1.0 - similarities

        return distances.astype(np.float32)

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_index: Optional[int] = None,
        filter_class: Optional[int] = None,
    ) -> RetrievalResult:
        """Return K nearest neighbors sorted by ascending distance.

        Args:
            query_embedding: Shape (D,) query embedding vector.
            k: Number of neighbors to return (range [1, 100]).
            exclude_index: Index to exclude from results (self-retrieval prevention).
            filter_class: If set, return only neighbors belonging to this class.

        Returns:
            RetrievalResult with indices, distances, and labels arrays of shape (K,).

        Raises:
            DimensionMismatchError: If query dim != database dim.
            ValueError: If k is out of valid range.
        """
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim != 1:
            raise ValueError(
                f"query_embedding must be 1D, got shape {query_embedding.shape}"
            )
        self._validate_query_dim(query_embedding)

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        # Compute distances to all database embeddings
        distances = self._compute_distances(query_embedding)

        # Build mask for valid candidates
        mask = np.ones(self._n_samples, dtype=bool)
        if exclude_index is not None:
            if 0 <= exclude_index < self._n_samples:
                mask[exclude_index] = False
        if filter_class is not None:
            mask &= self._labels == filter_class

        # Apply mask: set excluded distances to infinity
        distances[~mask] = np.inf

        # Get top-k indices (ascending distance)
        # Use argpartition for efficiency when k << N
        valid_count = int(mask.sum())
        actual_k = min(k, valid_count)

        if actual_k == 0:
            return RetrievalResult(
                indices=np.array([], dtype=np.int64),
                distances=np.array([], dtype=np.float32),
                labels=np.array([], dtype=np.int64),
            )

        # argpartition is O(N) for finding top-k
        if actual_k < self._n_samples:
            part_indices = np.argpartition(distances, actual_k)[:actual_k]
        else:
            part_indices = np.arange(self._n_samples)

        # Sort the top-k by distance
        sorted_order = np.argsort(distances[part_indices])
        top_indices = part_indices[sorted_order]

        return RetrievalResult(
            indices=top_indices.astype(np.int64),
            distances=distances[top_indices],
            labels=self._labels[top_indices],
        )

    def query_by_index(
        self,
        index: int,
        k: int = 10,
        filter_class: Optional[int] = None,
    ) -> RetrievalResult:
        """Query by dataset index with automatic self-exclusion.

        Uses the embedding at the given index as the query and excludes
        the index itself from the results.

        Args:
            index: Dataset index to use as query.
            k: Number of neighbors to return.
            filter_class: If set, return only neighbors belonging to this class.

        Returns:
            RetrievalResult with indices, distances, and labels arrays of shape (K,).

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= self._n_samples:
            raise IndexError(
                f"Index {index} out of range [0, {self._n_samples - 1}]"
            )

        query_embedding = self._embeddings[index]
        return self.query(
            query_embedding=query_embedding,
            k=k,
            exclude_index=index,
            filter_class=filter_class,
        )

    def query_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        exclude_indices: Optional[np.ndarray] = None,
    ) -> RetrievalResult:
        """Batch query for M queries.

        Args:
            query_embeddings: Shape (M, D) query embeddings.
            k: Number of neighbors per query.
            exclude_indices: Optional shape (M,) array of indices to exclude
                per query (for self-retrieval prevention in batch mode).

        Returns:
            RetrievalResult with indices, distances, and labels of shape (M, K).

        Raises:
            DimensionMismatchError: If query dim != database dim.
            ValueError: If shapes are invalid.
        """
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim != 2:
            raise ValueError(
                f"query_embeddings must be 2D, got shape {query_embeddings.shape}"
            )
        self._validate_query_dim(query_embeddings)

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        m = query_embeddings.shape[0]

        # Compute all pairwise distances: (M, N)
        all_distances = self._compute_distances_batch(query_embeddings)

        # Handle exclusion indices
        if exclude_indices is not None:
            exclude_indices = np.asarray(exclude_indices, dtype=np.int64)
            if exclude_indices.shape[0] != m:
                raise ValueError(
                    f"exclude_indices length {exclude_indices.shape[0]} "
                    f"does not match number of queries {m}"
                )
            # Set excluded distances to infinity
            for i in range(m):
                idx = exclude_indices[i]
                if 0 <= idx < self._n_samples:
                    all_distances[i, idx] = np.inf

        # Find top-k for each query
        actual_k = min(k, self._n_samples)
        if exclude_indices is not None:
            # In worst case, one fewer candidate per query
            actual_k = min(k, self._n_samples - 1)

        # Use argpartition for each row
        result_indices = np.zeros((m, actual_k), dtype=np.int64)
        result_distances = np.zeros((m, actual_k), dtype=np.float32)

        for i in range(m):
            row_distances = all_distances[i]
            # Count valid (non-inf) entries
            valid_count = int(np.isfinite(row_distances).sum())
            row_k = min(actual_k, valid_count)

            if row_k == 0:
                result_indices[i] = 0
                result_distances[i] = np.inf
                continue

            if row_k < self._n_samples:
                part_idx = np.argpartition(row_distances, row_k)[:row_k]
            else:
                part_idx = np.arange(self._n_samples)

            sorted_order = np.argsort(row_distances[part_idx])
            top_idx = part_idx[sorted_order]

            # Pad if row_k < actual_k
            if row_k < actual_k:
                padded_idx = np.full(actual_k, top_idx[-1], dtype=np.int64)
                padded_idx[:row_k] = top_idx
                padded_dist = np.full(actual_k, np.inf, dtype=np.float32)
                padded_dist[:row_k] = row_distances[top_idx]
                result_indices[i] = padded_idx
                result_distances[i] = padded_dist
            else:
                result_indices[i] = top_idx
                result_distances[i] = row_distances[top_idx]

        result_labels = self._labels[result_indices]

        return RetrievalResult(
            indices=result_indices,
            distances=result_distances,
            labels=result_labels,
        )
