"""Unit tests: retrieval engine."""

import numpy as np
import pytest

from src.evaluation.retrieval import RetrievalEngine, RetrievalResult
from src.exceptions import DimensionMismatchError


@pytest.fixture
def simple_embeddings():
    """Create simple 2D embeddings for testing.

    5 points in 2D space arranged for predictable nearest neighbors:
    - Point 0: (0, 0) label 0
    - Point 1: (1, 0) label 0
    - Point 2: (0, 1) label 1
    - Point 3: (3, 3) label 1
    - Point 4: (10, 10) label 2
    """
    embeddings = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 3.0], [10.0, 10.0]],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    return embeddings, labels


@pytest.fixture
def engine_euclidean(simple_embeddings):
    """Euclidean distance retrieval engine."""
    embeddings, labels = simple_embeddings
    return RetrievalEngine(embeddings, labels, distance_metric="euclidean")


@pytest.fixture
def engine_cosine(simple_embeddings):
    """Cosine distance retrieval engine."""
    embeddings, labels = simple_embeddings
    return RetrievalEngine(embeddings, labels, distance_metric="cosine")


class TestRetrievalEngineInit:
    """Tests for RetrievalEngine initialization."""

    def test_valid_init_euclidean(self, simple_embeddings):
        embeddings, labels = simple_embeddings
        engine = RetrievalEngine(embeddings, labels, distance_metric="euclidean")
        assert engine.n_samples == 5
        assert engine.embedding_dim == 2
        assert engine.distance_metric == "euclidean"

    def test_valid_init_cosine(self, simple_embeddings):
        embeddings, labels = simple_embeddings
        engine = RetrievalEngine(embeddings, labels, distance_metric="cosine")
        assert engine.n_samples == 5
        assert engine.embedding_dim == 2
        assert engine.distance_metric == "cosine"

    def test_invalid_metric(self, simple_embeddings):
        embeddings, labels = simple_embeddings
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            RetrievalEngine(embeddings, labels, distance_metric="manhattan")

    def test_invalid_embeddings_shape(self):
        embeddings = np.ones((10,), dtype=np.float32)
        labels = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="must be 2D"):
            RetrievalEngine(embeddings, labels)

    def test_label_shape_mismatch(self):
        embeddings = np.ones((10, 4), dtype=np.float32)
        labels = np.zeros(5, dtype=np.int64)
        with pytest.raises(ValueError, match="does not match"):
            RetrievalEngine(embeddings, labels)


class TestQuerySorting:
    """Test that results are sorted by ascending distance."""

    def test_euclidean_sorted_ascending(self, engine_euclidean):
        # Query from origin - distances should be sorted ascending
        query = np.array([0.0, 0.0], dtype=np.float32)
        result = engine_euclidean.query(query, k=5)
        assert len(result.distances) == 5
        # Check ascending order
        for i in range(len(result.distances) - 1):
            assert result.distances[i] <= result.distances[i + 1]

    def test_cosine_sorted_ascending(self, engine_cosine):
        query = np.array([1.0, 1.0], dtype=np.float32)
        result = engine_cosine.query(query, k=4)
        for i in range(len(result.distances) - 1):
            assert result.distances[i] <= result.distances[i + 1]

    def test_nearest_neighbor_correctness(self, engine_euclidean):
        # Query at (0.5, 0) should be closest to (0,0) and (1,0)
        query = np.array([0.5, 0.0], dtype=np.float32)
        result = engine_euclidean.query(query, k=2)
        # Both (0,0) and (1,0) are distance 0.5 from query
        assert set(result.indices.tolist()) == {0, 1}
        np.testing.assert_allclose(result.distances, [0.5, 0.5], atol=1e-5)


class TestSelfExclusion:
    """Test self-exclusion behavior."""

    def test_exclude_index_not_in_results(self, engine_euclidean):
        # Query from point 0 with self-exclusion
        result = engine_euclidean.query(
            np.array([0.0, 0.0], dtype=np.float32),
            k=4,
            exclude_index=0,
        )
        assert 0 not in result.indices

    def test_query_by_index_excludes_self(self, engine_euclidean):
        result = engine_euclidean.query_by_index(0, k=4)
        assert 0 not in result.indices
        assert len(result.indices) == 4

    def test_query_by_index_returns_correct_neighbors(self, engine_euclidean):
        # Point 0 is at (0,0). Nearest should be point 1 (1,0) and point 2 (0,1)
        result = engine_euclidean.query_by_index(0, k=2)
        assert set(result.indices.tolist()) == {1, 2}


class TestClassFiltering:
    """Test class-filtered retrieval."""

    def test_filter_returns_only_specified_class(self, engine_euclidean):
        # Query from origin, only return class 1 neighbors
        query = np.array([0.0, 0.0], dtype=np.float32)
        result = engine_euclidean.query(query, k=5, filter_class=1)
        # Only points 2 and 3 have label 1
        assert len(result.indices) == 2
        assert all(label == 1 for label in result.labels)

    def test_filter_with_no_matching_class(self, engine_euclidean):
        # Filter for class 99 which doesn't exist
        query = np.array([0.0, 0.0], dtype=np.float32)
        result = engine_euclidean.query(query, k=5, filter_class=99)
        assert len(result.indices) == 0

    def test_filter_combined_with_exclusion(self, engine_euclidean):
        # Point 2 has label 1, query by index 2 with filter_class=1
        # Should get only point 3 (label 1), excluding point 2 itself
        result = engine_euclidean.query_by_index(2, k=5, filter_class=1)
        assert 2 not in result.indices
        assert all(label == 1 for label in result.labels)
        assert 3 in result.indices


class TestBatchQueries:
    """Test batch query functionality."""

    def test_batch_shapes(self, engine_euclidean):
        queries = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        result = engine_euclidean.query_batch(queries, k=3)
        assert result.indices.shape == (2, 3)
        assert result.distances.shape == (2, 3)
        assert result.labels.shape == (2, 3)

    def test_batch_sorted_per_query(self, engine_euclidean):
        queries = np.array(
            [[0.0, 0.0], [10.0, 10.0], [3.0, 3.0]], dtype=np.float32
        )
        result = engine_euclidean.query_batch(queries, k=4)
        # Each row should be sorted by ascending distance
        for i in range(3):
            for j in range(3):
                assert result.distances[i, j] <= result.distances[i, j + 1]

    def test_batch_with_exclude_indices(self, engine_euclidean):
        # Each query excludes itself
        queries = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        exclude = np.array([0, 1], dtype=np.int64)
        result = engine_euclidean.query_batch(queries, k=3, exclude_indices=exclude)
        assert 0 not in result.indices[0]
        assert 1 not in result.indices[1]

    def test_batch_consistency_with_single(self, engine_euclidean):
        """Batch results should match individual queries."""
        queries = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        batch_result = engine_euclidean.query_batch(queries, k=3)

        for i, q in enumerate(queries):
            single_result = engine_euclidean.query(q, k=3)
            np.testing.assert_array_equal(
                batch_result.indices[i], single_result.indices
            )
            np.testing.assert_allclose(
                batch_result.distances[i], single_result.distances, atol=1e-5
            )


class TestDistanceMetrics:
    """Test distance metric correctness."""

    def test_euclidean_known_distance(self, engine_euclidean):
        # Distance from (0,0) to (3,3) should be sqrt(18) = 4.2426...
        query = np.array([0.0, 0.0], dtype=np.float32)
        result = engine_euclidean.query(query, k=5)
        # Find the distance to point 3 (at index 3)
        idx_3_pos = np.where(result.indices == 3)[0][0]
        expected = np.sqrt(18.0)
        np.testing.assert_allclose(result.distances[idx_3_pos], expected, atol=1e-5)

    def test_cosine_identical_direction(self):
        """Vectors in the same direction should have cosine distance ~0."""
        embeddings = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        labels = np.array([0, 0, 1], dtype=np.int64)
        engine = RetrievalEngine(embeddings, labels, distance_metric="cosine")

        query = np.array([3.0, 0.0], dtype=np.float32)
        result = engine.query(query, k=3)
        # Both (1,0) and (2,0) have same direction as query -> distance ~0
        assert result.distances[0] < 0.01
        assert result.distances[1] < 0.01
        # (0,1) is perpendicular -> distance ~1
        assert result.distances[2] > 0.9

    def test_cosine_opposite_direction(self):
        """Vectors in opposite direction should have cosine distance ~2."""
        embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        engine = RetrievalEngine(embeddings, labels, distance_metric="cosine")

        query = np.array([1.0, 0.0], dtype=np.float32)
        result = engine.query(query, k=2)
        # Same direction -> 0
        np.testing.assert_allclose(result.distances[0], 0.0, atol=1e-5)
        # Opposite -> 2
        np.testing.assert_allclose(result.distances[1], 2.0, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_dimension_mismatch_raises(self, engine_euclidean):
        query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(DimensionMismatchError):
            engine_euclidean.query(query, k=3)

    def test_k_larger_than_database(self, engine_euclidean):
        query = np.array([0.0, 0.0], dtype=np.float32)
        result = engine_euclidean.query(query, k=100)
        # Should return all 5 points
        assert len(result.indices) == 5

    def test_k_equals_1(self, engine_euclidean):
        query = np.array([0.0, 0.0], dtype=np.float32)
        result = engine_euclidean.query(query, k=1)
        assert len(result.indices) == 1
        assert result.indices[0] == 0  # Nearest to origin is the origin itself

    def test_invalid_k(self, engine_euclidean):
        query = np.array([0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="k must be >= 1"):
            engine_euclidean.query(query, k=0)

    def test_query_by_index_out_of_range(self, engine_euclidean):
        with pytest.raises(IndexError):
            engine_euclidean.query_by_index(100)

    def test_query_by_index_negative(self, engine_euclidean):
        with pytest.raises(IndexError):
            engine_euclidean.query_by_index(-1)

    def test_result_labels_match_database(self, engine_euclidean):
        query = np.array([0.0, 0.0], dtype=np.float32)
        result = engine_euclidean.query(query, k=5)
        # Labels should match the database labels at the returned indices
        expected_labels = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        for idx, label in zip(result.indices, result.labels):
            assert label == expected_labels[idx]


class TestLargerDataset:
    """Tests with a larger dataset to verify performance characteristics."""

    @pytest.fixture
    def large_engine(self):
        """Create engine with 1000 random embeddings in 128 dimensions."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((1000, 128)).astype(np.float32)
        labels = rng.integers(0, 9, size=1000).astype(np.int64)
        return RetrievalEngine(embeddings, labels, distance_metric="euclidean")

    def test_results_sorted(self, large_engine):
        rng = np.random.default_rng(123)
        query = rng.standard_normal(128).astype(np.float32)
        result = large_engine.query(query, k=50)
        for i in range(len(result.distances) - 1):
            assert result.distances[i] <= result.distances[i + 1]

    def test_batch_query_large(self, large_engine):
        rng = np.random.default_rng(456)
        queries = rng.standard_normal((10, 128)).astype(np.float32)
        result = large_engine.query_batch(queries, k=20)
        assert result.indices.shape == (10, 20)
        assert result.distances.shape == (10, 20)
        # All rows sorted
        for i in range(10):
            for j in range(19):
                assert result.distances[i, j] <= result.distances[i, j + 1]
