"""Shared fixtures and synthetic data generators for testing."""

import pytest
import numpy as np
import torch


@pytest.fixture
def synthetic_wafer_map():
    """Generate a random synthetic wafer map (H x W) with values in {0, 1, 2}."""
    def _make(height: int = 26, width: int = 26) -> np.ndarray:
        return np.random.choice([0, 1, 2], size=(height, width)).astype(np.uint8)
    return _make


@pytest.fixture
def synthetic_batch():
    """Generate a synthetic batch of preprocessed tensors (B, 3, H, W)."""
    def _make(batch_size: int = 4, height: int = 64, width: int = 64) -> torch.Tensor:
        return torch.rand(batch_size, 3, height, width, dtype=torch.float32)
    return _make


@pytest.fixture
def synthetic_embeddings():
    """Generate synthetic embedding matrix (N, D) with random unit vectors."""
    def _make(n_samples: int = 100, dim: int = 2048) -> np.ndarray:
        embeddings = np.random.randn(n_samples, dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    return _make


@pytest.fixture
def synthetic_labels():
    """Generate synthetic class labels for 9 wafer defect classes."""
    def _make(n_samples: int = 100, n_classes: int = 9) -> np.ndarray:
        return np.random.randint(0, n_classes, size=n_samples)
    return _make
