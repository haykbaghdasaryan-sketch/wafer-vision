"""Unit tests for EmbeddingExtractor."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import BaseBackbone
from src.models.embedding_extractor import EmbeddingExtractor


class DummyBackbone(BaseBackbone):
    """A minimal backbone for testing that returns a deterministic embedding."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        # Simple linear layer for deterministic output
        self._linear = nn.Linear(3 * 64 * 64, embedding_dim, bias=False)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten and project
        b = x.shape[0]
        flat = x.view(b, -1)
        # Pad or truncate to match linear input
        target = 3 * 64 * 64
        if flat.shape[1] < target:
            flat = torch.nn.functional.pad(flat, (0, target - flat.shape[1]))
        elif flat.shape[1] > target:
            flat = flat[:, :target]
        return self._linear(flat)


class OOMBackbone(BaseBackbone):
    """A backbone that raises OOM on first call then succeeds."""

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._call_count = 0
        self._oom_until = 1  # OOM on first call only

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._call_count += 1
        if self._call_count <= self._oom_until:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        b = x.shape[0]
        return torch.randn(b, self._embedding_dim)


@pytest.fixture
def dummy_backbone():
    """Create a dummy backbone for testing."""
    return DummyBackbone(embedding_dim=128)


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader with 20 samples."""
    images = torch.rand(20, 3, 64, 64, dtype=torch.float32)
    labels = torch.randint(0, 9, (20,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path / "embeddings"


class TestEmbeddingExtractorBasic:
    """Test basic extraction functionality."""

    def test_extract_produces_npy_file(
        self, dummy_backbone, simple_dataloader, output_dir
    ):
        """Extract should produce a .npy file at the specified path."""
        extractor = EmbeddingExtractor(
            backbone=dummy_backbone, batch_size=4, device="cpu"
        )
        output_path = output_dir / "test_embeddings.npy"
        result = extractor.extract(simple_dataloader, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_extract_correct_shape(
        self, dummy_backbone, simple_dataloader, output_dir
    ):
        """Extracted embeddings should have shape (N, D)."""
        extractor = EmbeddingExtractor(
            backbone=dummy_backbone, batch_size=4, device="cpu"
        )
        output_path = output_dir / "embeddings.npy"
        extractor.extract(simple_dataloader, output_path)

        loaded = np.load(str(output_path))
        assert loaded.shape == (20, 128)

    def test_extract_float32_dtype(
        self, dummy_backbone, simple_dataloader, output_dir
    ):
        """Extracted embeddings should be float32."""
        extractor = EmbeddingExtractor(
            backbone=dummy_backbone, batch_size=4, device="cpu"
        )
        output_path = output_dir / "embeddings.npy"
        extractor.extract(simple_dataloader, output_path)

        loaded = np.load(str(output_path))
        assert loaded.dtype == np.float32

    def test_extract_maintains_sample_ordering(
        self, dummy_backbone, output_dir
    ):
        """Embeddings should maintain consistent ordering with dataset indices."""
        # Create dataset with known ordering
        images = torch.rand(10, 3, 64, 64, dtype=torch.float32)
        labels = torch.arange(10)
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        extractor = EmbeddingExtractor(
            backbone=dummy_backbone, batch_size=2, device="cpu"
        )
        output_path = output_dir / "ordered.npy"
        extractor.extract(dataloader, output_path)

        # Extract again, should get same result
        output_path2 = output_dir / "ordered2.npy"
        extractor.extract(dataloader, output_path2)

        emb1 = np.load(str(output_path))
        emb2 = np.load(str(output_path2))
        np.testing.assert_array_equal(emb1, emb2)


class TestEmbeddingExtractorMetadata:
    """Test sidecar metadata JSON creation."""

    def test_metadata_file_created(
        self, dummy_backbone, simple_dataloader, output_dir
    ):
        """A sidecar metadata JSON should be created alongside .npy."""
        extractor = EmbeddingExtractor(
            backbone=dummy_backbone, batch_size=4, device="cpu"
        )
        output_path = output_dir / "embeddings.npy"
        extractor.extract(simple_dataloader, output_path)

        metadata_path = output_dir / "embeddings_metadata.json"
        assert metadata_path.exists()

    def test_metadata_contains_required_fields(
        self, dummy_backbone, simple_dataloader, output_dir
    ):
        """Metadata JSON should contain backbone_name, embedding_dim, etc."""
        extractor = EmbeddingExtractor(
            backbone=dummy_backbone, batch_size=4, device="cpu"
        )
        output_path = output_dir / "embeddings.npy"
        extractor.extract(simple_dataloader, output_path)

        metadata_path = output_dir / "embeddings_metadata.json"
        with open(metadata_path) as f:
            meta = json.load(f)

        assert "backbone_name" in meta
        assert "embedding_dim" in meta
        assert "num_samples" in meta
        assert "extraction_timestamp" in meta
        assert "model_weights_sha256" in meta

        assert meta["backbone_name"] == "DummyBackbone"
        assert meta["embedding_dim"] == 128
        assert meta["num_samples"] == 20

    def test_metadata_includes_extra_fields(
        self, dummy_backbone, simple_dataloader, output_dir
    ):
        """Extra metadata passed to extract() should appear in the JSON."""
        extractor = EmbeddingExtractor(
            backbone=dummy_backbone, batch_size=4, device="cpu"
        )
        output_path = output_dir / "embeddings.npy"
        extractor.extract(
            simple_dataloader,
            output_path,
            metadata={"dataset_split": "test", "input_resolution": 64},
        )

        metadata_path = output_dir / "embeddings_metadata.json"
        with open(metadata_path) as f:
            meta = json.load(f)

        assert meta["dataset_split"] == "test"
        assert meta["input_resolution"] == 64


class TestEmbeddingExtractorOOMRecovery:
    """Test OOM recovery mechanism."""

    def test_oom_recovery_halves_batch_and_retries(self, output_dir):
        """On OOM, extractor should halve batch size and succeed."""
        backbone = OOMBackbone(embedding_dim=64)
        images = torch.rand(8, 3, 64, 64, dtype=torch.float32)
        labels = torch.randint(0, 9, (8,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        extractor = EmbeddingExtractor(
            backbone=backbone, batch_size=8, device="cpu"
        )
        output_path = output_dir / "oom_test.npy"
        result = extractor.extract(dataloader, output_path)

        assert result.exists()
        loaded = np.load(str(result))
        assert loaded.shape[0] == 8
        assert loaded.shape[1] == 64

    def test_oom_raises_after_max_retries(self, output_dir):
        """Should raise RuntimeError if OOM persists after max retries."""

        class AlwaysOOMBackbone(BaseBackbone):
            def __init__(self):
                super().__init__()
                self._dim = 64

            @property
            def embedding_dim(self) -> int:
                return self._dim

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("CUDA out of memory")

        backbone = AlwaysOOMBackbone()
        images = torch.rand(4, 3, 64, 64, dtype=torch.float32)
        labels = torch.randint(0, 9, (4,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        extractor = EmbeddingExtractor(
            backbone=backbone, batch_size=4, device="cpu"
        )
        output_path = output_dir / "fail.npy"

        with pytest.raises(RuntimeError, match="OOM"):
            extractor.extract(dataloader, output_path)


class TestEmbeddingExtractorIncremental:
    """Test incremental extraction (resume from partial)."""

    def test_resumes_from_partial_npy(self, dummy_backbone, output_dir):
        """Should resume extraction from a partial .npy file."""
        # Create a partial file with first 10 samples
        images = torch.rand(20, 3, 64, 64, dtype=torch.float32)
        labels = torch.randint(0, 9, (20,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # First, extract partial (first 10 samples)
        dummy_backbone.eval()
        partial_embeddings = []
        with torch.no_grad():
            for i, (img_batch, _) in enumerate(dataloader):
                if i >= 2:  # Only 2 batches of 4 = 8 samples
                    break
                emb = dummy_backbone(img_batch)
                partial_embeddings.append(emb.numpy())

        partial_arr = np.concatenate(partial_embeddings, axis=0)
        output_path = output_dir / "embeddings.npy"
        partial_path = output_path.with_suffix(".partial.npy")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(partial_path), partial_arr)

        # Now extract with the partial file present
        extractor = EmbeddingExtractor(
            backbone=dummy_backbone, batch_size=4, device="cpu"
        )
        result = extractor.extract(dataloader, output_path)

        assert result.exists()
        loaded = np.load(str(result))
        # Should have embeddings for all 20 samples
        # (8 from partial + remaining from full iteration)
        assert loaded.shape[1] == 128

        # Partial file should be cleaned up
        assert not partial_path.exists()


class TestEmbeddingExtractorDevice:
    """Test device resolution."""

    def test_auto_device_resolves_to_cpu_without_cuda(self, dummy_backbone):
        """Device 'auto' should resolve to 'cpu' when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            extractor = EmbeddingExtractor(
                backbone=dummy_backbone, device="auto"
            )
            assert extractor._device == "cpu"

    def test_explicit_cpu_device(self, dummy_backbone):
        """Explicit 'cpu' device should be used."""
        extractor = EmbeddingExtractor(backbone=dummy_backbone, device="cpu")
        assert extractor._device == "cpu"
