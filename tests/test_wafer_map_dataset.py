"""Unit tests for WaferMapDataset.

Tests cover: loading, __len__, __getitem__, transform application,
from_split factory, and error handling.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.data.dataset import WaferMapDataset
from src.exceptions import DataFileNotFoundError


@pytest.fixture
def sample_split_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with synthetic preprocessed splits."""
    n_samples = 20
    h, w = 64, 64
    n_classes = 4

    data = torch.rand(n_samples, 3, h, w, dtype=torch.float32)
    labels = torch.randint(0, n_classes, (n_samples,), dtype=torch.int64)

    torch.save(data, tmp_path / "train_data.pt")
    torch.save(labels, tmp_path / "train_labels.pt")

    # Also create val split
    val_data = torch.rand(5, 3, h, w, dtype=torch.float32)
    val_labels = torch.randint(0, n_classes, (5,), dtype=torch.int64)
    torch.save(val_data, tmp_path / "val_data.pt")
    torch.save(val_labels, tmp_path / "val_labels.pt")

    return tmp_path


class TestWaferMapDatasetInit:
    """Tests for WaferMapDataset initialization."""

    def test_load_valid_files(self, sample_split_dir: Path):
        """Should load .pt files successfully."""
        ds = WaferMapDataset(
            data_path=sample_split_dir / "train_data.pt",
            labels_path=sample_split_dir / "train_labels.pt",
        )
        assert len(ds) == 20

    def test_raises_on_missing_data_file(self, tmp_path: Path):
        """Should raise DataFileNotFoundError if data file is missing."""
        labels = torch.randint(0, 4, (10,), dtype=torch.int64)
        torch.save(labels, tmp_path / "labels.pt")

        with pytest.raises(DataFileNotFoundError):
            WaferMapDataset(
                data_path=tmp_path / "nonexistent_data.pt",
                labels_path=tmp_path / "labels.pt",
            )

    def test_raises_on_missing_labels_file(self, tmp_path: Path):
        """Should raise DataFileNotFoundError if labels file is missing."""
        data = torch.rand(10, 3, 64, 64, dtype=torch.float32)
        torch.save(data, tmp_path / "data.pt")

        with pytest.raises(DataFileNotFoundError):
            WaferMapDataset(
                data_path=tmp_path / "data.pt",
                labels_path=tmp_path / "nonexistent_labels.pt",
            )

    def test_raises_on_mismatched_counts(self, tmp_path: Path):
        """Should raise ValueError if data and labels have different sample counts."""
        data = torch.rand(10, 3, 64, 64, dtype=torch.float32)
        labels = torch.randint(0, 4, (5,), dtype=torch.int64)  # mismatch
        torch.save(data, tmp_path / "data.pt")
        torch.save(labels, tmp_path / "labels.pt")

        with pytest.raises(ValueError, match="mismatched sample counts"):
            WaferMapDataset(
                data_path=tmp_path / "data.pt",
                labels_path=tmp_path / "labels.pt",
            )


class TestWaferMapDatasetGetItem:
    """Tests for __getitem__ and __len__."""

    def test_len_returns_correct_count(self, sample_split_dir: Path):
        """__len__ should return the number of samples."""
        ds = WaferMapDataset(
            data_path=sample_split_dir / "train_data.pt",
            labels_path=sample_split_dir / "train_labels.pt",
        )
        assert len(ds) == 20

    def test_getitem_returns_tuple(self, sample_split_dir: Path):
        """__getitem__ should return (tensor, label) tuple."""
        ds = WaferMapDataset(
            data_path=sample_split_dir / "train_data.pt",
            labels_path=sample_split_dir / "train_labels.pt",
        )
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_tensor_shape(self, sample_split_dir: Path):
        """Returned tensor should have shape (3, H, W)."""
        ds = WaferMapDataset(
            data_path=sample_split_dir / "train_data.pt",
            labels_path=sample_split_dir / "train_labels.pt",
        )
        tensor, label = ds[0]
        assert tensor.shape == (3, 64, 64)

    def test_getitem_label_is_scalar(self, sample_split_dir: Path):
        """Returned label should be a scalar tensor."""
        ds = WaferMapDataset(
            data_path=sample_split_dir / "train_data.pt",
            labels_path=sample_split_dir / "train_labels.pt",
        )
        tensor, label = ds[0]
        assert label.dim() == 0  # scalar
        assert label.dtype == torch.int64

    def test_getitem_no_transform_returns_raw(self, sample_split_dir: Path):
        """Without transform, __getitem__ should return raw tensor unchanged."""
        ds = WaferMapDataset(
            data_path=sample_split_dir / "train_data.pt",
            labels_path=sample_split_dir / "train_labels.pt",
            transform=None,
        )
        tensor, _ = ds[0]
        expected = ds.data[0]
        assert torch.equal(tensor, expected)


class TestWaferMapDatasetTransform:
    """Tests for lazy transform application."""

    def test_transform_is_applied(self, sample_split_dir: Path):
        """Transform should be applied at __getitem__ time."""
        # Simple transform that zeros out the tensor
        def zero_transform(t: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(t)

        ds = WaferMapDataset(
            data_path=sample_split_dir / "train_data.pt",
            labels_path=sample_split_dir / "train_labels.pt",
            transform=zero_transform,
        )
        tensor, _ = ds[0]
        assert torch.all(tensor == 0.0)

    def test_transform_does_not_mutate_data(self, sample_split_dir: Path):
        """Transform should not mutate the underlying stored data."""
        def add_one(t: torch.Tensor) -> torch.Tensor:
            return t + 1.0

        ds = WaferMapDataset(
            data_path=sample_split_dir / "train_data.pt",
            labels_path=sample_split_dir / "train_labels.pt",
            transform=add_one,
        )
        original = ds.data[0].clone()
        _ = ds[0]
        # The stored data should be unchanged
        assert torch.equal(ds.data[0], original)

    def test_transform_applied_lazily_gives_different_results(self, sample_split_dir: Path):
        """A stochastic transform should produce different results on repeated access."""
        def random_noise(t: torch.Tensor) -> torch.Tensor:
            return t + torch.randn_like(t) * 0.1

        ds = WaferMapDataset(
            data_path=sample_split_dir / "train_data.pt",
            labels_path=sample_split_dir / "train_labels.pt",
            transform=random_noise,
        )
        t1, _ = ds[0]
        t2, _ = ds[0]
        # Stochastic transform should (almost certainly) produce different results
        assert not torch.equal(t1, t2)


class TestWaferMapDatasetFromSplit:
    """Tests for the from_split class method."""

    def test_from_split_train(self, sample_split_dir: Path):
        """from_split should load the train split correctly."""
        ds = WaferMapDataset.from_split(sample_split_dir, split_name="train")
        assert len(ds) == 20

    def test_from_split_val(self, sample_split_dir: Path):
        """from_split should load the val split correctly."""
        ds = WaferMapDataset.from_split(sample_split_dir, split_name="val")
        assert len(ds) == 5

    def test_from_split_with_transform(self, sample_split_dir: Path):
        """from_split should pass transform to the dataset."""
        def negate(t: torch.Tensor) -> torch.Tensor:
            return -t

        ds = WaferMapDataset.from_split(
            sample_split_dir, split_name="train", transform=negate
        )
        tensor, _ = ds[0]
        raw = ds.data[0]
        assert torch.allclose(tensor, -raw)

    def test_from_split_missing_files(self, tmp_path: Path):
        """from_split should raise DataFileNotFoundError for missing split."""
        with pytest.raises(DataFileNotFoundError):
            WaferMapDataset.from_split(tmp_path, split_name="nonexistent")

    def test_from_split_default_is_train(self, sample_split_dir: Path):
        """from_split with no split_name should default to 'train'."""
        ds = WaferMapDataset.from_split(sample_split_dir)
        assert len(ds) == 20

    def test_from_split_string_path(self, sample_split_dir: Path):
        """from_split should accept string paths."""
        ds = WaferMapDataset.from_split(str(sample_split_dir), split_name="train")
        assert len(ds) == 20
