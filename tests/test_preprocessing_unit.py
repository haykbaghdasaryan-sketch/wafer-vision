"""Unit tests for the WaferPreprocessor pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.loader import WaferRecord
from src.data.preprocessing import WaferPreprocessor


class TestPreprocess:
    """Tests for the preprocess() method."""

    def test_output_shape(self):
        """Output tensor has shape (3, target_size, target_size)."""
        preprocessor = WaferPreprocessor(target_size=64)
        wafer_map = np.array(
            [[0, 1, 2, 0, 1],
             [1, 2, 0, 1, 2],
             [2, 0, 1, 2, 0],
             [0, 1, 2, 0, 1],
             [1, 2, 0, 1, 2]],
            dtype=np.int32,
        )
        result = preprocessor.preprocess(wafer_map)
        assert result.shape == (3, 64, 64)

    def test_output_dtype(self):
        """Output tensor has dtype float32."""
        preprocessor = WaferPreprocessor(target_size=32)
        wafer_map = np.ones((10, 10), dtype=np.int32) * 2
        result = preprocessor.preprocess(wafer_map)
        assert result.dtype == torch.float32

    def test_normalization_values(self):
        """After preprocessing, max value is 1.0 (from pixel value 2 / 2)."""
        preprocessor = WaferPreprocessor(target_size=32)
        # Uniform map of 2s -> all pixels become 1.0 after /2
        wafer_map = np.ones((10, 10), dtype=np.int32) * 2
        result = preprocessor.preprocess(wafer_map)
        assert torch.allclose(result, torch.ones_like(result))

    def test_normalization_zeros(self):
        """Pixel value 0 becomes 0.0 after normalization."""
        preprocessor = WaferPreprocessor(target_size=32)
        wafer_map = np.zeros((10, 10), dtype=np.int32)
        result = preprocessor.preprocess(wafer_map)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_normalization_ones(self):
        """Pixel value 1 becomes 0.5 after normalization."""
        preprocessor = WaferPreprocessor(target_size=32)
        wafer_map = np.ones((10, 10), dtype=np.int32)
        result = preprocessor.preprocess(wafer_map)
        expected = torch.full_like(result, 0.5)
        assert torch.allclose(result, expected, atol=0.05)

    def test_three_channels_identical(self):
        """All 3 channels should contain the same data."""
        preprocessor = WaferPreprocessor(target_size=32)
        wafer_map = np.random.choice([0, 1, 2], size=(10, 10))
        result = preprocessor.preprocess(wafer_map)
        assert torch.allclose(result[0], result[1])
        assert torch.allclose(result[1], result[2])

    def test_different_target_sizes(self):
        """Supports various target sizes in valid range."""
        wafer_map = np.random.choice([0, 1, 2], size=(15, 15))
        for size in [32, 64, 128, 224]:
            preprocessor = WaferPreprocessor(target_size=size)
            result = preprocessor.preprocess(wafer_map)
            assert result.shape == (3, size, size)


class TestTargetSizeValidation:
    """Tests for target size validation."""

    def test_too_small(self):
        """Raises ValueError for target_size < 32."""
        with pytest.raises(ValueError, match="Target resolution must be in range"):
            WaferPreprocessor(target_size=10)

    def test_too_large(self):
        """Raises ValueError for target_size > 224."""
        with pytest.raises(ValueError, match="Target resolution must be in range"):
            WaferPreprocessor(target_size=300)

    def test_boundary_min(self):
        """target_size=32 is valid."""
        preprocessor = WaferPreprocessor(target_size=32)
        assert preprocessor.target_size == 32

    def test_boundary_max(self):
        """target_size=224 is valid."""
        preprocessor = WaferPreprocessor(target_size=224)
        assert preprocessor.target_size == 224


class TestCreateSplits:
    """Tests for the create_splits() method."""

    @pytest.fixture
    def sample_records(self):
        """Create sample WaferRecord instances for testing."""
        classes = [
            "Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Near-full", "Random", "Scratch", "none",
        ]
        records = []
        np.random.seed(42)
        for i in range(900):
            cls = classes[i % 9]
            h, w = np.random.randint(10, 30), np.random.randint(10, 30)
            wafer = np.random.choice([0, 1, 2], size=(h, w)).astype(np.int32)
            records.append(
                WaferRecord(
                    wafer_map=wafer,
                    label=cls,
                    lot=f"lot_{i // 10}",
                    wafer_index=i,
                    original_index=i,
                )
            )
        return records

    def test_split_counts(self, sample_records):
        """Total samples across all splits equals input count."""
        preprocessor = WaferPreprocessor(target_size=32)
        splits = preprocessor.create_splits(sample_records, seed=42)
        total = (
            splits["train_data"].shape[0]
            + splits["val_data"].shape[0]
            + splits["test_data"].shape[0]
        )
        assert total == 900

    def test_split_ratios(self, sample_records):
        """Split proportions are approximately 70/15/15."""
        preprocessor = WaferPreprocessor(target_size=32)
        splits = preprocessor.create_splits(sample_records, seed=42)
        n_train = splits["train_data"].shape[0]
        n_val = splits["val_data"].shape[0]
        n_test = splits["test_data"].shape[0]
        total = n_train + n_val + n_test

        assert abs(n_train / total - 0.7) < 0.02
        assert abs(n_val / total - 0.15) < 0.02
        assert abs(n_test / total - 0.15) < 0.02

    def test_label_dtype(self, sample_records):
        """Labels are int64 tensors."""
        preprocessor = WaferPreprocessor(target_size=32)
        splits = preprocessor.create_splits(sample_records, seed=42)
        assert splits["train_labels"].dtype == torch.int64
        assert splits["val_labels"].dtype == torch.int64
        assert splits["test_labels"].dtype == torch.int64

    def test_data_dtype(self, sample_records):
        """Data tensors are float32."""
        preprocessor = WaferPreprocessor(target_size=32)
        splits = preprocessor.create_splits(sample_records, seed=42)
        assert splits["train_data"].dtype == torch.float32
        assert splits["val_data"].dtype == torch.float32
        assert splits["test_data"].dtype == torch.float32

    def test_label_to_index_mapping(self, sample_records):
        """label_to_index maps all classes to consecutive indices."""
        preprocessor = WaferPreprocessor(target_size=32)
        splits = preprocessor.create_splits(sample_records, seed=42)
        label_map = splits["label_to_index"]
        assert len(label_map) == 9
        assert set(label_map.values()) == set(range(9))

    def test_deterministic_with_seed(self, sample_records):
        """Same seed produces identical splits."""
        preprocessor = WaferPreprocessor(target_size=32)
        splits1 = preprocessor.create_splits(sample_records, seed=42)
        splits2 = preprocessor.create_splits(sample_records, seed=42)
        assert torch.equal(splits1["train_labels"], splits2["train_labels"])
        assert torch.equal(splits1["val_labels"], splits2["val_labels"])
        assert torch.equal(splits1["test_labels"], splits2["test_labels"])

    def test_filters_unlabeled(self):
        """Unlabeled records are excluded from splits."""
        records = []
        np.random.seed(0)
        for i in range(20):
            wafer = np.random.choice([0, 1, 2], size=(10, 10)).astype(np.int32)
            label = "Center" if i < 15 else "unlabeled"
            records.append(
                WaferRecord(
                    wafer_map=wafer,
                    label=label,
                    lot="lot_0",
                    wafer_index=i,
                    original_index=i,
                )
            )

        preprocessor = WaferPreprocessor(target_size=32)
        splits = preprocessor.create_splits(records, seed=42)
        total = (
            splits["train_data"].shape[0]
            + splits["val_data"].shape[0]
            + splits["test_data"].shape[0]
        )
        # Only 15 labeled records should be included
        assert total == 15

    def test_filters_small_maps(self):
        """Records with spatial dims < 5x5 are excluded."""
        records = []
        np.random.seed(0)
        for i in range(20):
            size = 10 if i < 15 else 3  # Last 5 are too small
            wafer = np.random.choice([0, 1, 2], size=(size, size)).astype(np.int32)
            records.append(
                WaferRecord(
                    wafer_map=wafer,
                    label="Center",
                    lot="lot_0",
                    wafer_index=i,
                    original_index=i,
                )
            )

        preprocessor = WaferPreprocessor(target_size=32)
        splits = preprocessor.create_splits(records, seed=42)
        total = (
            splits["train_data"].shape[0]
            + splits["val_data"].shape[0]
            + splits["test_data"].shape[0]
        )
        assert total == 15


class TestSaveSplits:
    """Tests for the save_splits() method."""

    @pytest.fixture
    def simple_splits(self):
        """Create a simple splits dictionary for testing save."""
        preprocessor = WaferPreprocessor(target_size=32)
        records = []
        np.random.seed(0)
        classes = ["Center", "Donut", "Edge-Loc"]
        for i in range(60):
            cls = classes[i % 3]
            wafer = np.random.choice([0, 1, 2], size=(10, 10)).astype(np.int32)
            records.append(
                WaferRecord(
                    wafer_map=wafer,
                    label=cls,
                    lot="lot_0",
                    wafer_index=i,
                    original_index=i,
                )
            )
        return preprocessor.create_splits(records, seed=42)

    def test_saves_all_files(self, simple_splits):
        """All expected files are created."""
        preprocessor = WaferPreprocessor(target_size=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            preprocessor.save_splits(output_dir, simple_splits)

            expected_files = [
                "train_data.pt",
                "train_labels.pt",
                "val_data.pt",
                "val_labels.pt",
                "test_data.pt",
                "test_labels.pt",
                "label_to_index.json",
                "manifest.json",
            ]
            for filename in expected_files:
                assert (output_dir / filename).exists(), f"Missing: {filename}"

    def test_roundtrip_tensors(self, simple_splits):
        """Saved tensors can be loaded back and match originals."""
        preprocessor = WaferPreprocessor(target_size=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            preprocessor.save_splits(output_dir, simple_splits)

            for split_name in ("train", "val", "test"):
                data = torch.load(
                    output_dir / f"{split_name}_data.pt", weights_only=True
                )
                labels = torch.load(
                    output_dir / f"{split_name}_labels.pt", weights_only=True
                )
                assert torch.equal(data, simple_splits[f"{split_name}_data"])
                assert torch.equal(labels, simple_splits[f"{split_name}_labels"])

    def test_label_to_index_json(self, simple_splits):
        """label_to_index.json contains correct mapping."""
        preprocessor = WaferPreprocessor(target_size=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            preprocessor.save_splits(output_dir, simple_splits)

            with open(output_dir / "label_to_index.json") as f:
                loaded_map = json.load(f)
            assert loaded_map == simple_splits["label_to_index"]

    def test_manifest_json_structure(self, simple_splits):
        """manifest.json has expected structure."""
        preprocessor = WaferPreprocessor(target_size=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            preprocessor.save_splits(output_dir, simple_splits)

            with open(output_dir / "manifest.json") as f:
                manifest = json.load(f)

            assert "target_size" in manifest
            assert manifest["target_size"] == 32
            assert "num_classes" in manifest
            assert "label_to_index" in manifest
            assert "splits" in manifest
            assert "checksums" in manifest
            for split_name in ("train", "val", "test"):
                assert split_name in manifest["splits"]
                split_info = manifest["splits"][split_name]
                assert "num_samples" in split_info
                assert "data_shape" in split_info
                assert "labels_shape" in split_info
                assert "per_class_counts" in split_info
