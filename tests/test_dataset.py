"""Unit tests: data loading.

Tests for the DatasetAdapter ABC, WM811KLoader, and validation utilities.
"""

import io
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.adapter import DatasetAdapter
from src.data.loader import (
    KNOWN_CLASSES,
    WM811KLoader,
    WaferRecord,
)
from src.data.validation import (
    MIN_SPATIAL_DIM,
    validate_pixel_values,
    validate_spatial_dimensions,
    validate_wafer_map,
)
from src.exceptions import DataCorruptionError, DataFileNotFoundError


# --- Validation tests ---


class TestValidatePixelValues:
    """Tests for validate_pixel_values function."""

    def test_valid_values_all_zero(self):
        wafer = np.zeros((10, 10), dtype=np.uint8)
        assert validate_pixel_values(wafer, 0) is True

    def test_valid_values_mixed(self):
        wafer = np.array([[0, 1, 2], [2, 1, 0], [1, 2, 0]], dtype=np.uint8)
        assert validate_pixel_values(wafer, 0) is True

    def test_invalid_values(self):
        wafer = np.array([[0, 1, 3], [2, 1, 0]], dtype=np.uint8)
        assert validate_pixel_values(wafer, 0) is False

    def test_negative_values(self):
        wafer = np.array([[0, -1, 2], [2, 1, 0]], dtype=np.int8)
        assert validate_pixel_values(wafer, 0) is False


class TestValidateSpatialDimensions:
    """Tests for validate_spatial_dimensions function."""

    def test_valid_dimensions(self):
        wafer = np.zeros((10, 10), dtype=np.uint8)
        assert validate_spatial_dimensions(wafer, 0) is True

    def test_exact_minimum(self):
        wafer = np.zeros((5, 5), dtype=np.uint8)
        assert validate_spatial_dimensions(wafer, 0) is True

    def test_below_minimum_height(self):
        wafer = np.zeros((4, 10), dtype=np.uint8)
        assert validate_spatial_dimensions(wafer, 0) is False

    def test_below_minimum_width(self):
        wafer = np.zeros((10, 4), dtype=np.uint8)
        assert validate_spatial_dimensions(wafer, 0) is False

    def test_both_below_minimum(self):
        wafer = np.zeros((3, 3), dtype=np.uint8)
        assert validate_spatial_dimensions(wafer, 0) is False


class TestValidateWaferMap:
    """Tests for the combined validate_wafer_map function."""

    def test_valid_wafer_map(self):
        wafer = np.array(
            [[0, 1, 2, 0, 1], [1, 0, 2, 1, 0], [2, 1, 0, 2, 1],
             [0, 1, 2, 0, 1], [2, 0, 1, 2, 0]],
            dtype=np.uint8,
        )
        is_valid, reason = validate_wafer_map(wafer, 0)
        assert is_valid is True
        assert reason == ""

    def test_none_wafer_map(self):
        is_valid, reason = validate_wafer_map(None, 0)
        assert is_valid is False
        assert "None" in reason

    def test_non_ndarray(self):
        is_valid, reason = validate_wafer_map([[1, 2], [3, 4]], 0)
        assert is_valid is False
        assert "ndarray" in reason

    def test_wrong_dimensions(self):
        wafer = np.zeros((5, 5, 3), dtype=np.uint8)
        is_valid, reason = validate_wafer_map(wafer, 0)
        assert is_valid is False
        assert "dimensions" in reason

    def test_too_small(self):
        wafer = np.zeros((3, 3), dtype=np.uint8)
        is_valid, reason = validate_wafer_map(wafer, 0)
        assert is_valid is False
        assert "spatial" in reason


# --- WM811KLoader tests ---


def _create_synthetic_dataset(
    n_records: int = 20,
    include_unlabeled: bool = True,
    include_unknown: bool = True,
    include_small: bool = True,
    include_invalid_pixels: bool = True,
) -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking WM-811K structure."""
    records = []
    labels = KNOWN_CLASSES.copy()  # 9 known classes

    for i in range(n_records):
        # Create a valid wafer map
        height = np.random.randint(13, 52)
        width = np.random.randint(13, 52)
        wafer_map = np.random.choice([0, 1, 2], size=(height, width)).astype(np.uint8)

        # Assign label based on index
        if include_unlabeled and i == n_records - 1:
            label = []  # empty list = unlabeled
        elif include_unknown and i == n_records - 2:
            label = ["SomeUnknownClass"]
        elif include_small and i == n_records - 3:
            wafer_map = np.zeros((3, 3), dtype=np.uint8)  # too small
            label = [labels[i % len(labels)]]
        elif include_invalid_pixels and i == n_records - 4:
            wafer_map[0, 0] = 5  # invalid pixel
            label = [labels[i % len(labels)]]
        else:
            label = [labels[i % len(labels)]]

        records.append({
            "waferMap": wafer_map,
            "failureType": label,
            "lotName": f"lot_{i // 3}",
            "waferIndex": i,
        })

    return pd.DataFrame(records)


def _save_synthetic_pickle(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as a pickle file (using standard pickle for tests)."""
    with open(path, "wb") as f:
        pickle.dump(df, f)


class TestWM811KLoader:
    """Tests for the WM811KLoader class."""

    def test_is_dataset_adapter(self):
        """WM811KLoader should be a subclass of DatasetAdapter."""
        loader = WM811KLoader()
        assert isinstance(loader, DatasetAdapter)

    def test_load_file_not_found(self, tmp_path):
        """Should raise DataFileNotFoundError for missing file."""
        loader = WM811KLoader()
        with pytest.raises(DataFileNotFoundError):
            loader.load(tmp_path / "nonexistent.pkl")

    @patch("src.data.loader.get_memory_info")
    def test_load_corrupted_file(self, mock_mem, tmp_path):
        """Should raise DataCorruptionError for corrupted pickle."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        bad_file = tmp_path / "bad.pkl"
        bad_file.write_bytes(b"not a pickle file at all")
        loader = WM811KLoader()
        with pytest.raises(DataCorruptionError):
            loader.load(bad_file)

    @patch("src.data.loader.get_memory_info")
    def test_load_insufficient_memory(self, mock_mem, tmp_path):
        """Should raise MemoryError when free RAM < 4 GB."""
        mock_mem.return_value = {
            "ram_used_mb": 12000.0,
            "ram_available_mb": 2000.0,  # < 4096 MB
            "ram_total_mb": 16000.0,
        }
        # Create a valid file so it doesn't fail on file check
        df = _create_synthetic_dataset(5, False, False, False, False)
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        with pytest.raises(MemoryError):
            loader.load(pkl_path)

    @patch("src.data.loader.get_memory_info")
    def test_load_success(self, mock_mem, tmp_path):
        """Should successfully load a valid synthetic dataset."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        df = _create_synthetic_dataset(10, False, False, False, False)
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        records = loader.load(pkl_path)

        assert len(records) == 10
        assert all(isinstance(r, WaferRecord) for r in records)

    @patch("src.data.loader.get_memory_info")
    def test_load_caching(self, mock_mem, tmp_path):
        """Calling load twice with same path should return cached data."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        df = _create_synthetic_dataset(5, False, False, False, False)
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        records1 = loader.load(pkl_path)
        records2 = loader.load(pkl_path)

        assert records1 is records2  # same object reference = cached

    @patch("src.data.loader.get_memory_info")
    def test_label_classification_known(self, mock_mem, tmp_path):
        """Records with known class labels should be classified correctly."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        # Create dataset with only known labels
        df = _create_synthetic_dataset(9, False, False, False, False)
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        records = loader.load(pkl_path)

        for record in records:
            assert record.label in KNOWN_CLASSES

    @patch("src.data.loader.get_memory_info")
    def test_label_classification_unlabeled(self, mock_mem, tmp_path):
        """Records with empty/None labels should be classified as unlabeled."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        # Create a record with empty failureType list
        wafer = np.random.choice([0, 1, 2], size=(10, 10)).astype(np.uint8)
        df = pd.DataFrame([{
            "waferMap": wafer,
            "failureType": [],
            "lotName": "lot_0",
            "waferIndex": 0,
        }])
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        records = loader.load(pkl_path)

        assert len(records) == 1
        assert records[0].label == "unlabeled"

    @patch("src.data.loader.get_memory_info")
    def test_label_classification_nan(self, mock_mem, tmp_path):
        """Records with NaN label should be classified as unlabeled."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        wafer = np.random.choice([0, 1, 2], size=(10, 10)).astype(np.uint8)
        df = pd.DataFrame([{
            "waferMap": wafer,
            "failureType": float("nan"),
            "lotName": "lot_0",
            "waferIndex": 0,
        }])
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        records = loader.load(pkl_path)

        assert len(records) == 1
        assert records[0].label == "unlabeled"

    @patch("src.data.loader.get_memory_info")
    def test_label_classification_unknown(self, mock_mem, tmp_path):
        """Records with unrecognized labels should be classified as unknown."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        wafer = np.random.choice([0, 1, 2], size=(10, 10)).astype(np.uint8)
        df = pd.DataFrame([{
            "waferMap": wafer,
            "failureType": ["NewUnknownDefect"],
            "lotName": "lot_0",
            "waferIndex": 0,
        }])
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        records = loader.load(pkl_path)

        assert len(records) == 1
        assert records[0].label == "unknown"

    @patch("src.data.loader.get_memory_info")
    def test_small_wafer_maps_discarded(self, mock_mem, tmp_path):
        """Wafer maps smaller than 5×5 should be discarded."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        small_wafer = np.zeros((3, 3), dtype=np.uint8)
        valid_wafer = np.random.choice([0, 1, 2], size=(10, 10)).astype(np.uint8)
        df = pd.DataFrame([
            {"waferMap": small_wafer, "failureType": ["Center"], "lotName": "lot_0", "waferIndex": 0},
            {"waferMap": valid_wafer, "failureType": ["Center"], "lotName": "lot_0", "waferIndex": 1},
        ])
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        records = loader.load(pkl_path)

        assert len(records) == 1
        assert records[0].wafer_map.shape[0] >= 5

    @patch("src.data.loader.get_memory_info")
    def test_get_summary(self, mock_mem, tmp_path):
        """get_summary should return correct statistics."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        df = _create_synthetic_dataset(20, True, True, True, False)
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        loader.load(pkl_path)
        summary = loader.get_summary()

        assert "total_count" in summary
        assert "labeled_count" in summary
        assert "unlabeled_count" in summary
        assert "per_class_counts" in summary
        assert "per_class_percentages" in summary
        assert "min_spatial_dims" in summary
        assert "max_spatial_dims" in summary
        assert "mean_spatial_dims" in summary
        assert "lot_count" in summary
        assert summary["total_count"] > 0

    def test_get_summary_before_load(self):
        """get_summary before load should raise RuntimeError."""
        loader = WM811KLoader()
        with pytest.raises(RuntimeError, match="No dataset loaded"):
            loader.get_summary()

    @patch("src.data.loader.get_memory_info")
    def test_record_fields(self, mock_mem, tmp_path):
        """Each WaferRecord should have all required fields."""
        mock_mem.return_value = {
            "ram_used_mb": 4000.0,
            "ram_available_mb": 8000.0,
            "ram_total_mb": 16000.0,
        }
        df = _create_synthetic_dataset(5, False, False, False, False)
        pkl_path = tmp_path / "test.pkl"
        _save_synthetic_pickle(df, pkl_path)

        loader = WM811KLoader()
        records = loader.load(pkl_path)

        for record in records:
            assert isinstance(record.wafer_map, np.ndarray)
            assert record.wafer_map.ndim == 2
            assert isinstance(record.label, str)
            assert record.lot is not None
            assert isinstance(record.wafer_index, int)
            assert isinstance(record.original_index, int)
