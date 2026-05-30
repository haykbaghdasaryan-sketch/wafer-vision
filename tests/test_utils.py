"""Unit tests for src/utils.py utility functions."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.exceptions import (
    AtomicWriteError,
    ConfigValidationError,
    InsufficientDiskError,
)
from src.utils import (
    atomic_write,
    check_disk_space,
    compute_md5,
    get_memory_info,
    get_worker_seed,
    set_all_seeds,
)


class TestCheckDiskSpace:
    """Tests for check_disk_space function."""

    def test_passes_when_sufficient_space(self, tmp_path):
        """Should not raise when disk has enough space."""
        # Disk should have more than 1 MB free
        check_disk_space(tmp_path, required_mb=1.0)

    def test_raises_when_insufficient_space(self, tmp_path):
        """Should raise InsufficientDiskError when requirement exceeds available."""
        with pytest.raises(InsufficientDiskError) as exc_info:
            # Request an unreasonably large amount of disk space
            check_disk_space(tmp_path, required_mb=1_000_000_000.0)
        assert exc_info.value.required_mb == 1_000_000_000.0
        assert exc_info.value.available_mb > 0

    def test_handles_nonexistent_path(self, tmp_path):
        """Should resolve to parent when path doesn't exist."""
        nonexistent = tmp_path / "does" / "not" / "exist"
        check_disk_space(nonexistent, required_mb=1.0)


class TestAtomicWrite:
    """Tests for atomic_write function."""

    def test_basic_write(self, tmp_path):
        """Should write content atomically to target path."""
        target = tmp_path / "output.txt"

        def write_fn(path):
            path.write_text("hello world")

        atomic_write(target, write_fn)
        assert target.read_text() == "hello world"

    def test_verify_fn_success(self, tmp_path):
        """Should succeed when verify function passes."""
        target = tmp_path / "output.txt"

        def write_fn(path):
            path.write_text("valid content")

        def verify_fn(path):
            content = path.read_text()
            assert "valid" in content

        atomic_write(target, write_fn, verify_fn)
        assert target.read_text() == "valid content"

    def test_verify_fn_failure_raises_error(self, tmp_path):
        """Should raise AtomicWriteError when verify fails."""
        target = tmp_path / "output.txt"

        def write_fn(path):
            path.write_text("bad content")

        def verify_fn(path):
            raise ValueError("Verification failed")

        with pytest.raises(AtomicWriteError) as exc_info:
            atomic_write(target, write_fn, verify_fn)

        assert "Verification failed" in exc_info.value.cause
        # Target should NOT exist since write failed
        assert not target.exists()

    def test_write_fn_failure_raises_error(self, tmp_path):
        """Should raise AtomicWriteError when write function fails."""
        target = tmp_path / "output.txt"

        def write_fn(path):
            raise IOError("Write failed")

        with pytest.raises(AtomicWriteError):
            atomic_write(target, write_fn)

        assert not target.exists()

    def test_no_partial_file_on_failure(self, tmp_path):
        """Should clean up temp file on failure."""
        target = tmp_path / "output.txt"

        def write_fn(path):
            path.write_text("partial data")
            raise RuntimeError("Simulated crash")

        with pytest.raises(AtomicWriteError):
            atomic_write(target, write_fn)

        # No temp files should remain
        remaining = list(tmp_path.glob(".*"))
        assert len(remaining) == 0

    def test_overwrites_existing_file(self, tmp_path):
        """Should overwrite existing file atomically."""
        target = tmp_path / "output.txt"
        target.write_text("old content")

        def write_fn(path):
            path.write_text("new content")

        atomic_write(target, write_fn)
        assert target.read_text() == "new content"


class TestComputeMd5:
    """Tests for compute_md5 function."""

    def test_known_hash(self, tmp_path):
        """Should compute correct MD5 for known content."""
        file = tmp_path / "test.txt"
        file.write_text("hello world")
        # MD5 of "hello world" is known
        result = compute_md5(file)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_file(self, tmp_path):
        """Should compute MD5 for empty file."""
        file = tmp_path / "empty.txt"
        file.write_bytes(b"")
        result = compute_md5(file)
        # MD5 of empty string is d41d8cd98f00b204e9800998ecf8427e
        assert result == "d41d8cd98f00b204e9800998ecf8427e"

    def test_deterministic(self, tmp_path):
        """Should return same hash for same content."""
        file = tmp_path / "test.bin"
        file.write_bytes(b"\x00\x01\x02\x03" * 1000)
        assert compute_md5(file) == compute_md5(file)

    def test_different_content_different_hash(self, tmp_path):
        """Different content should produce different hashes."""
        file1 = tmp_path / "a.txt"
        file2 = tmp_path / "b.txt"
        file1.write_text("content A")
        file2.write_text("content B")
        assert compute_md5(file1) != compute_md5(file2)


class TestSetAllSeeds:
    """Tests for set_all_seeds function."""

    def test_valid_seed_zero(self):
        """Should accept seed value 0."""
        set_all_seeds(0)

    def test_valid_seed_max(self):
        """Should accept maximum valid seed."""
        set_all_seeds(2**32 - 1)

    def test_negative_seed_raises(self):
        """Should raise ConfigValidationError for negative seed."""
        with pytest.raises(ConfigValidationError) as exc_info:
            set_all_seeds(-1)
        assert exc_info.value.parameter == "seed"

    def test_overflow_seed_raises(self):
        """Should raise ConfigValidationError for seed > 2^32-1."""
        with pytest.raises(ConfigValidationError) as exc_info:
            set_all_seeds(2**32)
        assert exc_info.value.parameter == "seed"

    def test_reproducible_random(self):
        """Setting same seed should produce identical random sequences."""
        import random

        set_all_seeds(42)
        seq1 = [random.random() for _ in range(10)]

        set_all_seeds(42)
        seq2 = [random.random() for _ in range(10)]

        assert seq1 == seq2

    def test_reproducible_numpy(self):
        """Setting same seed should produce identical numpy sequences."""
        set_all_seeds(42)
        arr1 = np.random.randn(10)

        set_all_seeds(42)
        arr2 = np.random.randn(10)

        np.testing.assert_array_equal(arr1, arr2)


class TestGetWorkerSeed:
    """Tests for get_worker_seed function."""

    def test_deterministic(self):
        """Same inputs should always produce same output."""
        assert get_worker_seed(42, 0) == get_worker_seed(42, 0)

    def test_different_workers_different_seeds(self):
        """Different worker IDs should produce different seeds."""
        seeds = [get_worker_seed(42, i) for i in range(10)]
        assert len(set(seeds)) == 10

    def test_different_master_different_seeds(self):
        """Different master seeds should produce different worker seeds."""
        seed1 = get_worker_seed(42, 0)
        seed2 = get_worker_seed(43, 0)
        assert seed1 != seed2

    def test_valid_range(self):
        """Output should be in valid range [0, 2^32 - 1]."""
        for worker_id in range(100):
            seed = get_worker_seed(42, worker_id)
            assert 0 <= seed < 2**32


class TestGetMemoryInfo:
    """Tests for get_memory_info function."""

    def test_returns_required_keys(self):
        """Should return dict with required RAM keys."""
        info = get_memory_info()
        assert "ram_used_mb" in info
        assert "ram_available_mb" in info
        assert "ram_total_mb" in info

    def test_positive_values(self):
        """All memory values should be positive."""
        info = get_memory_info()
        assert info["ram_used_mb"] > 0
        assert info["ram_available_mb"] > 0
        assert info["ram_total_mb"] > 0

    def test_total_equals_used_plus_available_approximately(self):
        """RAM total should approximately equal used + available."""
        info = get_memory_info()
        # Allow some tolerance since values change between calls
        total = info["ram_total_mb"]
        used_plus_avail = info["ram_used_mb"] + info["ram_available_mb"]
        # They should be within 10% given OS overhead/caching
        assert abs(total - used_plus_avail) / total < 0.5
