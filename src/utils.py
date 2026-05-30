"""Shared utilities: atomic writes, checksums, seeding, resource monitoring."""

import hashlib
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import psutil
import torch

from src.exceptions import (
    AtomicWriteError,
    ConfigValidationError,
    InsufficientDiskError,
)


def check_disk_space(path: Path, required_mb: float = 500.0) -> None:
    """Check available disk space and raise if insufficient.

    Args:
        path: Path on the target filesystem.
        required_mb: Minimum required free space in MB.

    Raises:
        InsufficientDiskError: If free space < required_mb.
    """
    path = Path(path)
    # Use the path itself or its parent if it doesn't exist yet
    check_path = path if path.exists() else path.parent
    while not check_path.exists():
        check_path = check_path.parent

    disk_usage = shutil.disk_usage(check_path)
    available_mb = disk_usage.free / (1024 * 1024)

    if available_mb < required_mb:
        raise InsufficientDiskError(
            required_mb=required_mb,
            available_mb=available_mb,
            path=str(path),
        )


def atomic_write(
    target_path: Path,
    write_fn: Callable[[Path], Any],
    verify_fn: Optional[Callable[[Path], Any]] = None,
) -> None:
    """Write a file atomically: write to temp -> verify -> rename.

    Prevents corruption from interrupted writes.

    Args:
        target_path: Final destination path.
        write_fn: Function that writes content to a given path.
        verify_fn: Optional function that verifies the written file is valid.

    Raises:
        AtomicWriteError: If write or verification fails.
        InsufficientDiskError: If disk space < 500 MB free.
    """
    target_path = Path(target_path)

    # Check disk space before writing
    check_disk_space(target_path, required_mb=500.0)

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic rename
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=str(target_path.parent),
            prefix=f".{target_path.name}.",
            suffix=".tmp",
        )
        os.close(temp_fd)
        temp_fd = None
        temp_path = Path(temp_path_str)

        # Write to temp file
        write_fn(temp_path)

        # Verify if verification function provided
        if verify_fn is not None:
            verify_fn(temp_path)

        # Atomic rename (on same filesystem)
        temp_path.replace(target_path)

    except InsufficientDiskError:
        # Re-raise disk space errors as-is
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise
    except Exception as e:
        # Clean up temp file on any failure
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise AtomicWriteError(
            target_path=str(target_path),
            temp_path=str(temp_path) if temp_path else "unknown",
            cause=str(e),
        ) from e


def compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file for integrity checking.

    Args:
        file_path: Path to file.

    Returns:
        Hex string of MD5 digest.
    """
    file_path = Path(file_path)
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def set_all_seeds(seed: int) -> None:
    """Set random seeds for full reproducibility.

    Sets: random.seed, np.random.seed, torch.manual_seed,
    torch.cuda.manual_seed_all, and CUDNN deterministic flags.

    Args:
        seed: Integer seed in range [0, 2^32 - 1].

    Raises:
        ConfigValidationError: If seed is out of valid range.
    """
    max_seed = 2**32 - 1
    if not isinstance(seed, int) or seed < 0 or seed > max_seed:
        raise ConfigValidationError(
            parameter="seed",
            value=seed,
            valid_range=f"[0, {max_seed}]",
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_worker_seed(master_seed: int, worker_id: int) -> int:
    """Derive a deterministic worker seed from master seed.

    Uses a simple but effective combination of master seed and worker id
    to produce unique, reproducible per-worker seeds.

    Args:
        master_seed: The global random seed.
        worker_id: DataLoader worker ID.

    Returns:
        Derived seed unique to this worker, within valid range [0, 2^32 - 1].
    """
    # Use hash-based derivation for good distribution
    combined = f"{master_seed}:{worker_id}".encode("utf-8")
    hash_val = int(hashlib.md5(combined).hexdigest(), 16)
    return hash_val % (2**32)


def get_memory_info() -> dict[str, float]:
    """Get current memory usage information.

    Returns:
        Dict with keys: ram_used_mb, ram_available_mb, ram_total_mb,
        and optionally gpu_used_mb, gpu_total_mb if CUDA available.
    """
    mem = psutil.virtual_memory()
    info: dict[str, float] = {
        "ram_used_mb": mem.used / (1024 * 1024),
        "ram_available_mb": mem.available / (1024 * 1024),
        "ram_total_mb": mem.total / (1024 * 1024),
    }

    if torch.cuda.is_available():
        try:
            # Get memory for the current default device
            device = torch.cuda.current_device()
            gpu_mem = torch.cuda.mem_get_info(device)
            # mem_get_info returns (free, total)
            gpu_free = gpu_mem[0] / (1024 * 1024)
            gpu_total = gpu_mem[1] / (1024 * 1024)
            info["gpu_used_mb"] = gpu_total - gpu_free
            info["gpu_total_mb"] = gpu_total
        except Exception:
            # If GPU memory query fails, skip GPU info
            pass

    return info
