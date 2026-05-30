"""Checkpoint save/load/verify with atomic writes."""

import logging
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from src.exceptions import CheckpointLoadError, InsufficientDiskError
from src.utils import atomic_write, check_disk_space

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving, loading, and integrity verification.

    Features:
        - Atomic writes (temp -> verify -> rename)
        - Integrity verification after save (10 random tensors compared)
        - Saves best, last, and intermediate checkpoints
        - Configurable retention for intermediate checkpoints

    Checkpoint contents:
        model_state_dict, optimizer_state_dict, scheduler_state_dict,
        epoch, metrics, config, random_states (Python/NumPy/Torch/CUDA).

    Args:
        checkpoint_dir: Directory for checkpoint files.
        retention: "best_and_last" (keep best + last) or "all" (keep all).
        save_every_n: Save intermediate every N epochs (0 = disabled).
        keep_last_m: Keep last M intermediate checkpoints (default 3).
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        retention: str = "best_and_last",
        save_every_n: int = 0,
        keep_last_m: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.retention = retention
        self.save_every_n = save_every_n
        self.keep_last_m = keep_last_m

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        metrics: dict[str, Any],
        config: Optional[dict] = None,
        is_best: bool = False,
    ) -> Path:
        """Save a complete checkpoint atomically.

        Saves: model_state_dict, optimizer_state_dict, scheduler_state_dict,
               epoch, metrics, config, random states (random, np, torch).
        Checks disk space >= 500MB before saving.
        Verifies integrity after save (loads back, checks 10 random tensors).

        Args:
            model: Model to save state of.
            optimizer: Optimizer state.
            scheduler: LR scheduler state (optional).
            epoch: Current epoch number.
            metrics: Dict of metric values (e.g., {"val_loss": 0.5, "knn_acc": 0.8}).
            config: Complete experiment configuration dict (optional).
            is_best: If True, also save as best_model.pth.

        Returns:
            Path to saved checkpoint file.

        Raises:
            InsufficientDiskError: If < 500 MB free.
            AtomicWriteError: If write/verification fails.
        """
        # Check disk space before saving
        check_disk_space(self.checkpoint_dir, required_mb=500.0)

        start_time = time.time()

        # Build checkpoint payload
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
            "random_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }

        # Determine target path
        target_path = self.checkpoint_dir / "last_model.pth"

        def _write_fn(path: Path) -> None:
            torch.save(checkpoint_data, path)

        def _verify_fn(path: Path) -> None:
            # Verify integrity: load back and check 10 random tensors
            self._verify_saved_checkpoint(path, model)

        # Atomic write with verification
        atomic_write(target_path, _write_fn, _verify_fn)

        save_duration_ms = (time.time() - start_time) * 1000
        file_size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Checkpoint saved",
            extra={
                "path": str(target_path),
                "epoch": epoch,
                "file_size_mb": round(file_size_mb, 2),
                "save_duration_ms": round(save_duration_ms, 1),
            },
        )

        # Save as best if flagged
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            atomic_write(best_path, _write_fn, _verify_fn)
            logger.info(
                "Best checkpoint updated",
                extra={"path": str(best_path), "epoch": epoch},
            )

        # Handle intermediate checkpoints
        if self.save_every_n > 0 and epoch % self.save_every_n == 0:
            intermediate_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            atomic_write(intermediate_path, _write_fn)
            self._enforce_retention()

        return target_path

    def load(self, path: Optional[Path] = None) -> dict:
        """Load a checkpoint. If path is None, loads the best checkpoint.

        Args:
            path: Checkpoint file path. If None, loads best_model.pth.

        Returns:
            Dict with all saved fields (model_state_dict, optimizer_state_dict,
            scheduler_state_dict, epoch, metrics, config, random_states).

        Raises:
            CheckpointLoadError: If corrupt or incompatible.
            FileNotFoundError: If checkpoint file does not exist.
        """
        if path is None:
            path = self.get_best_path()
            if path is None:
                path = self.get_last_path()
            if path is None:
                raise FileNotFoundError(
                    f"No checkpoint found in {self.checkpoint_dir}"
                )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise CheckpointLoadError(
                path=str(path),
                key="file",
                expected_shape=(),
                actual_shape=(),
            ) from e

        # Validate expected keys
        required_keys = {"model_state_dict", "epoch"}
        missing_keys = required_keys - set(checkpoint.keys())
        if missing_keys:
            raise CheckpointLoadError(
                path=str(path),
                key=str(missing_keys),
                expected_shape=(),
                actual_shape=(),
            )

        return checkpoint

    def get_best_path(self) -> Optional[Path]:
        """Get path to best checkpoint, or None."""
        best = self.checkpoint_dir / "best_model.pth"
        return best if best.exists() else None

    def get_last_path(self) -> Optional[Path]:
        """Get path to last checkpoint, or None."""
        last = self.checkpoint_dir / "last_model.pth"
        return last if last.exists() else None

    def verify_integrity(self, path: Path, model: nn.Module) -> bool:
        """Verify checkpoint integrity by comparing random weight tensors.

        Loads checkpoint and compares 10 random tensors against in-memory model.
        Max absolute difference must be < 1e-7.

        Args:
            path: Checkpoint to verify.
            model: In-memory model for comparison.

        Returns:
            True if verification passes.
        """
        try:
            self._verify_saved_checkpoint(path, model)
            return True
        except Exception:
            return False

    def _verify_saved_checkpoint(self, path: Path, model: nn.Module) -> None:
        """Internal verification: load checkpoint and compare tensors.

        Compares 10 random tensors from model state dict against saved values.
        Raises if max absolute difference >= 1e-7.
        """
        saved = torch.load(path, map_location="cpu", weights_only=False)
        saved_state = saved["model_state_dict"]
        current_state = model.state_dict()

        # Get parameter keys that are tensors
        tensor_keys = [
            k for k, v in current_state.items() if isinstance(v, torch.Tensor)
        ]

        if not tensor_keys:
            return

        # Select up to 10 random tensor keys for comparison
        n_check = min(10, len(tensor_keys))
        rng = random.Random(42)  # Deterministic selection
        check_keys = rng.sample(tensor_keys, n_check)

        for key in check_keys:
            if key not in saved_state:
                raise ValueError(
                    f"Integrity check failed: key '{key}' missing from saved checkpoint"
                )

            current_tensor = current_state[key].float()
            saved_tensor = saved_state[key].float()

            if current_tensor.shape != saved_tensor.shape:
                raise ValueError(
                    f"Integrity check failed: key '{key}' shape mismatch. "
                    f"Expected {current_tensor.shape}, got {saved_tensor.shape}"
                )

            max_diff = (current_tensor - saved_tensor).abs().max().item()
            if max_diff >= 1e-7:
                raise ValueError(
                    f"Integrity check failed: key '{key}' max diff = {max_diff:.2e} "
                    f"(threshold: 1e-7)"
                )

    def _enforce_retention(self) -> None:
        """Remove old intermediate checkpoints based on retention policy."""
        if self.retention == "all":
            return

        # Find intermediate checkpoint files
        intermediates = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda p: self._extract_epoch_from_path(p),
        )

        # Keep only last M
        if len(intermediates) > self.keep_last_m:
            to_remove = intermediates[: len(intermediates) - self.keep_last_m]
            for path in to_remove:
                path.unlink()
                logger.debug(
                    "Removed old intermediate checkpoint",
                    extra={"path": str(path)},
                )

    @staticmethod
    def _extract_epoch_from_path(path: Path) -> int:
        """Extract epoch number from checkpoint filename."""
        # Format: checkpoint_epoch_N.pth
        stem = path.stem  # checkpoint_epoch_N
        try:
            return int(stem.split("_")[-1])
        except (ValueError, IndexError):
            return 0
