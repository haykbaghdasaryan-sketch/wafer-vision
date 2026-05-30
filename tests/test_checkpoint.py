"""Unit tests for CheckpointManager: save/load round-trip, integrity verification, error handling."""

import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.exceptions import CheckpointLoadError, InsufficientDiskError
from src.training.checkpoint import CheckpointManager


class SimpleModel(nn.Module):
    """A simple model for testing checkpoints."""

    def __init__(self, input_dim: int = 16, output_dim: int = 4):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(8, output_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


@pytest.fixture
def model():
    """Create a simple model for testing."""
    torch.manual_seed(0)
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Create an optimizer for the model."""
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def scheduler(optimizer):
    """Create a scheduler for the optimizer."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


@pytest.fixture
def checkpoint_manager(tmp_path):
    """Create a CheckpointManager with a temporary directory."""
    return CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")


class TestCheckpointSaveLoad:
    """Test save and load round-trip functionality."""

    def test_save_creates_last_model(self, checkpoint_manager, model, optimizer, scheduler):
        """Save should create last_model.pth."""
        path = checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics={"val_loss": 0.5},
        )
        assert path.exists()
        assert path.name == "last_model.pth"

    def test_save_best_creates_best_model(self, checkpoint_manager, model, optimizer, scheduler):
        """Save with is_best=True should also create best_model.pth."""
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics={"val_loss": 0.3},
            is_best=True,
        )
        best_path = checkpoint_manager.get_best_path()
        assert best_path is not None
        assert best_path.exists()

    def test_load_roundtrip_model_state(self, checkpoint_manager, model, optimizer, scheduler):
        """Loading a saved checkpoint should restore model state exactly."""
        # Save original weights
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            metrics={"val_loss": 0.2, "knn_acc": 0.85},
            config={"backbone": "resnet50", "lr": 0.001},
        )

        # Load checkpoint
        checkpoint = checkpoint_manager.load()

        # Verify all keys present
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "metrics" in checkpoint
        assert "config" in checkpoint
        assert "random_states" in checkpoint

        # Verify epoch and metrics
        assert checkpoint["epoch"] == 5
        assert checkpoint["metrics"]["val_loss"] == 0.2
        assert checkpoint["metrics"]["knn_acc"] == 0.85
        assert checkpoint["config"]["backbone"] == "resnet50"

        # Verify model state matches
        for key in original_state:
            assert torch.allclose(
                checkpoint["model_state_dict"][key],
                original_state[key],
                atol=1e-7,
            )

    def test_load_restores_optimizer_state(self, checkpoint_manager, model, optimizer, scheduler):
        """Loading should restore optimizer state."""
        # Run a step to create optimizer state
        x = torch.randn(2, 16)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        original_opt_state = optimizer.state_dict()

        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            metrics={"val_loss": 0.4},
        )

        checkpoint = checkpoint_manager.load()
        assert "optimizer_state_dict" in checkpoint

        # Verify param_groups are preserved
        loaded_groups = checkpoint["optimizer_state_dict"]["param_groups"]
        assert len(loaded_groups) == len(original_opt_state["param_groups"])
        assert loaded_groups[0]["lr"] == original_opt_state["param_groups"][0]["lr"]

    def test_load_without_path_uses_best(self, checkpoint_manager, model, optimizer, scheduler):
        """load() without path should try best_model.pth first."""
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            metrics={"val_loss": 0.1},
            is_best=True,
        )

        checkpoint = checkpoint_manager.load()
        assert checkpoint["epoch"] == 3

    def test_load_without_path_falls_back_to_last(self, checkpoint_manager, model, optimizer, scheduler):
        """load() without path should fall back to last_model.pth if no best."""
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=7,
            metrics={"val_loss": 0.6},
            is_best=False,
        )

        checkpoint = checkpoint_manager.load()
        assert checkpoint["epoch"] == 7

    def test_save_without_scheduler(self, checkpoint_manager, model, optimizer):
        """Save should handle None scheduler gracefully."""
        path = checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            metrics={"val_loss": 0.5},
        )
        checkpoint = checkpoint_manager.load(path)
        assert checkpoint["scheduler_state_dict"] is None

    def test_random_states_saved_and_loaded(self, checkpoint_manager, model, optimizer, scheduler):
        """Random states should be saved and loadable."""
        # Set known seed states
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)

        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics={"val_loss": 0.5},
        )

        checkpoint = checkpoint_manager.load()
        assert "random_states" in checkpoint
        assert "python" in checkpoint["random_states"]
        assert "numpy" in checkpoint["random_states"]
        assert "torch" in checkpoint["random_states"]


class TestCheckpointIntegrity:
    """Test integrity verification functionality."""

    def test_verify_integrity_passes_for_valid_checkpoint(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Verify should return True for a valid checkpoint just saved."""
        path = checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics={"val_loss": 0.5},
        )
        assert checkpoint_manager.verify_integrity(path, model)

    def test_verify_integrity_fails_for_modified_model(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Verify should return False if model has been modified after save."""
        path = checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics={"val_loss": 0.5},
        )

        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        assert not checkpoint_manager.verify_integrity(path, model)

    def test_verify_integrity_fails_for_corrupt_file(
        self, checkpoint_manager, model, optimizer, scheduler, tmp_path
    ):
        """Verify should return False for a corrupted file."""
        corrupt_path = tmp_path / "corrupt.pth"
        corrupt_path.write_bytes(b"not a valid checkpoint")
        assert not checkpoint_manager.verify_integrity(corrupt_path, model)


class TestCheckpointRetention:
    """Test intermediate checkpoint retention policies."""

    def test_intermediate_checkpoints_saved(self, tmp_path, model, optimizer, scheduler):
        """Intermediate checkpoints should be saved every N epochs."""
        mgr = CheckpointManager(
            checkpoint_dir=tmp_path / "ckpts",
            save_every_n=2,
            keep_last_m=3,
        )

        for epoch in range(1, 7):
            mgr.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"val_loss": 1.0 / epoch},
            )

        # Epochs 2, 4, 6 should have intermediate checkpoints
        # With keep_last_m=3, all should be kept
        ckpt_dir = tmp_path / "ckpts"
        intermediates = list(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        assert len(intermediates) == 3

    def test_retention_removes_old_intermediates(self, tmp_path, model, optimizer, scheduler):
        """Old intermediate checkpoints beyond keep_last_m should be removed."""
        mgr = CheckpointManager(
            checkpoint_dir=tmp_path / "ckpts",
            save_every_n=1,
            keep_last_m=2,
        )

        for epoch in range(1, 6):
            mgr.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"val_loss": 0.5},
            )

        ckpt_dir = tmp_path / "ckpts"
        intermediates = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        # Should only keep last 2
        assert len(intermediates) == 2
        # Should be epochs 4 and 5
        assert "epoch_4" in intermediates[0].name
        assert "epoch_5" in intermediates[1].name

    def test_retention_all_keeps_everything(self, tmp_path, model, optimizer, scheduler):
        """retention='all' should keep all intermediate checkpoints."""
        mgr = CheckpointManager(
            checkpoint_dir=tmp_path / "ckpts",
            retention="all",
            save_every_n=1,
            keep_last_m=2,
        )

        for epoch in range(1, 5):
            mgr.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"val_loss": 0.5},
            )

        ckpt_dir = tmp_path / "ckpts"
        intermediates = list(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        assert len(intermediates) == 4


class TestCheckpointErrorHandling:
    """Test error conditions."""

    def test_load_nonexistent_raises_file_not_found(self, checkpoint_manager):
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load(Path("/nonexistent/path/model.pth"))

    def test_load_no_checkpoints_raises_file_not_found(self, tmp_path):
        """Loading without any checkpoints should raise FileNotFoundError."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path / "empty_ckpts")
        with pytest.raises(FileNotFoundError):
            mgr.load()

    def test_load_corrupt_file_raises_checkpoint_load_error(self, checkpoint_manager, tmp_path):
        """Loading a corrupt file should raise CheckpointLoadError."""
        corrupt_path = checkpoint_manager.checkpoint_dir / "bad.pth"
        corrupt_path.write_bytes(b"garbage data here")
        with pytest.raises(CheckpointLoadError):
            checkpoint_manager.load(corrupt_path)

    def test_load_missing_keys_raises_checkpoint_load_error(self, checkpoint_manager):
        """Loading a checkpoint with missing required keys should raise error."""
        bad_path = checkpoint_manager.checkpoint_dir / "missing_keys.pth"
        torch.save({"some_key": "value"}, bad_path)
        with pytest.raises(CheckpointLoadError):
            checkpoint_manager.load(bad_path)

    def test_insufficient_disk_space_raises_error(self, tmp_path, model, optimizer, scheduler):
        """Save should raise InsufficientDiskError when disk space < 500 MB."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path / "ckpts")

        with patch("src.training.checkpoint.check_disk_space") as mock_check:
            mock_check.side_effect = InsufficientDiskError(
                required_mb=500.0, available_mb=100.0, path=str(tmp_path)
            )
            with pytest.raises(InsufficientDiskError):
                mgr.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=1,
                    metrics={"val_loss": 0.5},
                )


class TestCheckpointHelpers:
    """Test helper methods."""

    def test_get_best_path_returns_none_if_no_best(self, checkpoint_manager):
        """get_best_path should return None if no best checkpoint exists."""
        assert checkpoint_manager.get_best_path() is None

    def test_get_last_path_returns_none_if_no_last(self, checkpoint_manager):
        """get_last_path should return None if no last checkpoint exists."""
        assert checkpoint_manager.get_last_path() is None

    def test_get_best_path_after_save(self, checkpoint_manager, model, optimizer, scheduler):
        """get_best_path should return path after saving best."""
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics={"val_loss": 0.3},
            is_best=True,
        )
        assert checkpoint_manager.get_best_path() is not None

    def test_get_last_path_after_save(self, checkpoint_manager, model, optimizer, scheduler):
        """get_last_path should return path after any save."""
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics={"val_loss": 0.5},
        )
        assert checkpoint_manager.get_last_path() is not None
