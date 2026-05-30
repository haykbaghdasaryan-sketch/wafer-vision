"""Unit tests for the unified Trainer class.

Tests basic flow with a tiny model and synthetic data across all 4 modes.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import BaseBackbone
from src.training.trainer import Trainer, TrainingConfig, TrainingResult
from src.exceptions import TrainingDivergenceError


class TinyBackbone(BaseBackbone):
    """Minimal backbone for testing: Linear(3*8*8 -> 32)."""

    def __init__(self) -> None:
        super().__init__(l2_normalize=False)
        self._linear = nn.Linear(3 * 8 * 8, 32)

    @property
    def embedding_dim(self) -> int:
        return 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims
        b = x.size(0)
        # Adaptive pool to 8x8 first
        x = nn.functional.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(b, -1)
        return self._linear(x)


def _make_loader(n_samples: int = 40, n_classes: int = 4, batch_size: int = 8) -> DataLoader:
    """Create a synthetic DataLoader for testing."""
    images = torch.rand(n_samples, 3, 32, 32, dtype=torch.float32)
    labels = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _make_simclr_loader(n_samples: int = 40, batch_size: int = 8) -> DataLoader:
    """Create a synthetic DataLoader with dual views for SimCLR testing."""
    view1 = torch.rand(n_samples, 3, 32, 32, dtype=torch.float32)
    view2 = torch.rand(n_samples, 3, 32, 32, dtype=torch.float32)
    labels = torch.randint(0, 4, (n_samples,))
    dataset = TensorDataset(view1, view2, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestTrainerPretrainedMode:
    """Test pretrained mode (no training)."""

    def test_pretrained_returns_immediately(self):
        model = TinyBackbone()
        loader = _make_loader()
        config = TrainingConfig(mode="pretrained")
        trainer = Trainer(config=config, model=model, train_loader=loader)
        result = trainer.train()
        assert isinstance(result, TrainingResult)
        assert result.total_epochs_trained == 0
        assert result.total_gradient_updates == 0


class TestTrainerFinetuneMode:
    """Test fine-tune mode with CrossEntropy."""

    def test_finetune_basic_flow(self):
        model = TinyBackbone()
        train_loader = _make_loader(n_samples=24, n_classes=4, batch_size=8)
        val_loader = _make_loader(n_samples=8, n_classes=4, batch_size=8)
        config = TrainingConfig(
            mode="finetune",
            num_epochs=3,
            learning_rate=1e-3,
            warmup_epochs=1,
            patience=10,
            num_classes=4,
            log_every_n_steps=1,
        )
        trainer = Trainer(
            config=config, model=model,
            train_loader=train_loader, val_loader=val_loader,
        )
        result = trainer.train()
        assert isinstance(result, TrainingResult)
        assert result.total_epochs_trained == 3
        assert result.total_gradient_updates > 0
        assert result.final_train_loss >= 0.0
        assert "train_loss" in result.history
        assert len(result.history["train_loss"]) == 3

    def test_finetune_with_class_weights(self):
        model = TinyBackbone()
        loader = _make_loader(n_samples=24, n_classes=4, batch_size=8)
        config = TrainingConfig(
            mode="finetune",
            num_epochs=2,
            warmup_epochs=0,
            learning_rate=1e-3,
            num_classes=4,
            class_counts=[10, 5, 3, 2],
            enable_class_weights=True,
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        result = trainer.train()
        assert result.total_epochs_trained == 2
        assert result.final_train_loss >= 0.0


class TestTrainerMetricMode:
    """Test metric learning mode."""

    def test_metric_triplet_basic_flow(self):
        model = TinyBackbone()
        loader = _make_loader(n_samples=32, n_classes=4, batch_size=16)
        config = TrainingConfig(
            mode="metric",
            loss_name="triplet",
            num_epochs=2,
            warmup_epochs=0,
            learning_rate=1e-3,
            margin=0.2,
            projection_hidden=64,
            projection_output=32,
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        result = trainer.train()
        assert result.total_epochs_trained == 2
        assert result.final_train_loss >= 0.0

    def test_metric_supcon_basic_flow(self):
        model = TinyBackbone()
        loader = _make_loader(n_samples=32, n_classes=4, batch_size=16)
        config = TrainingConfig(
            mode="metric",
            loss_name="supcon",
            num_epochs=2,
            warmup_epochs=0,
            learning_rate=1e-3,
            temperature=0.07,
            projection_hidden=64,
            projection_output=32,
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        result = trainer.train()
        assert result.total_epochs_trained == 2
        assert result.final_train_loss >= 0.0


class TestTrainerSelfSupervisedMode:
    """Test self-supervised (SimCLR) mode."""

    def test_selfsupervised_basic_flow(self):
        model = TinyBackbone()
        loader = _make_simclr_loader(n_samples=24, batch_size=8)
        config = TrainingConfig(
            mode="selfsupervised",
            loss_name="ntxent",
            num_epochs=2,
            warmup_epochs=0,
            learning_rate=1e-3,
            temperature=0.5,
            projection_hidden=64,
            projection_output=32,
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        result = trainer.train()
        assert result.total_epochs_trained == 2
        assert result.final_train_loss >= 0.0

    def test_selfsupervised_gradient_accumulation(self):
        model = TinyBackbone()
        loader = _make_simclr_loader(n_samples=24, batch_size=8)
        config = TrainingConfig(
            mode="selfsupervised",
            loss_name="ntxent",
            num_epochs=2,
            warmup_epochs=0,
            learning_rate=1e-3,
            temperature=0.5,
            accumulation_steps=2,
            projection_hidden=64,
            projection_output=32,
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        result = trainer.train()
        assert result.total_epochs_trained == 2
        # With accumulation_steps=2, should have fewer gradient updates
        assert result.total_gradient_updates > 0


class TestTrainerGradientClipping:
    """Test gradient clipping behavior."""

    def test_gradient_clipping_applied(self):
        model = TinyBackbone()
        loader = _make_loader(n_samples=16, n_classes=4, batch_size=8)
        config = TrainingConfig(
            mode="finetune",
            num_epochs=1,
            warmup_epochs=0,
            learning_rate=1e-3,
            gradient_clip_max_norm=0.5,
            num_classes=4,
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        result = trainer.train()
        # Should complete without error
        assert result.total_epochs_trained == 1


class TestTrainerNaNDetection:
    """Test NaN/Inf loss detection."""

    def test_nan_loss_raises_divergence_error(self):
        """Force NaN loss by using extreme learning rate or corrupted data."""
        model = TinyBackbone()
        # Create data that can cause numerical issues
        images = torch.full((8, 3, 32, 32), float("nan"), dtype=torch.float32)
        labels = torch.randint(0, 4, (8,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)

        config = TrainingConfig(
            mode="finetune",
            num_epochs=1,
            warmup_epochs=0,
            learning_rate=1e-3,
            num_classes=4,
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        with pytest.raises(TrainingDivergenceError):
            trainer.train()


class TestTrainerCheckpoint:
    """Test checkpoint save and resume."""

    def test_checkpoint_save_creates_file(self, tmp_path):
        model = TinyBackbone()
        loader = _make_loader(n_samples=16, n_classes=4, batch_size=8)
        config = TrainingConfig(
            mode="finetune",
            num_epochs=2,
            warmup_epochs=0,
            learning_rate=1e-3,
            num_classes=4,
            checkpoint_dir=str(tmp_path),
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        trainer.train()

        # Should have saved best and final checkpoints
        checkpoint_files = list(tmp_path.glob("*.pth"))
        assert len(checkpoint_files) > 0

    def test_resume_from_checkpoint(self, tmp_path):
        model = TinyBackbone()
        loader = _make_loader(n_samples=16, n_classes=4, batch_size=8)

        # First training run
        config = TrainingConfig(
            mode="finetune",
            num_epochs=2,
            warmup_epochs=0,
            learning_rate=1e-3,
            num_classes=4,
            checkpoint_dir=str(tmp_path),
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        trainer.train()

        # Find a checkpoint file
        checkpoint_files = list(tmp_path.glob("*.pth"))
        assert len(checkpoint_files) > 0
        checkpoint_path = str(checkpoint_files[0])

        # Resume training
        model2 = TinyBackbone()
        config2 = TrainingConfig(
            mode="finetune",
            num_epochs=4,
            warmup_epochs=0,
            learning_rate=1e-3,
            num_classes=4,
            checkpoint_dir=str(tmp_path),
            resume_from=checkpoint_path,
        )
        trainer2 = Trainer(config=config2, model=model2, train_loader=loader)
        result = trainer2.train()
        # Should have continued from where it left off
        assert result.total_epochs_trained > 0


class TestTrainerEarlyStopping:
    """Test early stopping integration."""

    def test_early_stopping_triggers(self):
        """With patience=2 and all-same-class data, triplet loss is 0 always -> early stop."""
        model = TinyBackbone()
        # Use metric mode with all same class -> loss = 0 always (no valid triplets)
        # This means loss never improves from 0.0, early stopping triggers
        images = torch.rand(16, 3, 32, 32, dtype=torch.float32)
        labels = torch.zeros(16, dtype=torch.long)  # all same class
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)

        config = TrainingConfig(
            mode="metric",
            loss_name="triplet",
            num_epochs=20,
            warmup_epochs=0,
            learning_rate=1e-3,
            patience=2,
            projection_hidden=64,
            projection_output=32,
        )
        trainer = Trainer(config=config, model=model, train_loader=loader)
        result = trainer.train()
        # Loss is constantly 0.0 -> no improvement -> early stopping at epoch patience+1
        # After patience (2) epochs with no improvement from best (first epoch), stops
        assert result.total_epochs_trained <= 4


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_default_values(self):
        result = TrainingResult()
        assert result.best_epoch == 0
        assert result.best_metric_value == 0.0
        assert result.total_epochs_trained == 0
        assert result.total_time_seconds == 0.0
        assert result.final_train_loss == 0.0
        assert result.final_val_loss == 0.0
        assert result.total_gradient_updates == 0
        assert result.history == {}
