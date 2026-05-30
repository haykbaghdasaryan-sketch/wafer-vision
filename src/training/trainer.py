"""Unified Trainer with all training modes.

Supports four modes:
- pretrained: Zero-shot embedding extraction (no training)
- finetune: CrossEntropy + class weights + Adam + cosine annealing
- metric: Triplet/SupCon + BalancedBatchSampler + ProjectionHead
- selfsupervised: SimCLR dual views + NT-Xent + gradient accumulation

Includes gradient clipping, NaN/Inf detection, embedding collapse detection,
per-epoch logging, step-level logging, and checkpoint resume support.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.exceptions import (
    EmbeddingCollapseWarning,
    TrainingDivergenceError,
)
from src.models.backbone import BaseBackbone
from src.models.projection_head import ProjectionHead
from src.training.early_stopping import EarlyStopping
from src.training.losses import (
    NTXentLoss,
    SupervisedContrastiveLoss,
    TripletMarginLossWithMining,
    WeightedCrossEntropyLoss,
)
from src.training.scheduler import WarmupCosineScheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration dataclass for the Trainer.

    Contains all hyperparameters needed for training across all modes.
    """

    mode: str = "finetune"
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer_name: str = "adam"
    loss_name: str = "crossentropy"
    margin: float = 0.2
    temperature: float = 0.07
    patience: int = 10
    gradient_clip_max_norm: float = 1.0
    warmup_epochs: int = 5
    accumulation_steps: int = 1
    num_classes: int = 9
    class_counts: Optional[list[int]] = None
    enable_class_weights: bool = True
    projection_hidden: int = 512
    projection_output: int = 128
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_every_n: int = 0
    resume_from: Optional[str] = None
    log_every_n_steps: int = 10
    device: str = "cpu"
    collapse_threshold: float = 0.01
    verbose: bool = False


@dataclass
class TrainingResult:
    """Results from a training run."""

    best_epoch: int = 0
    best_metric_value: float = 0.0
    total_epochs_trained: int = 0
    total_time_seconds: float = 0.0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    total_gradient_updates: int = 0
    history: dict[str, list[float]] = field(default_factory=dict)


class Trainer:
    """Unified trainer supporting all four training modes.

    Modes:
        - pretrained: No training; just returns (for embedding extraction).
        - finetune: Appends Linear(D→num_classes), CrossEntropy loss.
        - metric: Uses ProjectionHead, Triplet/SupCon loss.
        - selfsupervised: SimCLR with NT-Xent loss, gradient accumulation.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: BaseBackbone,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_manager: Optional[Any] = None,
    ) -> None:
        """Initialize the Trainer.

        Args:
            config: TrainingConfig with all hyperparameters.
            model: BaseBackbone model instance.
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            checkpoint_manager: Optional CheckpointManager for save/load.
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_manager = checkpoint_manager
        self.device = torch.device(config.device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Build mode-specific components
        self._classifier_head: Optional[nn.Linear] = None
        self._projection_head: Optional[ProjectionHead] = None
        self._loss_fn: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[WarmupCosineScheduler] = None
        self._early_stopping: Optional[EarlyStopping] = None

        # State
        self._start_epoch = 0
        self._total_gradient_updates = 0

        if config.mode != "pretrained":
            self._setup_training()

    def _setup_training(self) -> None:
        """Set up loss, optimizer, scheduler, and mode-specific heads."""
        config = self.config

        # Mode-specific heads and loss
        if config.mode == "finetune":
            self._classifier_head = nn.Linear(
                self.model.embedding_dim, config.num_classes
            ).to(self.device)
            self._loss_fn = WeightedCrossEntropyLoss(
                class_counts=config.class_counts,
                num_classes=config.num_classes,
                enable_weights=config.enable_class_weights,
            ).to(self.device)

        elif config.mode == "metric":
            self._projection_head = ProjectionHead(
                input_dim=self.model.embedding_dim,
                hidden_dim=config.projection_hidden,
                output_dim=config.projection_output,
            ).to(self.device)
            if config.loss_name == "triplet":
                self._loss_fn = TripletMarginLossWithMining(
                    margin=config.margin
                ).to(self.device)
            else:  # supcon
                self._loss_fn = SupervisedContrastiveLoss(
                    temperature=config.temperature
                ).to(self.device)

        elif config.mode == "selfsupervised":
            self._projection_head = ProjectionHead(
                input_dim=self.model.embedding_dim,
                hidden_dim=config.projection_hidden,
                output_dim=config.projection_output,
            ).to(self.device)
            self._loss_fn = NTXentLoss(
                temperature=config.temperature
            ).to(self.device)

        # Collect all trainable parameters
        params = list(self.model.parameters())
        if self._classifier_head is not None:
            params += list(self._classifier_head.parameters())
        if self._projection_head is not None:
            params += list(self._projection_head.parameters())

        # Optimizer
        if config.optimizer_name == "adam":
            self._optimizer = torch.optim.Adam(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer_name == "sgd":
            self._optimizer = torch.optim.SGD(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9,
            )
        else:
            # Default to Adam for lars or unknown
            self._optimizer = torch.optim.Adam(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # LR Scheduler
        self._scheduler = WarmupCosineScheduler(
            optimizer=self._optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.num_epochs,
        )

        # Early stopping
        self._early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=0.0,
            mode="min",  # monitor val loss
        )

        # Resume from checkpoint if specified
        if config.resume_from:
            self._resume_checkpoint(config.resume_from)

    def _resume_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint file.

        Args:
            path: Path to checkpoint .pth file.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            logger.warning("Checkpoint not found at %s, starting fresh.", path)
            return

        logger.info("Resuming training from checkpoint: %s", path)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Restore model state
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        # Restore optimizer state
        if "optimizer_state_dict" in checkpoint and self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Restore scheduler state
        if "scheduler_state_dict" in checkpoint and self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Restore epoch
        if "epoch" in checkpoint:
            self._start_epoch = checkpoint["epoch"] + 1
        # Restore gradient updates
        if "total_gradient_updates" in checkpoint:
            self._total_gradient_updates = checkpoint["total_gradient_updates"]
        # Restore classifier head
        if "classifier_head_state_dict" in checkpoint and self._classifier_head is not None:
            self._classifier_head.load_state_dict(checkpoint["classifier_head_state_dict"])
        # Restore projection head
        if "projection_head_state_dict" in checkpoint and self._projection_head is not None:
            self._projection_head.load_state_dict(checkpoint["projection_head_state_dict"])

        logger.info("Resumed from epoch %d", self._start_epoch)

    def train(self) -> TrainingResult:
        """Run the full training loop.

        Returns:
            TrainingResult with metrics from the run.

        Raises:
            TrainingDivergenceError: If loss becomes NaN/Inf.
        """
        if self.config.mode == "pretrained":
            logger.info("Pretrained mode: no training needed.")
            return TrainingResult()

        start_time = time.time()
        best_val_loss = float("inf")
        best_epoch = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        logger.info(
            "Starting training: mode=%s, epochs=%d, lr=%.2e, device=%s",
            self.config.mode,
            self.config.num_epochs,
            self.config.learning_rate,
            self.device,
        )

        for epoch in range(self._start_epoch, self.config.num_epochs):
            epoch_start = time.time()

            # Train one epoch
            train_metrics = self._train_epoch(epoch)
            train_loss = train_metrics["loss"]
            history["train_loss"].append(train_loss)

            # Validate
            val_metrics: dict[str, float] = {}
            val_loss = train_loss  # fallback if no val_loader
            if self.val_loader is not None:
                val_metrics = self._validate(epoch)
                val_loss = val_metrics.get("loss", train_loss)
            history["val_loss"].append(val_loss)

            # LR scheduler step
            current_lr = self._optimizer.param_groups[0]["lr"]  # type: ignore[union-attr]
            history["learning_rate"].append(current_lr)
            self._scheduler.step()  # type: ignore[union-attr]

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)

            # Periodic checkpoint
            if (
                self.config.checkpoint_every_n > 0
                and (epoch + 1) % self.config.checkpoint_every_n == 0
            ):
                self._save_checkpoint(epoch, is_best=False)

            # Per-epoch logging
            epoch_time = time.time() - epoch_start
            throughput = len(self.train_loader.dataset) / epoch_time if epoch_time > 0 else 0  # type: ignore[arg-type]

            logger.info(
                "Epoch %d/%d - train_loss: %.4f, val_loss: %.4f, "
                "lr: %.2e, time: %.1fs, throughput: %.0f samples/s",
                epoch + 1,
                self.config.num_epochs,
                train_loss,
                val_loss,
                current_lr,
                epoch_time,
                throughput,
            )

            # Early stopping check
            if self._early_stopping is not None:
                if self._early_stopping.step(val_loss):
                    logger.info(
                        "Early stopping triggered at epoch %d", epoch + 1
                    )
                    break

        total_time = time.time() - start_time
        total_epochs = len(history["train_loss"])

        # Save final checkpoint
        self._save_checkpoint(
            self._start_epoch + total_epochs - 1, is_best=False
        )

        result = TrainingResult(
            best_epoch=best_epoch,
            best_metric_value=best_val_loss,
            total_epochs_trained=total_epochs,
            total_time_seconds=total_time,
            final_train_loss=history["train_loss"][-1] if history["train_loss"] else 0.0,
            final_val_loss=history["val_loss"][-1] if history["val_loss"] else 0.0,
            total_gradient_updates=self._total_gradient_updates,
            history=history,
        )

        logger.info(
            "Training complete: %d epochs, best_epoch=%d, best_val_loss=%.4f, "
            "total_time=%.1fs",
            total_epochs,
            best_epoch,
            best_val_loss,
            total_time,
        )

        return result

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Dispatch to mode-specific epoch training."""
        if self.config.mode == "finetune":
            return self._train_epoch_finetune(epoch)
        elif self.config.mode == "metric":
            return self._train_epoch_metric(epoch)
        elif self.config.mode == "selfsupervised":
            return self._train_epoch_selfsupervised(epoch)
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode}")

    def _train_epoch_finetune(self, epoch: int) -> dict[str, float]:
        """Fine-tune training epoch: CrossEntropy + classification head.

        Returns:
            Dict with 'loss' and 'accuracy' keys.
        """
        self.model.train()
        self._classifier_head.train()  # type: ignore[union-attr]

        total_loss = 0.0
        correct = 0
        total_samples = 0

        self._optimizer.zero_grad()  # type: ignore[union-attr]

        for batch_idx, batch in enumerate(self.train_loader):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            # Forward: backbone -> classifier head
            embeddings = self.model(images)
            logits = self._classifier_head(embeddings)  # type: ignore[misc]
            loss = self._loss_fn(logits, labels)  # type: ignore[misc]

            # Check for NaN/Inf
            self._check_loss(loss, epoch, batch_idx)

            # Gradient accumulation
            scaled_loss = loss / self.config.accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                self._clip_gradients()
                self._optimizer.step()  # type: ignore[union-attr]
                self._optimizer.zero_grad()  # type: ignore[union-attr]
                self._total_gradient_updates += 1

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            # Step-level logging
            self._step_log(epoch, batch_idx, loss.item())

        # Flush remaining gradients
        if total_samples > 0 and (len(self.train_loader)) % self.config.accumulation_steps != 0:
            self._clip_gradients()
            self._optimizer.step()  # type: ignore[union-attr]
            self._optimizer.zero_grad()  # type: ignore[union-attr]
            self._total_gradient_updates += 1

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)

        # Embedding collapse detection
        self._check_embedding_collapse(epoch)

        return {"loss": avg_loss, "accuracy": accuracy}

    def _train_epoch_metric(self, epoch: int) -> dict[str, float]:
        """Metric learning training epoch: Triplet/SupCon + ProjectionHead.

        Returns:
            Dict with 'loss' key.
        """
        self.model.train()
        self._projection_head.train()  # type: ignore[union-attr]

        total_loss = 0.0
        total_samples = 0

        self._optimizer.zero_grad()  # type: ignore[union-attr]

        for batch_idx, batch in enumerate(self.train_loader):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            # Forward: backbone -> projection head
            embeddings = self.model(images)
            projections = self._projection_head(embeddings)  # type: ignore[misc]

            # Compute loss (Triplet or SupCon)
            loss = self._loss_fn(projections, labels)  # type: ignore[misc]

            # Check for NaN/Inf
            self._check_loss(loss, epoch, batch_idx)

            # Gradient accumulation
            scaled_loss = loss / self.config.accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                self._clip_gradients()
                self._optimizer.step()  # type: ignore[union-attr]
                self._optimizer.zero_grad()  # type: ignore[union-attr]
                self._total_gradient_updates += 1

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Step-level logging
            self._step_log(epoch, batch_idx, loss.item())

        # Flush remaining gradients
        if total_samples > 0 and len(self.train_loader) % self.config.accumulation_steps != 0:
            self._clip_gradients()
            self._optimizer.step()  # type: ignore[union-attr]
            self._optimizer.zero_grad()  # type: ignore[union-attr]
            self._total_gradient_updates += 1

        avg_loss = total_loss / max(total_samples, 1)

        # Embedding collapse detection
        self._check_embedding_collapse(epoch)

        return {"loss": avg_loss}

    def _train_epoch_selfsupervised(self, epoch: int) -> dict[str, float]:
        """Self-supervised training epoch: SimCLR dual views + NT-Xent.

        Expects the DataLoader to yield (view1, view2) tensor pairs
        or a single tensor that will be split.

        Returns:
            Dict with 'loss' key.
        """
        self.model.train()
        self._projection_head.train()  # type: ignore[union-attr]

        total_loss = 0.0
        total_samples = 0

        self._optimizer.zero_grad()  # type: ignore[union-attr]

        for batch_idx, batch in enumerate(self.train_loader):
            # Handle different batch formats:
            # Option 1: (view1, view2, labels) - SimCLR dataloader
            # Option 2: ((view1, view2), labels) - tuple views
            # Option 3: (images, labels) - we apply transform inline
            if len(batch) >= 3:
                view1, view2 = batch[0].to(self.device), batch[1].to(self.device)
            elif isinstance(batch[0], (tuple, list)):
                view1, view2 = batch[0][0].to(self.device), batch[0][1].to(self.device)
            else:
                # Single tensor: split batch in half for dual views
                images = batch[0].to(self.device)
                half = images.size(0) // 2
                if half < 1:
                    continue
                view1, view2 = images[:half], images[half : 2 * half]

            # Forward: backbone -> projection head for both views
            z_i = self._projection_head(self.model(view1))  # type: ignore[misc]
            z_j = self._projection_head(self.model(view2))  # type: ignore[misc]

            # NT-Xent loss
            loss = self._loss_fn(z_i, z_j)  # type: ignore[misc]

            # Check for NaN/Inf
            self._check_loss(loss, epoch, batch_idx)

            # Gradient accumulation (important for large effective batch)
            scaled_loss = loss / self.config.accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                self._clip_gradients()
                self._optimizer.step()  # type: ignore[union-attr]
                self._optimizer.zero_grad()  # type: ignore[union-attr]
                self._total_gradient_updates += 1

            total_loss += loss.item() * view1.size(0)
            total_samples += view1.size(0)

            # Step-level logging
            self._step_log(epoch, batch_idx, loss.item())

        # Flush remaining gradients
        if total_samples > 0 and len(self.train_loader) % self.config.accumulation_steps != 0:
            self._clip_gradients()
            self._optimizer.step()  # type: ignore[union-attr]
            self._optimizer.zero_grad()  # type: ignore[union-attr]
            self._total_gradient_updates += 1

        avg_loss = total_loss / max(total_samples, 1)

        # Embedding collapse detection
        self._check_embedding_collapse(epoch)

        return {"loss": avg_loss}

    def _validate(self, epoch: int) -> dict[str, float]:
        """Run validation loop.

        Returns:
            Dict with 'loss' and optionally 'accuracy' keys.
        """
        self.model.eval()
        if self._classifier_head is not None:
            self._classifier_head.eval()
        if self._projection_head is not None:
            self._projection_head.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:  # type: ignore[union-attr]
                if self.config.mode == "selfsupervised":
                    # For self-supervised, validate with NT-Xent on views
                    if len(batch) >= 3:
                        view1, view2 = batch[0].to(self.device), batch[1].to(self.device)
                    elif isinstance(batch[0], (tuple, list)):
                        view1, view2 = batch[0][0].to(self.device), batch[0][1].to(self.device)
                    else:
                        images = batch[0].to(self.device)
                        half = images.size(0) // 2
                        if half < 1:
                            continue
                        view1, view2 = images[:half], images[half : 2 * half]

                    z_i = self._projection_head(self.model(view1))  # type: ignore[misc]
                    z_j = self._projection_head(self.model(view2))  # type: ignore[misc]
                    loss = self._loss_fn(z_i, z_j)  # type: ignore[misc]
                    total_loss += loss.item() * view1.size(0)
                    total_samples += view1.size(0)

                elif self.config.mode == "finetune":
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                    embeddings = self.model(images)
                    logits = self._classifier_head(embeddings)  # type: ignore[misc]
                    loss = self._loss_fn(logits, labels)  # type: ignore[misc]
                    total_loss += loss.item() * images.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total_samples += images.size(0)

                elif self.config.mode == "metric":
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                    embeddings = self.model(images)
                    projections = self._projection_head(embeddings)  # type: ignore[misc]
                    loss = self._loss_fn(projections, labels)  # type: ignore[misc]
                    total_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        metrics: dict[str, float] = {"loss": avg_loss}

        if self.config.mode == "finetune" and total_samples > 0:
            metrics["accuracy"] = correct / total_samples

        return metrics

    def _check_loss(self, loss: torch.Tensor, epoch: int, batch_idx: int) -> None:
        """Check if loss is NaN or Inf and raise TrainingDivergenceError.

        Args:
            loss: Current loss tensor.
            epoch: Current epoch number.
            batch_idx: Current batch index.

        Raises:
            TrainingDivergenceError: If loss is NaN or Inf.
        """
        loss_val = loss.item()
        if not torch.isfinite(loss):
            raise TrainingDivergenceError(
                epoch=epoch, batch_idx=batch_idx, loss_value=loss_val
            )

    def _clip_gradients(self) -> None:
        """Apply gradient clipping to all trainable parameters."""
        params = list(self.model.parameters())
        if self._classifier_head is not None:
            params += list(self._classifier_head.parameters())
        if self._projection_head is not None:
            params += list(self._projection_head.parameters())

        torch.nn.utils.clip_grad_norm_(
            params, max_norm=self.config.gradient_clip_max_norm
        )

    def _check_embedding_collapse(self, epoch: int) -> None:
        """Detect embedding collapse by computing std dev of embeddings.

        If std_dev < collapse_threshold (default 0.01), logs a warning.
        Uses a small sample from the training data for efficiency.

        Args:
            epoch: Current epoch number.
        """
        self.model.eval()
        try:
            # Sample a batch for collapse check
            batch = next(iter(self.train_loader))
            images = batch[0].to(self.device)
            with torch.no_grad():
                embeddings = self.model(images)
            std_dev = embeddings.std().item()

            if std_dev < self.config.collapse_threshold:
                warning = EmbeddingCollapseWarning(epoch=epoch, std_dev=std_dev)
                logger.warning(str(warning))
        except StopIteration:
            pass
        finally:
            self.model.train()

    def _step_log(self, epoch: int, batch_idx: int, loss: float) -> None:
        """Log at step level every N batches.

        Args:
            epoch: Current epoch.
            batch_idx: Current batch index.
            loss: Current batch loss.
        """
        if self.config.log_every_n_steps > 0 and (batch_idx + 1) % self.config.log_every_n_steps == 0:
            logger.debug(
                "Epoch %d, Step %d - loss: %.4f",
                epoch + 1,
                batch_idx + 1,
                loss,
            )

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a training checkpoint.

        If a checkpoint_manager is provided, delegates to it.
        Otherwise, saves directly to checkpoint_dir.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best checkpoint so far.
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler else None,
            "total_gradient_updates": self._total_gradient_updates,
            "config": {
                "mode": self.config.mode,
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
            },
        }

        if self._classifier_head is not None:
            checkpoint_data["classifier_head_state_dict"] = self._classifier_head.state_dict()
        if self._projection_head is not None:
            checkpoint_data["projection_head_state_dict"] = self._projection_head.state_dict()

        if self.checkpoint_manager is not None:
            try:
                self.checkpoint_manager.save(checkpoint_data, epoch=epoch, is_best=is_best)
            except Exception as e:
                logger.warning("Failed to save checkpoint via manager: %s", e)
        else:
            # Direct save
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            suffix = "best" if is_best else f"epoch_{epoch}"
            path = checkpoint_dir / f"checkpoint_{suffix}.pth"
            try:
                torch.save(checkpoint_data, path)
                logger.debug("Saved checkpoint: %s", path)
            except Exception as e:
                logger.warning("Failed to save checkpoint to %s: %s", path, e)
