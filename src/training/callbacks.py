"""Training callbacks: logging, wandb, CSV fallback, and training curve plotting.

Provides:
- TrainingCallback: ABC for all callbacks.
- JSONLogCallback: Logs epoch metrics to a structured JSON Lines file.
- CSVLogCallback: Logs epoch metrics to a CSV file.
- WandBCallback: Optional wandb integration (when WANDB_API_KEY is set).
- TrainingCurvePlotter: Generates training curve PNGs at training end.
"""

import csv
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/CI environments

import matplotlib.pyplot as plt


class TrainingCallback(ABC):
    """Base class for training callbacks.

    Subclasses implement hooks that are invoked by the Trainer at
    specific points during the training lifecycle.
    """

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Zero-indexed epoch number.
            metrics: Dict of metric names to values for this epoch.
        """

    def on_training_end(self, history: dict[str, list[Any]]) -> None:
        """Called when training finishes.

        Args:
            history: Dict mapping metric names to lists of per-epoch values.
        """


class JSONLogCallback(TrainingCallback):
    """Logs epoch metrics to a structured JSON Lines file.

    Each epoch appends one JSON line containing the epoch number,
    timestamp, and all metric key-value pairs.

    The log file is stored at ``<log_dir>/training_log.jsonl``.
    """

    def __init__(self, log_dir: Path) -> None:
        """Initialize JSON log callback.

        Args:
            log_dir: Directory for the JSON log file (created if needed).
        """
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / "training_log.jsonl"

    @property
    def log_path(self) -> Path:
        """Return the path to the JSON log file."""
        return self._log_path

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Append one JSON line per epoch with all metrics.

        Args:
            epoch: Zero-indexed epoch number.
            metrics: Dict of metric names to values.
        """
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_json_default) + "\n")


class CSVLogCallback(TrainingCallback):
    """Logs epoch metrics to a CSV file.

    Writes headers on the first epoch and appends a row for each
    subsequent epoch. The CSV is stored at ``<csv_path>``.
    """

    def __init__(self, csv_path: Path) -> None:
        """Initialize CSV log callback.

        Args:
            csv_path: Path for the output CSV file. Parent dirs are created.
        """
        self._csv_path = Path(csv_path)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._headers_written = False
        self._fieldnames: list[str] = []

    @property
    def csv_path(self) -> Path:
        """Return the path to the CSV file."""
        return self._csv_path

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Write headers on first epoch, then append a row each epoch.

        Args:
            epoch: Zero-indexed epoch number.
            metrics: Dict of metric names to values.
        """
        row = {
            "epoch": epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }

        if not self._headers_written:
            self._fieldnames = list(row.keys())
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self._headers_written = True
        else:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self._fieldnames, extrasaction="ignore"
                )
                writer.writerow(row)


class WandBCallback(TrainingCallback):
    """Optional wandb integration.

    Activated only when the ``wandb`` package is importable AND the
    ``WANDB_API_KEY`` environment variable is set. If either condition
    is not met, all methods are no-ops and ``is_active`` returns False.
    """

    def __init__(
        self,
        project: str = "wafer-vision",
        run_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize wandb callback.

        Attempts to import wandb and initialize a run. Falls back
        gracefully if wandb is unavailable or API key is missing.

        Args:
            project: wandb project name.
            run_name: Optional human-readable run name.
            config: Optional hyperparameter dict to log.
        """
        self._active = False
        self._wandb: Any = None

        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            return

        try:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                reinit=True,
            )
            self._active = True
        except Exception:
            # If wandb init fails (network error, invalid key, etc.),
            # fall back silently.
            self._active = False

    @property
    def is_active(self) -> bool:
        """Return True if wandb is initialized and logging."""
        return self._active

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Log metrics to wandb.

        Args:
            epoch: Zero-indexed epoch number.
            metrics: Dict of metric names to values.
        """
        if not self._active:
            return
        self._wandb.log({"epoch": epoch, **metrics})

    def on_training_end(self, history: dict[str, list[Any]]) -> None:
        """Finish the wandb run.

        Args:
            history: Full training history dict.
        """
        if not self._active:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass


class TrainingCurvePlotter(TrainingCallback):
    """Generates training curve PNG figures at the end of training.

    Produces up to three plots:
    - ``loss_curve.png``: Train and validation loss over epochs.
    - ``lr_schedule.png``: Learning rate schedule over epochs.
    - ``metrics.png``: Accuracy and/or Recall metrics over epochs.

    All figures are saved at 300 DPI, 10×6 inches.
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize curve plotter.

        Args:
            output_dir: Directory for output PNG files (created if needed).
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        """Return the output directory path."""
        return self._output_dir

    def on_training_end(self, history: dict[str, list[Any]]) -> None:
        """Generate training curve PNGs from the training history.

        Args:
            history: Dict mapping metric names to lists of per-epoch values.
                Expected keys include subsets of: train_loss, val_loss,
                learning_rate, train_accuracy, val_accuracy,
                val_recall_at_1, val_recall_at_5.
        """
        self._plot_loss_curve(history)
        self._plot_lr_schedule(history)
        self._plot_metrics(history)

    def _plot_loss_curve(self, history: dict[str, list[Any]]) -> None:
        """Plot train/val loss curves."""
        has_train_loss = "train_loss" in history and history["train_loss"]
        has_val_loss = "val_loss" in history and history["val_loss"]

        if not has_train_loss and not has_val_loss:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if has_train_loss:
            epochs = list(range(len(history["train_loss"])))
            ax.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue")

        if has_val_loss:
            epochs = list(range(len(history["val_loss"])))
            ax.plot(
                epochs, history["val_loss"], label="Val Loss", color="tab:orange"
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self._output_dir / "loss_curve.png", dpi=300)
        plt.close(fig)

    def _plot_lr_schedule(self, history: dict[str, list[Any]]) -> None:
        """Plot learning rate schedule."""
        if "learning_rate" not in history or not history["learning_rate"]:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = list(range(len(history["learning_rate"])))
        ax.plot(epochs, history["learning_rate"], color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self._output_dir / "lr_schedule.png", dpi=300)
        plt.close(fig)

    def _plot_metrics(self, history: dict[str, list[Any]]) -> None:
        """Plot accuracy and recall metrics."""
        metric_keys = [
            k
            for k in (
                "train_accuracy",
                "val_accuracy",
                "val_recall_at_1",
                "val_recall_at_5",
            )
            if k in history and history[k]
        ]

        if not metric_keys:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]
        for idx, key in enumerate(metric_keys):
            epochs = list(range(len(history[key])))
            label = key.replace("_", " ").title()
            ax.plot(epochs, history[key], label=label, color=colors[idx % len(colors)])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        ax.set_title("Training Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self._output_dir / "metrics.png", dpi=300)
        plt.close(fig)


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for non-standard types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        # Handle numpy/torch scalars
        return obj.item()
    return str(obj)
