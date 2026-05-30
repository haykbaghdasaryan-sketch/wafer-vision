"""Unit tests for training callbacks (JSON and CSV logging)."""

import csv
import json
from pathlib import Path

import pytest

from src.training.callbacks import (
    CSVLogCallback,
    JSONLogCallback,
    TrainingCallback,
    TrainingCurvePlotter,
    WandBCallback,
)


class TestTrainingCallbackABC:
    """Test the base TrainingCallback interface."""

    def test_base_class_has_default_methods(self):
        """TrainingCallback methods are not abstract; they default to no-ops."""

        class MyCallback(TrainingCallback):
            pass

        cb = MyCallback()
        # Should not raise
        cb.on_epoch_end(0, {"loss": 0.5})
        cb.on_training_end({"loss": [0.5]})


class TestJSONLogCallback:
    """Test JSONLogCallback writing structured JSON lines."""

    def test_creates_log_dir(self, tmp_path: Path):
        log_dir = tmp_path / "logs" / "nested"
        cb = JSONLogCallback(log_dir=log_dir)
        assert log_dir.exists()
        assert cb.log_path == log_dir / "training_log.jsonl"

    def test_writes_single_epoch(self, tmp_path: Path):
        cb = JSONLogCallback(log_dir=tmp_path)
        cb.on_epoch_end(0, {"train_loss": 1.5, "val_loss": 1.2})

        lines = cb.log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["epoch"] == 0
        assert entry["train_loss"] == 1.5
        assert entry["val_loss"] == 1.2
        assert "timestamp" in entry

    def test_appends_multiple_epochs(self, tmp_path: Path):
        cb = JSONLogCallback(log_dir=tmp_path)
        for epoch in range(5):
            cb.on_epoch_end(epoch, {"loss": 1.0 - epoch * 0.1})

        lines = cb.log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert entry["epoch"] == i
            assert abs(entry["loss"] - (1.0 - i * 0.1)) < 1e-9

    def test_handles_various_metric_types(self, tmp_path: Path):
        cb = JSONLogCallback(log_dir=tmp_path)
        metrics = {
            "loss": 0.5,
            "accuracy": 0.95,
            "learning_rate": 1e-4,
            "epoch_duration": 45.2,
        }
        cb.on_epoch_end(3, metrics)

        entry = json.loads(cb.log_path.read_text(encoding="utf-8").strip())
        assert entry["loss"] == 0.5
        assert entry["accuracy"] == 0.95
        assert entry["learning_rate"] == 1e-4
        assert entry["epoch_duration"] == 45.2

    def test_timestamp_is_iso_format(self, tmp_path: Path):
        cb = JSONLogCallback(log_dir=tmp_path)
        cb.on_epoch_end(0, {"loss": 1.0})

        entry = json.loads(cb.log_path.read_text(encoding="utf-8").strip())
        # Should be a valid ISO timestamp
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(entry["timestamp"])
        assert parsed.tzinfo is not None  # Should have timezone info

    def test_handles_empty_metrics(self, tmp_path: Path):
        cb = JSONLogCallback(log_dir=tmp_path)
        cb.on_epoch_end(0, {})

        entry = json.loads(cb.log_path.read_text(encoding="utf-8").strip())
        assert entry["epoch"] == 0
        assert "timestamp" in entry


class TestCSVLogCallback:
    """Test CSVLogCallback writing epoch metrics as CSV."""

    def test_creates_parent_dirs(self, tmp_path: Path):
        csv_path = tmp_path / "results" / "logs" / "metrics.csv"
        cb = CSVLogCallback(csv_path=csv_path)
        assert csv_path.parent.exists()
        assert cb.csv_path == csv_path

    def test_writes_headers_and_first_row(self, tmp_path: Path):
        csv_path = tmp_path / "metrics.csv"
        cb = CSVLogCallback(csv_path=csv_path)
        cb.on_epoch_end(0, {"loss": 1.5, "accuracy": 0.6})

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["epoch"] == "0"
        assert rows[0]["loss"] == "1.5"
        assert rows[0]["accuracy"] == "0.6"
        assert "timestamp" in rows[0]

    def test_appends_multiple_rows(self, tmp_path: Path):
        csv_path = tmp_path / "metrics.csv"
        cb = CSVLogCallback(csv_path=csv_path)

        for epoch in range(3):
            cb.on_epoch_end(epoch, {"loss": 1.0 - epoch * 0.2})

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        for i, row in enumerate(rows):
            assert row["epoch"] == str(i)
            assert abs(float(row["loss"]) - (1.0 - i * 0.2)) < 1e-9

    def test_headers_not_repeated(self, tmp_path: Path):
        csv_path = tmp_path / "metrics.csv"
        cb = CSVLogCallback(csv_path=csv_path)

        cb.on_epoch_end(0, {"loss": 1.0})
        cb.on_epoch_end(1, {"loss": 0.8})

        text = csv_path.read_text(encoding="utf-8")
        # "epoch" should appear exactly once as header
        header_count = text.strip().split("\n")[0].count("epoch")
        assert header_count == 1
        assert text.strip().count("\n") == 2  # header + 2 data rows

    def test_handles_consistent_fields(self, tmp_path: Path):
        csv_path = tmp_path / "metrics.csv"
        cb = CSVLogCallback(csv_path=csv_path)

        cb.on_epoch_end(0, {"loss": 1.0, "lr": 0.001})
        cb.on_epoch_end(1, {"loss": 0.8, "lr": 0.0005})

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["lr"] == "0.001"
        assert rows[1]["lr"] == "0.0005"


class TestWandBCallback:
    """Test WandBCallback graceful fallback behavior."""

    def test_inactive_when_no_api_key(self, monkeypatch):
        """WandBCallback is inactive when WANDB_API_KEY not set."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        cb = WandBCallback()
        assert cb.is_active is False

    def test_on_epoch_end_noop_when_inactive(self, monkeypatch):
        """on_epoch_end does nothing when wandb is inactive."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        cb = WandBCallback()
        # Should not raise
        cb.on_epoch_end(0, {"loss": 0.5})

    def test_on_training_end_noop_when_inactive(self, monkeypatch):
        """on_training_end does nothing when wandb is inactive."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        cb = WandBCallback()
        # Should not raise
        cb.on_training_end({"loss": [0.5, 0.4]})


class TestTrainingCurvePlotter:
    """Test TrainingCurvePlotter PNG generation."""

    def test_creates_output_dir(self, tmp_path: Path):
        out_dir = tmp_path / "figures" / "nested"
        cb = TrainingCurvePlotter(output_dir=out_dir)
        assert out_dir.exists()
        assert cb.output_dir == out_dir

    def test_generates_loss_curve(self, tmp_path: Path):
        cb = TrainingCurvePlotter(output_dir=tmp_path)
        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.4, 0.3],
            "val_loss": [1.1, 0.9, 0.7, 0.5, 0.4],
        }
        cb.on_training_end(history)

        loss_png = tmp_path / "loss_curve.png"
        assert loss_png.exists()
        assert loss_png.stat().st_size > 0

    def test_generates_lr_schedule(self, tmp_path: Path):
        cb = TrainingCurvePlotter(output_dir=tmp_path)
        history = {
            "learning_rate": [0.001, 0.0008, 0.0005, 0.0002, 0.0001],
        }
        cb.on_training_end(history)

        lr_png = tmp_path / "lr_schedule.png"
        assert lr_png.exists()
        assert lr_png.stat().st_size > 0

    def test_generates_metrics_plot(self, tmp_path: Path):
        cb = TrainingCurvePlotter(output_dir=tmp_path)
        history = {
            "train_accuracy": [0.5, 0.6, 0.7, 0.8, 0.85],
            "val_accuracy": [0.45, 0.55, 0.65, 0.75, 0.8],
        }
        cb.on_training_end(history)

        metrics_png = tmp_path / "metrics.png"
        assert metrics_png.exists()
        assert metrics_png.stat().st_size > 0

    def test_generates_all_plots(self, tmp_path: Path):
        cb = TrainingCurvePlotter(output_dir=tmp_path)
        history = {
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
            "learning_rate": [0.001, 0.0008, 0.0005],
            "train_accuracy": [0.5, 0.6, 0.7],
            "val_accuracy": [0.48, 0.58, 0.68],
        }
        cb.on_training_end(history)

        assert (tmp_path / "loss_curve.png").exists()
        assert (tmp_path / "lr_schedule.png").exists()
        assert (tmp_path / "metrics.png").exists()

    def test_skips_missing_metrics_gracefully(self, tmp_path: Path):
        cb = TrainingCurvePlotter(output_dir=tmp_path)
        # Empty history - no plots should be generated, no errors
        cb.on_training_end({})

        assert not (tmp_path / "loss_curve.png").exists()
        assert not (tmp_path / "lr_schedule.png").exists()
        assert not (tmp_path / "metrics.png").exists()

    def test_handles_single_epoch(self, tmp_path: Path):
        cb = TrainingCurvePlotter(output_dir=tmp_path)
        history = {
            "train_loss": [1.0],
            "val_loss": [1.1],
        }
        cb.on_training_end(history)

        assert (tmp_path / "loss_curve.png").exists()

    def test_handles_recall_metrics(self, tmp_path: Path):
        cb = TrainingCurvePlotter(output_dir=tmp_path)
        history = {
            "val_recall_at_1": [0.3, 0.4, 0.5],
            "val_recall_at_5": [0.5, 0.6, 0.7],
        }
        cb.on_training_end(history)

        assert (tmp_path / "metrics.png").exists()
