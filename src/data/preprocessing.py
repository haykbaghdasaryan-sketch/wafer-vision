"""WaferPreprocessor with validation and stratified splitting.

Implements the complete preprocessing pipeline for WM-811K wafer maps:
- Resize to configurable target resolution using bilinear interpolation
- Normalize pixel values (divide by 2) to {0.0, 0.5, 1.0}
- Convert to 3-channel float32 tensors for backbone compatibility
- Stratified train/val/test splitting preserving class proportions
- Atomic persistence of preprocessed splits to .pt files with metadata
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.data.loader import WaferRecord
from src.utils import atomic_write, compute_md5

logger = logging.getLogger(__name__)

# Valid range for target resolution
MIN_TARGET_SIZE = 32
MAX_TARGET_SIZE = 224

# Minimum spatial dimensions for a wafer map to be processed
MIN_SPATIAL_DIM = 5


class WaferPreprocessor:
    """Preprocesses wafer maps into tensors suitable for neural network backbones.

    Features:
    - Configurable target resolution (32-224)
    - Bilinear interpolation for resizing
    - Value normalization to {0.0, 0.5, 1.0}
    - 3-channel tensor output (float32)
    - Stratified train/val/test splitting with deterministic seed
    - Atomic persistence with metadata (manifest.json, label_to_index.json)

    Usage:
        preprocessor = WaferPreprocessor(target_size=64)
        tensor = preprocessor.preprocess(wafer_map)

        splits = preprocessor.create_splits(records, seed=42)
        preprocessor.save_splits(Path("data/processed"), splits)
    """

    def __init__(self, target_size: int = 64) -> None:
        """Initialize the preprocessor with a target resolution.

        Args:
            target_size: Target spatial resolution (height and width).
                Must be in range [32, 224].

        Raises:
            ValueError: If target_size is outside [32, 224].
        """
        if not (MIN_TARGET_SIZE <= target_size <= MAX_TARGET_SIZE):
            raise ValueError(
                f"Target resolution must be in range [{MIN_TARGET_SIZE}, "
                f"{MAX_TARGET_SIZE}], got: {target_size}"
            )
        self.target_size = target_size

    def preprocess(self, wafer_map: np.ndarray) -> torch.Tensor:
        """Preprocess a single wafer map into a 3-channel float32 tensor.

        Steps:
            1. Resize to (target_size, target_size) using bilinear interpolation
            2. Normalize by dividing by 2.0 → values in {0.0, 0.5, 1.0}
            3. Repeat to 3 channels: (1, H, W) → (3, H, W)
            4. Output dtype: torch.float32

        Args:
            wafer_map: 2D numpy array with integer values in {0, 1, 2}.

        Returns:
            Tensor of shape (3, target_size, target_size) with dtype float32
            and values in {0.0, 0.5, 1.0}.
        """
        # Convert to float32 tensor and add batch+channel dims for interpolation
        # Shape: (H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(wafer_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        # Resize using bilinear interpolation
        resized = F.interpolate(
            tensor,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        # Remove batch dim: (1, 1, H, W) -> (1, H, W)
        resized = resized.squeeze(0)

        # Normalize: divide by 2.0
        normalized = resized / 2.0

        # Repeat to 3 channels: (1, H, W) -> (3, H, W)
        three_channel = normalized.repeat(3, 1, 1)

        return three_channel

    def preprocess_batch(
        self, records: list[WaferRecord]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """Preprocess a batch of wafer records into tensors.

        Filters out records with spatial dimensions < 5×5 and unlabeled records.
        Builds a label-to-index mapping for supervised training.

        Args:
            records: List of WaferRecord instances to preprocess.

        Returns:
            Tuple of (data_tensor, label_tensor, label_to_index):
                - data_tensor: shape (N, 3, target_size, target_size), float32
                - label_tensor: shape (N,), int64
                - label_to_index: mapping from class name to integer index
        """
        # Filter records: exclude unlabeled/unknown and small maps
        valid_records = self._filter_records(records)

        # Build label-to-index mapping (sorted for determinism)
        labels = sorted(set(r.label for r in valid_records))
        label_to_index: dict[str, int] = {label: idx for idx, label in enumerate(labels)}

        # Preprocess each record
        tensors: list[torch.Tensor] = []
        label_indices: list[int] = []

        for record in valid_records:
            tensor = self.preprocess(record.wafer_map)
            tensors.append(tensor)
            label_indices.append(label_to_index[record.label])

        # Stack into batch tensors
        data_tensor = torch.stack(tensors)  # (N, 3, H, W)
        label_tensor = torch.tensor(label_indices, dtype=torch.int64)

        return data_tensor, label_tensor, label_to_index

    def create_splits(
        self,
        records: list[WaferRecord],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Create stratified train/val/test splits from wafer records.

        Uses sklearn's train_test_split with stratification to ensure each split
        preserves class proportions within ±1% relative deviation.

        Args:
            records: List of labeled WaferRecord instances.
            train_ratio: Proportion of data for training (default 0.7).
            val_ratio: Proportion of data for validation (default 0.15).
            test_ratio: Proportion of data for testing (default 0.15).
            seed: Random seed for reproducibility (default 42).

        Returns:
            Dictionary with keys:
                - "train_data": Tensor (N_train, 3, H, W)
                - "train_labels": Tensor (N_train,) int64
                - "val_data": Tensor (N_val, 3, H, W)
                - "val_labels": Tensor (N_val,) int64
                - "test_data": Tensor (N_test, 3, H, W)
                - "test_labels": Tensor (N_test,) int64
                - "label_to_index": dict mapping class names to indices

        Raises:
            ValueError: If ratios don't sum to 1.0 (±0.01).
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0 (±0.01), got {total_ratio:.3f}"
            )

        # Filter records (exclude unlabeled/unknown and small maps)
        valid_records = self._filter_records(records)

        # Build label-to-index mapping (sorted for determinism)
        labels_set = sorted(set(r.label for r in valid_records))
        label_to_index: dict[str, int] = {
            label: idx for idx, label in enumerate(labels_set)
        }

        # Build label array without preprocessing all data yet (memory efficient)
        label_indices: list[int] = []
        for record in valid_records:
            label_indices.append(label_to_index[record.label])

        label_array = np.array(label_indices)

        # Stratified split: first split off test, then split remainder into train/val
        # val_ratio relative to (train + val) after removing test
        val_relative = val_ratio / (train_ratio + val_ratio)

        indices = np.arange(len(valid_records))

        # Split into train+val and test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=label_array,
        )

        # Split train+val into train and val
        train_val_labels = label_array[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_relative,
            random_state=seed,
            stratify=train_val_labels,
        )

        # Verify no overlap
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())

        assert train_set.isdisjoint(val_set), "Train and val splits overlap!"
        assert train_set.isdisjoint(test_set), "Train and test splits overlap!"
        assert val_set.isdisjoint(test_set), "Val and test splits overlap!"

        # Verify per-class proportion within ±1% relative deviation
        self._verify_proportions(label_array, train_idx, val_idx, test_idx)

        # Now preprocess each split separately (memory efficient)
        label_tensor = torch.tensor(label_indices, dtype=torch.int64)

        splits: dict[str, Any] = {"label_to_index": label_to_index}

        for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            logger.info(f"Preprocessing {split_name} split ({len(split_idx)} samples)...")
            split_tensors = []
            for i in split_idx:
                tensor = self.preprocess(valid_records[i].wafer_map)
                split_tensors.append(tensor)
            splits[f"{split_name}_data"] = torch.stack(split_tensors)
            splits[f"{split_name}_labels"] = label_tensor[split_idx]
            logger.info(f"  {split_name}: {splits[f'{split_name}_data'].shape}")

        # Log split statistics
        logger.info(
            "Created splits - Train: %d, Val: %d, Test: %d, Classes: %d",
            len(train_idx),
            len(val_idx),
            len(test_idx),
            len(label_to_index),
        )

        return splits

    def save_splits(
        self,
        output_dir: Path,
        splits: dict[str, Any],
    ) -> None:
        """Save preprocessed splits to disk with atomic writes.

        Saves:
            - train_data.pt, train_labels.pt
            - val_data.pt, val_labels.pt
            - test_data.pt, test_labels.pt
            - label_to_index.json
            - manifest.json (metadata with counts, shapes, checksums)

        Args:
            output_dir: Directory to save preprocessed data to.
            splits: Dictionary returned by create_splits().
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        label_to_index = splits["label_to_index"]

        # Define files to save
        tensor_files = [
            ("train_data.pt", splits["train_data"]),
            ("train_labels.pt", splits["train_labels"]),
            ("val_data.pt", splits["val_data"]),
            ("val_labels.pt", splits["val_labels"]),
            ("test_data.pt", splits["test_data"]),
            ("test_labels.pt", splits["test_labels"]),
        ]

        checksums: dict[str, str] = {}

        # Save each tensor file atomically
        for filename, tensor in tensor_files:
            target_path = output_dir / filename

            def _write_tensor(path: Path, t: torch.Tensor = tensor) -> None:
                torch.save(t, path)

            def _verify_tensor(path: Path, t: torch.Tensor = tensor) -> None:
                loaded = torch.load(path, weights_only=True)
                assert loaded.shape == t.shape, (
                    f"Shape mismatch: expected {t.shape}, got {loaded.shape}"
                )

            atomic_write(target_path, _write_tensor, _verify_tensor)
            checksums[filename] = compute_md5(target_path)

        # Save label_to_index.json atomically
        label_to_index_path = output_dir / "label_to_index.json"

        def _write_label_map(path: Path) -> None:
            with open(path, "w") as f:
                json.dump(label_to_index, f, indent=2, sort_keys=True)

        def _verify_label_map(path: Path) -> None:
            with open(path, "r") as f:
                loaded = json.load(f)
            assert loaded == label_to_index, "label_to_index.json verification failed"

        atomic_write(label_to_index_path, _write_label_map, _verify_label_map)
        checksums["label_to_index.json"] = compute_md5(label_to_index_path)

        # Build manifest
        manifest = self._build_manifest(output_dir, splits, checksums)

        # Save manifest.json atomically
        manifest_path = output_dir / "manifest.json"

        def _write_manifest(path: Path) -> None:
            with open(path, "w") as f:
                json.dump(manifest, f, indent=2)

        atomic_write(manifest_path, _write_manifest)

        # Log per-split details
        for split_name in ("train", "val", "test"):
            data_key = f"{split_name}_data"
            labels_key = f"{split_name}_labels"
            data_tensor = splits[data_key]
            labels_tensor = splits[labels_key]

            data_file = output_dir / f"{split_name}_data.pt"
            labels_file = output_dir / f"{split_name}_labels.pt"
            data_size_mb = data_file.stat().st_size / (1024 * 1024)
            labels_size_mb = labels_file.stat().st_size / (1024 * 1024)

            logger.info(
                "Split '%s' - Samples: %d, Data shape: %s, Labels shape: %s, "
                "Data dtype: %s, Labels dtype: %s, "
                "Data disk size: %.2f MB, Labels disk size: %.2f MB",
                split_name,
                data_tensor.shape[0],
                list(data_tensor.shape),
                list(labels_tensor.shape),
                data_tensor.dtype,
                labels_tensor.dtype,
                data_size_mb,
                labels_size_mb,
            )

        logger.info("All splits saved to: %s", output_dir)

    def _filter_records(self, records: list[WaferRecord]) -> list[WaferRecord]:
        """Filter records, excluding unlabeled/unknown and small maps.

        Args:
            records: List of WaferRecord instances.

        Returns:
            Filtered list containing only valid, labeled records.
        """
        valid: list[WaferRecord] = []
        discarded_small = 0
        discarded_unlabeled = 0

        for record in records:
            # Exclude unlabeled and unknown records
            if record.label in ("unlabeled", "unknown"):
                discarded_unlabeled += 1
                continue

            # Exclude records with spatial dims < 5×5
            h, w = record.wafer_map.shape[:2]
            if h < MIN_SPATIAL_DIM or w < MIN_SPATIAL_DIM:
                discarded_small += 1
                logger.warning(
                    "Discarding record (index %d) with spatial dimensions "
                    "%dx%d (below minimum %dx%d).",
                    record.original_index,
                    h,
                    w,
                    MIN_SPATIAL_DIM,
                    MIN_SPATIAL_DIM,
                )
                continue

            valid.append(record)

        if discarded_small > 0:
            logger.info(
                "Filtered out %d records with spatial dims < %dx%d.",
                discarded_small,
                MIN_SPATIAL_DIM,
                MIN_SPATIAL_DIM,
            )
        if discarded_unlabeled > 0:
            logger.info(
                "Filtered out %d unlabeled/unknown records.",
                discarded_unlabeled,
            )

        return valid

    def _verify_proportions(
        self,
        all_labels: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        """Verify that per-class proportions are preserved within ±1% relative.

        Args:
            all_labels: Full array of label indices.
            train_idx: Indices for the training split.
            val_idx: Indices for the validation split.
            test_idx: Indices for the test split.

        Logs a warning if any class deviates more than 1% in any split.
        """
        total_counts = Counter(all_labels.tolist())
        total_n = len(all_labels)

        for split_name, idx in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            split_labels = all_labels[idx]
            split_counts = Counter(split_labels.tolist())
            split_n = len(split_labels)

            for cls, total_count in total_counts.items():
                expected_proportion = total_count / total_n
                actual_count = split_counts.get(cls, 0)
                actual_proportion = actual_count / split_n if split_n > 0 else 0.0

                if expected_proportion > 0:
                    relative_deviation = abs(
                        actual_proportion - expected_proportion
                    ) / expected_proportion
                    if relative_deviation > 0.01:
                        logger.warning(
                            "Class %d in '%s' split deviates %.2f%% from "
                            "expected proportion (expected: %.4f, actual: %.4f).",
                            cls,
                            split_name,
                            relative_deviation * 100,
                            expected_proportion,
                            actual_proportion,
                        )

    def _build_manifest(
        self,
        output_dir: Path,
        splits: dict[str, Any],
        checksums: dict[str, str],
    ) -> dict[str, Any]:
        """Build the manifest.json metadata dictionary.

        Args:
            output_dir: Directory where files are saved.
            splits: The splits dictionary.
            checksums: MD5 checksums for each saved file.

        Returns:
            Manifest dictionary with metadata about all saved files.
        """
        label_to_index = splits["label_to_index"]
        index_to_label = {v: k for k, v in label_to_index.items()}

        manifest: dict[str, Any] = {
            "target_size": self.target_size,
            "num_classes": len(label_to_index),
            "label_to_index": label_to_index,
            "splits": {},
            "checksums": checksums,
        }

        for split_name in ("train", "val", "test"):
            data_tensor = splits[f"{split_name}_data"]
            labels_tensor = splits[f"{split_name}_labels"]

            # Per-class counts for this split
            label_counts: dict[str, int] = {}
            for idx in labels_tensor.tolist():
                class_name = index_to_label[idx]
                label_counts[class_name] = label_counts.get(class_name, 0) + 1

            data_file = output_dir / f"{split_name}_data.pt"
            labels_file = output_dir / f"{split_name}_labels.pt"

            manifest["splits"][split_name] = {
                "num_samples": data_tensor.shape[0],
                "data_shape": list(data_tensor.shape),
                "labels_shape": list(labels_tensor.shape),
                "data_dtype": str(data_tensor.dtype),
                "labels_dtype": str(labels_tensor.dtype),
                "data_file": f"{split_name}_data.pt",
                "labels_file": f"{split_name}_labels.pt",
                "data_size_bytes": data_file.stat().st_size,
                "labels_size_bytes": labels_file.stat().st_size,
                "per_class_counts": label_counts,
            }

        return manifest
