"""WaferMapDataset (PyTorch Dataset) for preprocessed splits.

Loads preprocessed .pt tensor files and applies configurable transforms
lazily at __getitem__ time to maximize effective training diversity.
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torch.utils.data import Dataset

from src.exceptions import DataFileNotFoundError

logger = logging.getLogger(__name__)


class WaferMapDataset(Dataset):
    """PyTorch Dataset for preprocessed wafer map splits.

    Loads data and labels from .pt files and applies optional transforms
    lazily at access time. This ensures augmentation diversity across
    epochs and avoids pre-computing augmented views.

    Usage:
        dataset = WaferMapDataset(
            data_path=Path("data/processed/train_data.pt"),
            labels_path=Path("data/processed/train_labels.pt"),
            transform=WaferAugmentation(strength="medium"),
        )
        tensor, label = dataset[0]

    Or using the factory class method:
        dataset = WaferMapDataset.from_split(
            split_dir=Path("data/processed"),
            split_name="train",
            transform=WaferAugmentation(strength="medium"),
        )
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        labels_path: Union[str, Path],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Initialize the dataset by loading .pt files into memory.

        Args:
            data_path: Path to the data tensor .pt file (shape: N, 3, H, W).
            labels_path: Path to the labels tensor .pt file (shape: N,).
            transform: Optional callable applied to each sample tensor at
                access time. Should accept and return a (3, H, W) tensor.

        Raises:
            DataFileNotFoundError: If data_path or labels_path does not exist.
            ValueError: If data and labels have mismatched sample counts.
        """
        data_path = Path(data_path)
        labels_path = Path(labels_path)

        if not data_path.exists():
            raise DataFileNotFoundError(str(data_path))
        if not labels_path.exists():
            raise DataFileNotFoundError(str(labels_path))

        self.data: torch.Tensor = torch.load(data_path, weights_only=True)
        self.labels: torch.Tensor = torch.load(labels_path, weights_only=True)

        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Data and labels have mismatched sample counts: "
                f"data has {self.data.shape[0]}, labels has {self.labels.shape[0]}"
            )

        self.transform = transform

        logger.info(
            "Loaded WaferMapDataset: %d samples, data shape %s, labels shape %s",
            len(self.data),
            list(self.data.shape),
            list(self.labels.shape),
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample with optional transform applied.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (tensor, label) where tensor is shape (3, H, W) with
            transform applied if configured, and label is a scalar tensor.
        """
        tensor = self.data[idx]
        label = self.labels[idx]

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

    @classmethod
    def from_split(
        cls,
        split_dir: Union[str, Path],
        split_name: str = "train",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> "WaferMapDataset":
        """Create a dataset from a preprocessed split directory.

        Loads {split_name}_data.pt and {split_name}_labels.pt from split_dir.

        Args:
            split_dir: Directory containing preprocessed .pt files.
            split_name: Name of the split ("train", "val", or "test").
            transform: Optional callable applied to each sample tensor.

        Returns:
            WaferMapDataset instance loaded from the specified split.

        Raises:
            DataFileNotFoundError: If the expected .pt files don't exist.
        """
        split_dir = Path(split_dir)
        data_path = split_dir / f"{split_name}_data.pt"
        labels_path = split_dir / f"{split_name}_labels.pt"

        return cls(
            data_path=data_path,
            labels_path=labels_path,
            transform=transform,
        )
