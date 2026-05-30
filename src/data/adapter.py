"""DatasetAdapter ABC for extensibility.

Provides the abstract base class that all dataset loaders must implement,
ensuring a consistent interface for loading and summarizing datasets.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DatasetAdapter(ABC):
    """Abstract base class for dataset loaders.

    All dataset loaders must implement `load()` and `get_summary()`.
    This enables extensibility: new datasets can be added by subclassing
    DatasetAdapter and registering with the dataset_registry.
    """

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Load dataset from the given path.

        Args:
            path: Path to the dataset file or directory.

        Returns:
            Loaded dataset in a format specific to the adapter implementation.
            Typically a list of parsed records or a structured container.

        Raises:
            DataFileNotFoundError: If the path does not exist.
            DataCorruptionError: If the file cannot be deserialized safely.
            MemoryError: If insufficient system memory is available.
        """
        ...

    @abstractmethod
    def get_summary(self) -> dict[str, Any]:
        """Generate a summary report of the loaded dataset.

        Must be called after a successful `load()` invocation.

        Returns:
            Dictionary containing dataset statistics including:
            - total_count: Total number of records loaded.
            - labeled_count: Number of records with valid defect labels.
            - unlabeled_count: Number of records without a valid label.
            - per_class_counts: Dict mapping class name to sample count.
            - per_class_percentages: Dict mapping class name to percentage.
            - min_spatial_dims: Tuple (min_height, min_width).
            - max_spatial_dims: Tuple (max_height, max_width).
            - mean_spatial_dims: Tuple (mean_height, mean_width).
            - lot_count: Number of unique manufacturing lots.

        Raises:
            RuntimeError: If called before load() has been successfully invoked.
        """
        ...
