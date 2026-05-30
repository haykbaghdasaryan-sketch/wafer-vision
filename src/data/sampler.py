"""BalancedBatchSampler for metric learning.

Constructs mini-batches with exactly P randomly selected classes and K randomly
selected samples per class, yielding batch size = P × K. Essential for metric
learning loss functions (Triplet, SupCon) that require multiple samples per class.
"""

import logging
import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler

from src.exceptions import ConfigValidationError

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler[list[int]]):
    """Constructs batches with exactly P classes × K samples per class.

    If a class has fewer than K samples, oversamples with replacement
    and logs the oversampling ratio.

    Invariants:
        - Each batch has exactly P distinct classes
        - Each class in a batch has exactly K samples
        - Batch size = P × K

    Args:
        labels: List of integer class labels for all samples.
        p_classes: Number of classes per batch (default 9, range [2, 9]).
        k_samples: Samples per class per batch (default 4, range [2, 16]).

    Raises:
        ConfigValidationError: If p_classes > num_unique_classes or parameters
            are out of valid range.
    """

    def __init__(
        self,
        labels: list[int],
        p_classes: int = 9,
        k_samples: int = 4,
    ) -> None:
        super().__init__()

        # Validate parameters
        if not (2 <= p_classes <= 9):
            raise ConfigValidationError(
                parameter="p_classes",
                value=p_classes,
                valid_range="[2, 9]",
            )
        if not (2 <= k_samples <= 16):
            raise ConfigValidationError(
                parameter="k_samples",
                value=k_samples,
                valid_range="[2, 16]",
            )

        self.labels = labels
        self.p_classes = p_classes
        self.k_samples = k_samples

        # Build index mapping: class_label -> list of sample indices
        self._class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self._class_to_indices[label].append(idx)

        self._classes = list(self._class_to_indices.keys())
        num_unique_classes = len(self._classes)

        if p_classes > num_unique_classes:
            raise ConfigValidationError(
                parameter="p_classes",
                value=p_classes,
                valid_range=f"[2, {num_unique_classes}] (dataset has {num_unique_classes} classes)",
            )

        # Compute number of batches per epoch
        # Each epoch uses all samples at least once: total samples / batch_size
        total_samples = len(labels)
        self._batch_size = p_classes * k_samples
        self._num_batches = total_samples // self._batch_size

        # Ensure at least one batch
        if self._num_batches == 0:
            self._num_batches = 1

        # Log oversampling info for classes with fewer than K samples
        for cls, indices in self._class_to_indices.items():
            if len(indices) < k_samples:
                ratio = k_samples / len(indices)
                logger.info(
                    "Class %d has %d samples < K=%d, will oversample with "
                    "replacement (ratio: %.2fx)",
                    cls,
                    len(indices),
                    k_samples,
                    ratio,
                )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batch index lists of size P×K.

        Each batch contains exactly P randomly selected classes with K randomly
        selected samples per class. Classes and samples are shuffled each epoch.
        If a class has fewer than K samples, samples are drawn with replacement.
        """
        # Shuffle class order for this epoch
        available_classes = self._classes.copy()

        for _ in range(self._num_batches):
            # Select P random classes for this batch
            random.shuffle(available_classes)
            selected_classes = available_classes[: self.p_classes]

            batch_indices: list[int] = []
            for cls in selected_classes:
                class_indices = self._class_to_indices[cls]

                if len(class_indices) >= self.k_samples:
                    # Sample without replacement
                    sampled = random.sample(class_indices, self.k_samples)
                else:
                    # Oversample with replacement
                    sampled = random.choices(class_indices, k=self.k_samples)
                    ratio = self.k_samples / len(class_indices)
                    logger.debug(
                        "Oversampling class %d: %d samples -> %d required "
                        "(ratio: %.2fx)",
                        cls,
                        len(class_indices),
                        self.k_samples,
                        ratio,
                    )

                batch_indices.extend(sampled)

            yield batch_indices

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self._num_batches
