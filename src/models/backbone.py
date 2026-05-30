"""BaseBackbone ABC for all backbone architectures."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


class BaseBackbone(ABC, nn.Module):
    """Unified interface for all backbone architectures.

    Contract:
        - forward(x) accepts (B, 3, H, W) where H,W >= 32, dtype float32
        - forward(x) returns (B, D) where D = self.embedding_dim
        - Output device matches input device
        - Supports batch sizes 1 to 256

    Subclasses must implement forward() and embedding_dim property.
    """

    def __init__(self, l2_normalize: bool = False) -> None:
        """Initialize BaseBackbone.

        Args:
            l2_normalize: If True, L2-normalize output embeddings in
                extract_embeddings() to project onto the unit hypersphere.
        """
        super().__init__()
        self._l2_normalize = l2_normalize

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: image batch -> embedding batch.

        Args:
            x: Input tensor, shape (B, 3, H, W), dtype float32, values [0,1].
               H and W must be >= 32.

        Returns:
            Embedding tensor, shape (B, D), same device as input.

        Raises:
            TypeError: If x.dtype is not float32.
            ValueError: If spatial dims < 32 or wrong number of channels.
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality D."""
        ...

    def validate_input(self, x: Tensor) -> None:
        """Validate input tensor meets backbone requirements.

        Args:
            x: Input to validate.

        Raises:
            TypeError: If dtype != float32.
            ValueError: If shape invalid (not 4D, channels != 3, spatial < 32).
        """
        if x.dtype != torch.float32:
            raise TypeError(
                f"Expected input dtype torch.float32, got {x.dtype}. "
                f"Convert with tensor.float() before passing to backbone."
            )

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, 3, H, W), got {x.ndim}D "
                f"tensor with shape {tuple(x.shape)}."
            )

        if x.shape[1] != 3:
            raise ValueError(
                f"Expected 3 input channels (B, 3, H, W), got {x.shape[1]} "
                f"channels with shape {tuple(x.shape)}."
            )

        h, w = x.shape[2], x.shape[3]
        if h < 32 or w < 32:
            raise ValueError(
                f"Expected spatial dimensions >= 32x32, got {h}x{w}. "
                f"Resize input to at least 32x32 pixels."
            )

    def extract_embeddings(
        self,
        dataloader: DataLoader,
        device: str = "cuda",
        normalize: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings for an entire dataset.

        Runs in eval mode with torch.no_grad() for determinism and memory
        efficiency.

        Args:
            dataloader: DataLoader yielding (tensor, label) batches.
            device: Target device ("cuda" or "cpu").
            normalize: If True, L2-normalize output embeddings. If None,
                uses the instance's l2_normalize setting.

        Returns:
            Tuple of (embeddings (N, D) float32, labels (N,) int64).
        """
        should_normalize = normalize if normalize is not None else self._l2_normalize

        self.eval()
        self.to(device)

        all_embeddings: list[Tensor] = []
        all_labels: list[Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch[0], batch[1]
                images = images.to(device)

                embeddings = self.forward(images)

                if should_normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)

        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        return embeddings_tensor.numpy(), labels_tensor.numpy()
