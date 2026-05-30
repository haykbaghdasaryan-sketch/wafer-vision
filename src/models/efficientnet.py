"""EfficientNetB0Backbone implementation.

Provides a 1280-dimensional embedding from EfficientNet-B0's adaptive
average pooling layer, with the classification head removed.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from src.models.backbone import BaseBackbone
from src.registry import backbone_registry

logger = logging.getLogger(__name__)


@backbone_registry.register("efficientnet_b0")
class EfficientNetB0Backbone(BaseBackbone):
    """EfficientNet-B0 backbone producing 1280-dimensional embeddings.

    Uses torchvision's EfficientNet-B0 architecture with the classification
    head removed. Embeddings are extracted from the adaptive average pooling
    layer following the feature extraction layers.

    Attributes:
        embedding_dim: Always 1280 for EfficientNet-B0.
    """

    def __init__(
        self, pretrained: bool = True, l2_normalize: bool = False
    ) -> None:
        """Initialize EfficientNetB0Backbone.

        Args:
            pretrained: If True, load ImageNet-pretrained weights.
                If False, initialize with random weights.
            l2_normalize: If True, L2-normalize output embeddings in
                extract_embeddings() to project onto the unit hypersphere.
        """
        super().__init__(l2_normalize=l2_normalize)

        # Load the EfficientNet-B0 model
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            base_model = models.efficientnet_b0(weights=weights)
        else:
            base_model = models.efficientnet_b0(weights=None)

        # Keep only the feature extraction layers and adaptive pooling
        self.features = base_model.features
        self.avgpool = base_model.avgpool  # AdaptiveAvgPool2d((1, 1))

        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes

        logger.info(
            "EfficientNetB0Backbone initialized",
            extra={
                "pretrained": pretrained,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": f"{model_size_mb:.2f}",
            },
        )

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality (1280 for EfficientNet-B0)."""
        return 1280

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: image batch -> 1280-dim embedding batch.

        Args:
            x: Input tensor, shape (B, 3, H, W), dtype float32, values [0,1].
               H and W must be >= 32.

        Returns:
            Embedding tensor, shape (B, 1280), same device as input.

        Raises:
            TypeError: If x.dtype is not float32.
            ValueError: If spatial dims < 32 or wrong number of channels.
        """
        self.validate_input(x)

        # Feature extraction through convolutional layers
        x = self.features(x)

        # Adaptive average pooling to (1, 1) spatial dims
        x = self.avgpool(x)

        # Flatten from (B, 1280, 1, 1) to (B, 1280)
        x = torch.flatten(x, 1)

        return x
