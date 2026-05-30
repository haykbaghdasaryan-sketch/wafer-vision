"""ResNet50Backbone implementation.

Provides a ResNet50 feature extractor that produces 2048-dimensional
embeddings from the average pooling layer, with the classification head
removed. Supports both ImageNet-pretrained and random weight initialization.
"""

import logging

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.models.backbone import BaseBackbone
from src.registry import backbone_registry

logger = logging.getLogger(__name__)


@backbone_registry.register("resnet50")
class ResNet50Backbone(BaseBackbone):
    """ResNet50 backbone for 2048-dimensional embedding extraction.

    Uses torchvision's ResNet50 with the final fully-connected classification
    layer removed (replaced with Identity). The forward pass returns the output
    of the global average pooling layer as a (B, 2048) embedding tensor.

    Args:
        pretrained: If True, load ImageNet-pretrained weights. If False,
            initialize with random weights for ablation studies.
        l2_normalize: If True, L2-normalize output embeddings in
            extract_embeddings() to project onto the unit hypersphere.
    """

    def __init__(self, pretrained: bool = True, l2_normalize: bool = False) -> None:
        """Initialize ResNet50Backbone.

        Args:
            pretrained: Whether to load ImageNet-pretrained weights.
            l2_normalize: Whether to L2-normalize embeddings.
        """
        super().__init__(l2_normalize=l2_normalize)

        # Load ResNet50 with or without pretrained weights
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.resnet = models.resnet50(weights=weights)
            logger.info(
                "ResNet50 loaded with ImageNet-pretrained weights (IMAGENET1K_V2)"
            )
        else:
            self.resnet = models.resnet50(weights=None)
            logger.info("ResNet50 loaded with random weights")

        # Remove classification head - replace with Identity
        self.resnet.fc = nn.Identity()

        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = sum(
            p.numel() * p.element_size() for p in self.parameters()
        ) / (1024 * 1024)

        logger.info(
            "ResNet50Backbone initialized: "
            f"total_params={total_params:,}, "
            f"trainable_params={trainable_params:,}, "
            f"model_size_mb={model_size_mb:.2f}"
        )

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality (2048 for ResNet50)."""
        return 2048

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: image batch -> 2048-dimensional embedding batch.

        Args:
            x: Input tensor, shape (B, 3, H, W), dtype float32, values [0,1].
               H and W must be >= 32.

        Returns:
            Embedding tensor, shape (B, 2048), same device as input.

        Raises:
            TypeError: If x.dtype is not float32.
            ValueError: If spatial dims < 32 or wrong number of channels.
        """
        self.validate_input(x)

        # Forward through resnet (fc is Identity, so output is avgpool)
        embeddings = self.resnet(x)

        return embeddings
