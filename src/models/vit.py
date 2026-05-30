"""ViTB16Backbone implementation using torchvision ViT-B/16.

Provides a 768-dimensional embedding from the [CLS] token output
of a Vision Transformer with 16×16 patch size.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models
from torchvision.models import ViT_B_16_Weights

from src.models.backbone import BaseBackbone
from src.registry import backbone_registry

logger = logging.getLogger(__name__)

# ViT-B/16 expects 224×224 input
_VIT_INPUT_SIZE = 224


@backbone_registry.register("vit_b16")
class ViTB16Backbone(BaseBackbone):
    """Vision Transformer B/16 backbone for embedding extraction.

    Uses torchvision's ViT-B/16 model with the classification head removed.
    Produces 768-dimensional embeddings from the [CLS] token.

    The model expects 224×224 input. If input spatial dimensions differ,
    the forward pass will resize using bilinear interpolation.

    Attributes:
        embedding_dim: Always 768 for ViT-B/16.
    """

    def __init__(self, pretrained: bool = True, l2_normalize: bool = False) -> None:
        """Initialize ViT-B/16 backbone.

        Args:
            pretrained: If True, load ImageNet-pretrained weights.
                If False, initialize with random weights.
            l2_normalize: If True, L2-normalize output embeddings in
                extract_embeddings().
        """
        super().__init__(l2_normalize=l2_normalize)

        # Load ViT-B/16 model
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self._model = models.vit_b_16(weights=weights)
            logger.info("Loaded ViT-B/16 with ImageNet-pretrained weights")
        else:
            self._model = models.vit_b_16(weights=None)
            logger.info("Loaded ViT-B/16 with random weights")

        # Remove classification head - replace with identity
        self._model.heads = nn.Identity()

        # Log model statistics
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes

        logger.info(
            "ViT-B/16 backbone initialized",
            extra={
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": round(model_size_mb, 2),
                "embedding_dim": 768,
                "pretrained": pretrained,
            },
        )

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality (768 for ViT-B/16)."""
        return 768

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: image batch -> 768-dim embedding batch.

        Validates input, resizes to 224×224 if needed, then passes through
        the ViT model to get [CLS] token embeddings.

        Args:
            x: Input tensor, shape (B, 3, H, W), dtype float32, values [0,1].
               H and W must be >= 32.

        Returns:
            Embedding tensor, shape (B, 768), same device as input.

        Raises:
            TypeError: If x.dtype is not float32.
            ValueError: If spatial dims < 32 or wrong number of channels.
        """
        self.validate_input(x)

        # ViT-B/16 requires 224×224 input - resize if necessary
        h, w = x.shape[2], x.shape[3]
        if h != _VIT_INPUT_SIZE or w != _VIT_INPUT_SIZE:
            x = F.interpolate(
                x,
                size=(_VIT_INPUT_SIZE, _VIT_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # Forward through ViT - with heads=Identity(), this returns the
        # [CLS] token output directly as (B, 768)
        embeddings = self._model(x)

        return embeddings
