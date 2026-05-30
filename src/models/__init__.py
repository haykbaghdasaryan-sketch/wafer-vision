"""Embedding engine: backbone architectures and embedding extraction."""

from src.models.backbone import BaseBackbone
from src.models.embedding_extractor import EmbeddingExtractor
from src.models.factory import get_backbone
from src.models.projection_head import ProjectionHead

# Import backbone modules to trigger registry registration
import src.models.resnet  # noqa: F401
import src.models.efficientnet  # noqa: F401
import src.models.vit  # noqa: F401

__all__ = [
    "BaseBackbone",
    "EmbeddingExtractor",
    "get_backbone",
    "ProjectionHead",
]
