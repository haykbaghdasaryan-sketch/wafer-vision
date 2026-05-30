"""Backbone registry and factory for model instantiation."""

from src.exceptions import UnsupportedBackboneError
from src.models.backbone import BaseBackbone
from src.registry import backbone_registry, _normalize_name


def get_backbone(
    name: str, pretrained: bool = True, l2_normalize: bool = False
) -> BaseBackbone:
    """Factory function to instantiate a backbone by name.

    Normalizes the name (lowercase, underscores) before registry lookup.

    Args:
        name: Backbone name (case-insensitive, hyphens/spaces normalized).
            Examples: "ResNet50", "efficientnet-b0", "vit_b16".
        pretrained: Whether to load ImageNet-pretrained weights.
        l2_normalize: Whether to L2-normalize embeddings in extract_embeddings().

    Returns:
        Instantiated backbone model.

    Raises:
        UnsupportedBackboneError: If name is not registered in the backbone
            registry.
    """
    normalized_name = _normalize_name(name)

    if normalized_name not in backbone_registry:
        supported = backbone_registry.list_registered()
        raise UnsupportedBackboneError(name=name, supported=supported)

    backbone: BaseBackbone = backbone_registry.create(
        normalized_name, pretrained=pretrained, l2_normalize=l2_normalize
    )
    return backbone
