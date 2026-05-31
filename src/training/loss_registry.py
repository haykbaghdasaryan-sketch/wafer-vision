"""Loss factory + registry for extensible loss function lookup.

Registers all loss functions with the global loss_registry from src.registry.
Import this module to ensure losses are registered and accessible via
loss_registry.create("name", **kwargs).
"""

from src.registry import loss_registry
from src.training.losses import (
    FocalLoss,
    NTXentLoss,
    SupervisedContrastiveLoss,
    TripletMarginLossWithMining,
    WeightedCrossEntropyLoss,
)

# Register all loss functions with the global registry
loss_registry.register("triplet")(TripletMarginLossWithMining)
loss_registry.register("supcon")(SupervisedContrastiveLoss)
loss_registry.register("ntxent")(NTXentLoss)
loss_registry.register("crossentropy")(WeightedCrossEntropyLoss)
loss_registry.register("focal")(FocalLoss)

__all__ = [
    "FocalLoss",
    "TripletMarginLossWithMining",
    "SupervisedContrastiveLoss",
    "NTXentLoss",
    "WeightedCrossEntropyLoss",
]
