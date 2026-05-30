"""LR schedulers: cosine annealing with warmup, and layerwise LR decay.

Provides:
- WarmupCosineScheduler: Linear warmup followed by cosine annealing.
- LayerwiseLRDecay: Per-layer-group learning rate assignment with decay.
"""

import math
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Learning rate scheduler with linear warmup followed by cosine annealing.

    During the warmup phase (epoch < warmup_epochs), the learning rate increases
    linearly from 0 to base_lr. After warmup, the learning rate follows a cosine
    annealing schedule from base_lr down to eta_min over the remaining epochs.

    Formulas:
        Warmup (epoch < warmup_epochs):
            lr = base_lr * (epoch / warmup_epochs)
        Cosine (epoch >= warmup_epochs):
            lr = eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of epochs for the linear warmup phase.
        total_epochs: Total number of training epochs.
        eta_min: Minimum learning rate after cosine annealing. Default: 1e-6.
        last_epoch: The index of last epoch. Default: -1.

    Raises:
        ValueError: If warmup_epochs >= total_epochs or if values are negative.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=100)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {warmup_epochs}"
            )
        if total_epochs <= 0:
            raise ValueError(
                f"total_epochs must be positive, got {total_epochs}"
            )
        if warmup_epochs >= total_epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) must be less than "
                f"total_epochs ({total_epochs})"
            )
        if eta_min < 0:
            raise ValueError(f"eta_min must be non-negative, got {eta_min}")

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute current learning rate for each parameter group.

        Returns:
            List of learning rates, one per parameter group.
        """
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup: lr increases from 0 to base_lr
            if self.warmup_epochs == 0:
                scale = 1.0
            else:
                scale = epoch / self.warmup_epochs
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing: lr decreases from base_lr to eta_min
            cosine_epochs = self.total_epochs - self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / cosine_epochs
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]


class LayerwiseLRDecay:
    """Assigns decreasing learning rates to parameter groups from top to bottom.

    Top layers (closer to the output) receive higher learning rates, while lower
    layers (closer to the input) receive progressively smaller rates. This
    follows the principle that pretrained lower layers need less adaptation.

    The decay formula is:
        layer_lr = base_lr * decay_factor^depth

    where depth=0 is the topmost (output-closest) layer group and depth increases
    going deeper into the network.

    Args:
        decay_factor: Multiplicative factor applied per layer depth.
            Default: 0.9. Must be in (0, 1].

    Example:
        >>> param_groups = LayerwiseLRDecay.create_param_groups(
        ...     model, base_lr=1e-3, decay_factor=0.9
        ... )
        >>> optimizer = torch.optim.Adam(param_groups)
    """

    def __init__(self, optimizer: Optimizer, model: nn.Module, decay_factor: float = 0.9) -> None:
        """Initialize LayerwiseLRDecay and apply decay to optimizer param groups.

        This re-assigns learning rates to optimizer parameter groups based on
        layer depth. The optimizer must already have been created with the
        parameter groups from create_param_groups().

        Args:
            optimizer: Optimizer with layer-wise parameter groups.
            model: The model whose parameters are being optimized.
            decay_factor: Multiplicative decay per layer depth. Default: 0.9.

        Raises:
            ValueError: If decay_factor is not in (0, 1].
        """
        if decay_factor <= 0 or decay_factor > 1.0:
            raise ValueError(
                f"decay_factor must be in (0, 1], got {decay_factor}"
            )

        self.optimizer = optimizer
        self.model = model
        self.decay_factor = decay_factor

        # Apply layer-wise LR decay to existing parameter groups
        n_groups = len(optimizer.param_groups)
        for i, group in enumerate(optimizer.param_groups):
            # depth 0 = topmost (last group added), depth increases towards input
            depth = n_groups - 1 - i
            group["lr"] = group["lr"] * (decay_factor ** depth)

    @staticmethod
    def create_param_groups(
        model: nn.Module,
        base_lr: float,
        decay_factor: float = 0.9,
    ) -> list[dict[str, Any]]:
        """Create optimizer parameter groups with layer-wise learning rate decay.

        Traverses the model's named children (top-level modules) and assigns
        each group a learning rate that decreases by decay_factor for each
        layer depth from the output towards the input.

        The ordering is: later layers (closer to output) get higher LR.
        Layer groups are determined by the model's top-level children.

        Args:
            model: Neural network model with named_children.
            base_lr: Base learning rate for the topmost layer.
            decay_factor: Multiplicative decay per layer depth. Default: 0.9.

        Returns:
            List of parameter group dicts suitable for an optimizer constructor.
            Each dict has 'params' (list of Parameters) and 'lr' (float).

        Raises:
            ValueError: If decay_factor is not in (0, 1] or base_lr is not positive.

        Example:
            >>> groups = LayerwiseLRDecay.create_param_groups(model, base_lr=1e-3)
            >>> optimizer = torch.optim.Adam(groups)
        """
        if decay_factor <= 0 or decay_factor > 1.0:
            raise ValueError(
                f"decay_factor must be in (0, 1], got {decay_factor}"
            )
        if base_lr <= 0:
            raise ValueError(f"base_lr must be positive, got {base_lr}")

        # Collect layer groups from model's named children
        layer_groups: list[tuple[str, list[nn.Parameter]]] = []

        for name, child in model.named_children():
            params = list(child.parameters())
            if params:  # Skip modules with no parameters
                layer_groups.append((name, params))

        # If model has no children with parameters, use all parameters as one group
        if not layer_groups:
            all_params = list(model.parameters())
            if all_params:
                return [{"params": all_params, "lr": base_lr}]
            return []

        n_layers = len(layer_groups)

        # Assign LR: top layers (last in order) get base_lr * decay^0,
        # lower layers get base_lr * decay^depth
        param_groups: list[dict[str, Any]] = []
        for i, (name, params) in enumerate(layer_groups):
            # depth: distance from the top (output end)
            # layer_groups[0] is deepest (input-closest), layer_groups[-1] is topmost
            depth = n_layers - 1 - i
            lr = base_lr * (decay_factor ** depth)
            param_groups.append({
                "params": params,
                "lr": lr,
                "name": name,
            })

        return param_groups
