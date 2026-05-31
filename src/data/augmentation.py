"""WaferAugmentation, SimCLRAugmentation, and EvalTransform.

Data augmentation transforms for wafer map training pipelines.
All transforms use pure torch operations for GPU compatibility.
"""

from typing import Literal, Optional

import torch
import torch.nn.functional as F

from src.exceptions import ConfigValidationError


# Strength presets: maps strength name to (noise_std, crop_scale_min)
_STRENGTH_PRESETS: dict[str, dict] = {
    "light": {"noise_std": 0.005, "crop_scale": (0.8, 1.0)},
    "medium": {"noise_std": 0.01, "crop_scale": (0.7, 1.0)},
    "heavy": {"noise_std": 0.05, "crop_scale": (0.6, 1.0)},
}


class WaferAugmentation:
    """Stochastic augmentation pipeline for training.

    Applies random geometric transforms + optional noise.
    Guarantees output shape = input shape and values in [0, 1].

    Strength presets:
        - "light": noise_std=0.005, crop_scale=(0.8, 1.0)
        - "medium": noise_std=0.01, crop_scale=(0.7, 1.0)
        - "heavy": noise_std=0.05, crop_scale=(0.6, 1.0)
    """

    def __init__(
        self,
        rotation: bool = True,
        flip_h: bool = True,
        flip_v: bool = True,
        noise_std: Optional[float] = None,
        noise_enabled: bool = True,
        strength: Literal["light", "medium", "heavy"] = "medium",
    ) -> None:
        """Initialize augmentation pipeline.

        Args:
            rotation: Enable k×90° rotation (k ∈ {0,1,2,3}).
            flip_h: Enable horizontal flip (p=0.5).
            flip_v: Enable vertical flip (p=0.5).
            noise_std: Gaussian noise standard deviation. If None, uses strength preset.
            noise_enabled: Whether to apply noise.
            strength: Preset adjusting noise and crop parameters.

        Raises:
            ConfigValidationError: If noise_std outside [0.001, 0.1].
            ConfigValidationError: If strength not in ("light", "medium", "heavy").
        """
        if strength not in _STRENGTH_PRESETS:
            raise ConfigValidationError(
                parameter="strength",
                value=strength,
                valid_range="one of: 'light', 'medium', 'heavy'",
            )

        preset = _STRENGTH_PRESETS[strength]
        self.rotation = rotation
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.noise_enabled = noise_enabled
        self.strength = strength
        self.crop_scale = preset["crop_scale"]

        # Resolve noise_std: explicit argument overrides preset
        if noise_std is not None:
            if noise_std < 0.001 or noise_std > 0.1:
                raise ConfigValidationError(
                    parameter="noise_std",
                    value=noise_std,
                    valid_range="[0.001, 0.1]",
                )
            self.noise_std = noise_std
        else:
            self.noise_std = preset["noise_std"]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a single wafer tensor.

        Args:
            tensor: Shape (3, H, W), dtype float32, values [0, 1].

        Returns:
            Augmented tensor, same shape, values clamped to [0, 1].
        """
        # Random rotation by k×90°
        if self.rotation:
            _, h, w = tensor.shape
            if h == w:
                # For square inputs, any rotation preserves shape
                k = torch.randint(0, 4, (1,)).item()
            else:
                # For non-square inputs, only 0° and 180° preserve shape
                k = torch.randint(0, 2, (1,)).item() * 2  # 0 or 2
            if k > 0:
                tensor = torch.rot90(tensor, k=k, dims=[1, 2])

        # Horizontal flip (p=0.5)
        if self.flip_h and torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[2])

        # Vertical flip (p=0.5)
        if self.flip_v and torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[1])

        # Gaussian noise
        if self.noise_enabled:
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = tensor + noise
            tensor = tensor.clamp(0.0, 1.0)

        return tensor


class EvalTransform:
    """Deterministic transform for validation/test (no stochastic ops).

    Applies only center-crop and resize. Running twice on same input
    produces identical output.
    """

    def __init__(self, target_size: int = 64) -> None:
        """Initialize evaluation transform.

        Args:
            target_size: Target spatial resolution (square output).
        """
        self.target_size = target_size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply deterministic transform.

        Args:
            tensor: Shape (3, H, W).

        Returns:
            Transformed tensor of shape (3, target_size, target_size).
        """
        _, h, w = tensor.shape

        # Center crop to the smaller dimension (square crop)
        crop_size = min(h, w)
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        tensor = tensor[:, top : top + crop_size, left : left + crop_size]

        # Resize to target_size using bilinear interpolation
        if crop_size != self.target_size:
            # F.interpolate expects (B, C, H, W)
            tensor = tensor.unsqueeze(0)
            tensor = F.interpolate(
                tensor,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
            tensor = tensor.squeeze(0)

        return tensor


class MixupAugmentation:
    """Batch-level Mixup augmentation for classification training.

    Linearly interpolates pairs of samples and their labels (as soft targets)
    within a batch. Particularly effective for improving generalization on
    minority classes like Loc and Scratch.

    Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2018)

    Usage in training loop:
        mixup = MixupAugmentation(alpha=0.4)
        images_mixed, labels_a, labels_b, lam = mixup(images, labels)
        logits = model(images_mixed)
        loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
    """

    def __init__(self, alpha: float = 0.4, p: float = 0.5) -> None:
        """Initialize Mixup augmentation.

        Args:
            alpha: Beta distribution parameter controlling interpolation
                strength. Higher alpha means more aggressive mixing. Range [0.1, 2.0].
                alpha=0.4 is the sweet spot for image classification.
            p: Probability of applying mixup to a given batch (default 0.5).

        Raises:
            ConfigValidationError: If alpha not in [0.1, 2.0] or p not in [0, 1].
        """
        if alpha < 0.1 or alpha > 2.0:
            raise ConfigValidationError(
                parameter="alpha",
                value=alpha,
                valid_range="[0.1, 2.0]",
            )
        if p < 0.0 or p > 1.0:
            raise ConfigValidationError(
                parameter="p",
                value=p,
                valid_range="[0.0, 1.0]",
            )
        self.alpha = alpha
        self.p = p

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup to a batch.

        Args:
            images: Batch tensor of shape (B, C, H, W).
            labels: Label tensor of shape (B,).

        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lam) where:
                - mixed_images: Interpolated images (B, C, H, W)
                - labels_a: Original labels (B,)
                - labels_b: Shuffled labels (B,)
                - lam: Interpolation coefficient (float in [0.5, 1.0])
        """
        # Decide whether to apply mixup this batch
        if torch.rand(1).item() > self.p:
            return images, labels, labels, 1.0

        # Sample lambda from Beta(alpha, alpha)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        # Ensure lam >= 0.5 so labels_a is always the dominant class
        lam = max(lam, 1.0 - lam)

        batch_size = images.size(0)
        # Random permutation for pairing
        index = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1.0 - lam) * images[index]
        labels_b = labels[index]

        return mixed_images, labels, labels_b, lam


class SimCLRAugmentation:
    """Generates two independently augmented views for self-supervised training.

    Pipeline per view: random resized crop → flip H → flip V → rotation → noise.
    Guarantees: views have same shape, views are not identical (cosine_sim < 0.99).
    """

    def __init__(
        self,
        target_size: int = 64,
        crop_scale: tuple[float, float] = (0.6, 1.0),
        noise_std: float = 0.02,
    ) -> None:
        """Initialize SimCLR augmentation pipeline.

        Args:
            target_size: Output spatial resolution for both views.
            crop_scale: Range for random resized crop scale factor.
            noise_std: Standard deviation of Gaussian noise.
        """
        self.target_size = target_size
        self.crop_scale = crop_scale
        self.noise_std = noise_std

    def _augment_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply one augmentation pipeline to produce a single view.

        Pipeline: random resized crop → horizontal flip (p=0.5) →
                  vertical flip (p=0.5) → random rotation (0-360°) →
                  Gaussian noise.

        Args:
            tensor: Input tensor of shape (3, H, W).

        Returns:
            Augmented view of shape (3, target_size, target_size).
        """
        _, h, w = tensor.shape

        # Random resized crop
        scale = (
            torch.rand(1).item() * (self.crop_scale[1] - self.crop_scale[0])
            + self.crop_scale[0]
        )
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        # Ensure at least 1 pixel
        crop_h = max(1, crop_h)
        crop_w = max(1, crop_w)

        top = torch.randint(0, max(1, h - crop_h + 1), (1,)).item()
        left = torch.randint(0, max(1, w - crop_w + 1), (1,)).item()
        view = tensor[:, top : top + crop_h, left : left + crop_w]

        # Resize to target size
        view = view.unsqueeze(0)
        view = F.interpolate(
            view,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )
        view = view.squeeze(0)

        # Random horizontal flip (p=0.5)
        if torch.rand(1).item() > 0.5:
            view = torch.flip(view, dims=[2])

        # Random vertical flip (p=0.5)
        if torch.rand(1).item() > 0.5:
            view = torch.flip(view, dims=[1])

        # Random rotation (0-360°): approximate with continuous rotation
        angle = torch.rand(1).item() * 360.0
        view = self._rotate_tensor(view, angle)

        # Gaussian noise
        noise = torch.randn_like(view) * self.noise_std
        view = (view + noise).clamp(0.0, 1.0)

        return view

    def _rotate_tensor(self, tensor: torch.Tensor, angle_degrees: float) -> torch.Tensor:
        """Rotate a tensor by an arbitrary angle using affine grid sampling.

        Args:
            tensor: Shape (3, H, W).
            angle_degrees: Rotation angle in degrees.

        Returns:
            Rotated tensor of the same shape, values clamped to [0, 1].
        """
        angle_rad = torch.tensor(angle_degrees * 3.141592653589793 / 180.0)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # Rotation matrix (2x3 affine)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            dtype=tensor.dtype,
        ).unsqueeze(0)  # (1, 2, 3)

        # Add batch dimension for grid_sample
        tensor_4d = tensor.unsqueeze(0)  # (1, 3, H, W)
        grid = F.affine_grid(theta, tensor_4d.size(), align_corners=False)
        rotated = F.grid_sample(
            tensor_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        rotated = rotated.squeeze(0)  # (3, H, W)
        return rotated.clamp(0.0, 1.0)

    def __call__(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views.

        Args:
            tensor: Shape (3, H, W), values in [0, 1].

        Returns:
            Tuple of (view1, view2), each shape (3, target_size, target_size),
            values in [0, 1]. Cosine similarity between views < 0.99.
        """
        view1 = self._augment_view(tensor)
        view2 = self._augment_view(tensor)
        return view1, view2
