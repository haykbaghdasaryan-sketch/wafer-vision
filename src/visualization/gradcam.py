"""GradCAMVisualizer: CNN + ViT unified explainability.

Provides a unified interface for generating spatial attention heatmaps
using Grad-CAM (for CNN backbones) and Attention Rollout (for ViT backbones).
Supports single-image and batch computation, overlay generation, and
guided Grad-CAM for higher-resolution explanations.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.backbone import BaseBackbone

logger = logging.getLogger(__name__)

# Maximum batch size for batch computation
_MAX_BATCH_SIZE = 16

# Uniform heatmap value for all-zeros edge case
_UNIFORM_HEATMAP_VALUE = 0.5


def _is_vit_backbone(model: BaseBackbone) -> bool:
    """Check if a model is a ViT backbone by checking for encoder attribute."""
    # Check for torchvision ViT structure: model._model.encoder
    if hasattr(model, "_model") and hasattr(model._model, "encoder"):
        return True
    return False


def _get_default_target_layer_cnn(model: BaseBackbone) -> str:
    """Auto-detect the last convolutional layer for CNN backbones.

    For ResNet50: 'resnet.layer4' (last residual block)
    For EfficientNet-B0: 'features' (last feature block)

    Returns:
        String name of the target layer attribute path.
    """
    # ResNet50: has model.resnet.layer4
    if hasattr(model, "resnet"):
        return "resnet.layer4"
    # EfficientNet-B0: has model.features (last block is features[-1])
    if hasattr(model, "features"):
        return "features"
    raise ValueError(
        f"Cannot auto-detect target layer for model type {type(model).__name__}. "
        "Please specify target_layer explicitly."
    )


def _resolve_layer(model: BaseBackbone, layer_path: str) -> torch.nn.Module:
    """Resolve a dot-separated layer path to the actual module.

    Args:
        model: The backbone model.
        layer_path: Dot-separated path like 'resnet.layer4'.

    Returns:
        The resolved nn.Module.
    """
    module = model
    for attr in layer_path.split("."):
        if attr.isdigit():
            module = module[int(attr)]
        else:
            module = getattr(module, attr)
    return module


class GradCAMVisualizer:
    """Unified explainability interface for CNN and ViT backbones.

    Automatically selects Grad-CAM for CNN models (ResNet50, EfficientNet-B0)
    or Attention Rollout for ViT models. Provides heatmap computation,
    overlay generation, and batch processing capabilities.

    Attributes:
        model: The backbone model to explain.
        is_vit: Whether the model is a Vision Transformer.
        target_layer: The target layer for Grad-CAM (CNN only).

    Example:
        >>> visualizer = GradCAMVisualizer(model)
        >>> heatmap = visualizer.compute_explanation(image)
        >>> overlay = GradCAMVisualizer.create_overlay(wafer_map, heatmap)
    """

    def __init__(
        self,
        model: BaseBackbone,
        target_layer: Optional[str] = None,
    ) -> None:
        """Initialize GradCAMVisualizer.

        Auto-detects target layer if not specified:
        - CNN (ResNet50): last residual block (layer4)
        - CNN (EfficientNet-B0): last feature block (features[-1])
        - ViT: uses attention rollout (no target layer needed)

        Args:
            model: A BaseBackbone instance (ResNet50, EfficientNet-B0, or ViT-B/16).
            target_layer: Optional explicit target layer path for Grad-CAM.
                Ignored for ViT models. If None, auto-detects for CNN.
        """
        self.model = model
        self.is_vit = _is_vit_backbone(model)
        self._target_layer_path = target_layer

        if not self.is_vit:
            # Resolve target layer for CNN
            if target_layer is None:
                self._target_layer_path = _get_default_target_layer_cnn(model)
            self._target_module = _resolve_layer(model, self._target_layer_path)
            logger.info(
                f"GradCAMVisualizer initialized for CNN backbone "
                f"({type(model).__name__}), target_layer={self._target_layer_path}"
            )
        else:
            self._target_module = None
            logger.info(
                f"GradCAMVisualizer initialized for ViT backbone "
                f"({type(model).__name__}), using Attention Rollout"
            )

    def compute_explanation(
        self,
        image: Tensor,
        target_class: Optional[int] = None,
        guided: bool = False,
    ) -> np.ndarray:
        """Compute a spatial attention heatmap for a single image.

        Auto-selects Grad-CAM for CNN or Attention Rollout for ViT.

        Args:
            image: Input tensor of shape (1, 3, H, W) or (3, H, W), dtype float32.
            target_class: Target class index for Grad-CAM. If None, uses the
                class with highest activation. Ignored for ViT (Attention Rollout).
            guided: If True, compute guided Grad-CAM (CNN only). Combines
                Grad-CAM with guided backpropagation for sharper explanations.

        Returns:
            Heatmap of shape (H, W) with values in [0.0, 1.0].
            If heatmap is all zeros, returns uniform 0.5 with a warning.
        """
        # Ensure 4D input
        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.shape[0] != 1:
            raise ValueError(
                f"compute_explanation expects a single image (batch size 1), "
                f"got batch size {image.shape[0]}. Use compute_batch() for multiple images."
            )

        start_time = time.perf_counter()

        if self.is_vit:
            heatmap = self._attention_rollout(image)
        else:
            if guided:
                heatmap = self._guided_gradcam(image, target_class)
            else:
                heatmap = self._gradcam(image, target_class)

        # Handle all-zeros heatmap
        heatmap = self._handle_zero_heatmap(heatmap)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Heatmap computed in {elapsed_ms:.1f}ms")

        return heatmap

    def compute_batch(
        self,
        images: Tensor,
        target_classes: Optional[list[int]] = None,
    ) -> list[np.ndarray]:
        """Compute heatmaps for a batch of images (up to 16).

        Args:
            images: Input tensor of shape (B, 3, H, W), dtype float32.
                B must be <= 16.
            target_classes: Optional list of target class indices, one per image.
                If None, uses highest-activation class for each image.

        Returns:
            List of B heatmaps, each of shape (H, W) with values in [0.0, 1.0].

        Raises:
            ValueError: If batch size exceeds 16.
        """
        if images.ndim != 4:
            raise ValueError(
                f"Expected 4D input (B, 3, H, W), got shape {tuple(images.shape)}"
            )

        batch_size = images.shape[0]
        if batch_size > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {batch_size} exceeds maximum of {_MAX_BATCH_SIZE}. "
                f"Process in smaller chunks."
            )

        if target_classes is not None and len(target_classes) != batch_size:
            raise ValueError(
                f"target_classes length ({len(target_classes)}) must match "
                f"batch size ({batch_size})"
            )

        start_time = time.perf_counter()
        heatmaps = []

        for i in range(batch_size):
            single_image = images[i:i + 1]
            target = target_classes[i] if target_classes is not None else None
            heatmap = self.compute_explanation(single_image, target_class=target)
            heatmaps.append(heatmap)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Batch heatmap computation: {batch_size} images in {elapsed_ms:.1f}ms "
            f"({elapsed_ms / batch_size:.1f}ms per image)"
        )

        return heatmaps

    @staticmethod
    def create_overlay(
        wafer_map: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Blend a colorized wafer map with a heatmap overlay.

        The wafer map is colorized: background=black, normal=green, defect=red.
        The heatmap uses the jet colormap.

        Args:
            wafer_map: Original wafer map, shape (H, W), values in {0, 1, 2}.
            heatmap: Attention heatmap, shape (H, W), values in [0.0, 1.0].
            alpha: Blending factor (0 = wafer only, 1 = heatmap only).
                Default 0.5. Range [0.1, 0.9].

        Returns:
            RGB overlay image, shape (H, W, 3), dtype uint8, values [0, 255].
        """
        alpha = float(np.clip(alpha, 0.1, 0.9))

        h, w = wafer_map.shape[:2]

        # Resize heatmap to match wafer map dimensions if needed
        if heatmap.shape != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Colorize wafer map: 0=black, 1=green, 2=red
        wafer_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        wafer_rgb[wafer_map == 1] = [0, 200, 0]  # Green for normal
        wafer_rgb[wafer_map == 2] = [200, 0, 0]  # Red for defect

        # Apply jet colormap to heatmap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        # cv2 uses BGR, convert to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(
            wafer_rgb, 1.0 - alpha, heatmap_colored, alpha, 0
        )

        return overlay

    def _gradcam(self, image: Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Compute Grad-CAM for CNN backbones.

        Uses hook-based gradient computation on the target layer.

        Args:
            image: Input tensor (1, 3, H, W).
            target_class: Target class for gradient computation.

        Returns:
            Heatmap of shape (H, W), values in [0, 1].
        """
        device = next(self.model.parameters()).device
        image = image.to(device)

        # Storage for activations and gradients
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        # Register hooks
        handle_fwd = self._target_module.register_forward_hook(forward_hook)
        handle_bwd = self._target_module.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            self.model.eval()
            image.requires_grad_(True)
            output = self.model(image)  # (1, D)

            # For embedding models without classification head, use the
            # target_class-th dimension of the output embedding as the scalar
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Create a scalar target by selecting the target_class-th neuron
            if target_class < output.shape[1]:
                score = output[0, target_class]
            else:
                # Fallback: use the neuron with highest activation
                score = output[0, output.argmax(dim=1).item()]

            # Backward pass
            self.model.zero_grad()
            score.backward()

            # Compute Grad-CAM
            if not activations or not gradients:
                logger.warning("No activations/gradients captured. Returning zeros.")
                h, w = image.shape[2], image.shape[3]
                return np.zeros((h, w), dtype=np.float32)

            activation = activations[0]  # (1, C, h', w')
            gradient = gradients[0]  # (1, C, h', w')

            # Global average pooling of gradients -> channel weights
            weights = gradient.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of activation maps
            cam = (weights * activation).sum(dim=1, keepdim=True)  # (1, 1, h', w')

            # ReLU - only keep positive contributions
            cam = F.relu(cam)

            # Resize to input spatial dimensions
            h, w = image.shape[2], image.shape[3]
            cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

            # Normalize to [0, 1]
            cam = cam.squeeze().cpu().numpy()
            cam = self._normalize_heatmap(cam)

            return cam

        finally:
            handle_fwd.remove()
            handle_bwd.remove()

    def _guided_gradcam(
        self, image: Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """Compute guided Grad-CAM for CNN backbones.

        Combines Grad-CAM with guided backpropagation for sharper explanations.
        Only positive gradients are propagated through ReLU layers.

        Args:
            image: Input tensor (1, 3, H, W).
            target_class: Target class for gradient computation.

        Returns:
            Heatmap of shape (H, W), values in [0, 1].
        """
        device = next(self.model.parameters()).device
        image = image.to(device).clone().detach().requires_grad_(True)

        # Step 1: Compute standard Grad-CAM
        gradcam_map = self._gradcam(image.detach(), target_class)

        # Step 2: Compute guided backpropagation
        guided_grads = self._guided_backprop(image, target_class)

        # Step 3: Element-wise product
        # Resize Grad-CAM to match input spatial dims if needed
        h, w = image.shape[2], image.shape[3]
        gradcam_resized = cv2.resize(
            gradcam_map, (w, h), interpolation=cv2.INTER_LINEAR
        )

        # guided_grads is (3, H, W) - take max across channels for spatial map
        guided_spatial = np.max(guided_grads, axis=0)  # (H, W)

        # Element-wise multiplication
        guided_gradcam = gradcam_resized * guided_spatial

        # Normalize to [0, 1]
        guided_gradcam = self._normalize_heatmap(guided_gradcam)

        return guided_gradcam

    def _guided_backprop(
        self, image: Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """Compute guided backpropagation gradients.

        Modifies ReLU layers to disable inplace operations and only
        propagate positive gradients during the backward pass.

        Args:
            image: Input tensor (1, 3, H, W), requires_grad=True.
            target_class: Target class for gradient computation.

        Returns:
            Guided gradients of shape (3, H, W).
        """
        device = next(self.model.parameters()).device
        image = image.to(device)
        if not image.requires_grad:
            image = image.clone().detach().requires_grad_(True)

        # Disable inplace ReLU operations to avoid autograd conflicts
        original_inplace = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU) and module.inplace:
                original_inplace[name] = True
                module.inplace = False

        # Register hooks on ReLU layers to clip negative gradients
        relu_hooks = []

        def guided_relu_hook(module, grad_input, grad_output):
            return (torch.clamp(grad_output[0], min=0.0),)

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                hook = module.register_full_backward_hook(guided_relu_hook)
                relu_hooks.append(hook)

        try:
            self.model.eval()
            output = self.model(image)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            if target_class < output.shape[1]:
                score = output[0, target_class]
            else:
                score = output[0, output.argmax(dim=1).item()]

            self.model.zero_grad()
            score.backward()

            # Get input gradients
            guided_grads = image.grad.data.squeeze(0).cpu().numpy()  # (3, H, W)

            # Clip negative values
            guided_grads = np.maximum(guided_grads, 0)

            return guided_grads

        finally:
            for hook in relu_hooks:
                hook.remove()
            # Restore inplace settings
            for name, module in self.model.named_modules():
                if name in original_inplace:
                    module.inplace = True

    def _attention_rollout(self, image: Tensor) -> np.ndarray:
        """Compute Attention Rollout for ViT backbones.

        Multiplies attention matrices across all transformer layers,
        averages across heads, and reshapes to spatial dimensions.

        Args:
            image: Input tensor (1, 3, H, W).

        Returns:
            Heatmap of shape (H, W), values in [0, 1].
        """
        device = next(self.model.parameters()).device
        image = image.to(device)

        # Get the ViT model internals
        vit_model = self.model._model

        # Extract attention weights from all encoder layers
        attention_maps = self._extract_attention_weights(image, vit_model)

        # Attention Rollout: multiply attention matrices across layers
        if not attention_maps:
            logger.warning("No attention maps captured for ViT. Returning zeros.")
            h, w = image.shape[2], image.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        rollout = self._compute_rollout(attention_maps)

        # Extract CLS token attention to all patches (first row, skip CLS itself)
        # rollout shape: (num_tokens, num_tokens) where tokens = 1 + num_patches
        cls_attention = rollout[0, 1:]  # Attention from CLS to all patches

        # Reshape to spatial grid
        # ViT-B/16 with 224x224 input -> 14x14 patches
        num_patches = cls_attention.shape[0]
        grid_size = int(np.sqrt(num_patches))

        if grid_size * grid_size != num_patches:
            # Non-square patch grid - try to infer dimensions
            logger.warning(
                f"Non-square patch count {num_patches}, using closest square root"
            )
            grid_size = int(np.ceil(np.sqrt(num_patches)))
            # Pad if needed
            padded = np.zeros(grid_size * grid_size)
            padded[:num_patches] = cls_attention
            cls_attention = padded

        spatial_map = cls_attention.reshape(grid_size, grid_size)

        # Upsample to original image dimensions
        h, w = image.shape[2], image.shape[3]
        spatial_map = cv2.resize(
            spatial_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1]
        spatial_map = self._normalize_heatmap(spatial_map)

        return spatial_map

    def _extract_attention_weights(
        self, image: Tensor, vit_model: torch.nn.Module
    ) -> list[np.ndarray]:
        """Extract attention weight matrices from all ViT encoder layers.

        Uses forward hooks on MultiheadAttention to capture the input
        and manually computes attention weights as softmax(Q*K^T / sqrt(d)).

        For torchvision ViT, self_attention is called with batch_first=True,
        so inputs are (batch, seq_len, embed_dim).

        Args:
            image: Input tensor (1, 3, H, W).
            vit_model: The internal torchvision ViT model.

        Returns:
            List of attention matrices, each shape (num_tokens, num_tokens).
        """
        attention_maps = []
        hooks = []

        def make_attn_hook():
            def hook_fn(module, args, output):
                # torchvision ViT calls: self_attention(x, x, x, need_weights=False)
                # args = (query, key, value) with shape (batch, seq_len, embed_dim)
                # since batch_first=True
                query = args[0]
                key = args[1]

                embed_dim = module.embed_dim
                num_heads = module.num_heads
                head_dim = embed_dim // num_heads

                # Project query and key using in_proj_weight
                # in_proj_weight has shape (3*embed_dim, embed_dim)
                w = module.in_proj_weight
                b = module.in_proj_bias

                # Q projection
                w_q = w[:embed_dim]
                b_q = b[:embed_dim] if b is not None else None
                q = F.linear(query, w_q, b_q)

                # K projection
                w_k = w[embed_dim:2 * embed_dim]
                b_k = b[embed_dim:2 * embed_dim] if b is not None else None
                k = F.linear(key, w_k, b_k)

                # Input shape: (batch, seq_len, embed_dim)
                batch_size = q.shape[0]
                seq_len = q.shape[1]

                # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
                q = q.view(batch_size, seq_len, num_heads, head_dim)
                k = k.view(batch_size, seq_len, num_heads, head_dim)

                # Transpose to (batch, num_heads, seq_len, head_dim)
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)

                # Compute attention: (batch, num_heads, seq_len, seq_len)
                scale = head_dim ** 0.5
                attn = torch.matmul(q, k.transpose(-2, -1)) / scale
                attn = torch.softmax(attn, dim=-1)

                # Average over heads: (batch, seq_len, seq_len)
                attn_avg = attn.mean(dim=1)

                # Take first batch element
                attn_np = attn_avg[0].detach().cpu().numpy()
                attention_maps.append(attn_np)

            return hook_fn

        # Register hooks on all self_attention layers
        for encoder_layer in vit_model.encoder.layers:
            sa = encoder_layer.self_attention
            hook = sa.register_forward_hook(make_attn_hook())
            hooks.append(hook)

        try:
            # Forward pass to trigger hooks
            self.model.eval()
            with torch.no_grad():
                _ = self.model(image)
        finally:
            for h in hooks:
                h.remove()

        return attention_maps

    def _compute_rollout(self, attention_maps: list[np.ndarray]) -> np.ndarray:
        """Compute attention rollout by multiplying attention matrices.

        Adds identity matrix to each attention layer (residual connections)
        and normalizes rows before multiplication.

        Args:
            attention_maps: List of attention matrices, each (num_tokens, num_tokens).

        Returns:
            Rolled-out attention matrix (num_tokens, num_tokens).
        """
        num_tokens = attention_maps[0].shape[0]
        rollout = np.eye(num_tokens, dtype=np.float32)

        for attn in attention_maps:
            # Add identity for residual connection
            attn_with_residual = attn + np.eye(num_tokens, dtype=np.float32)

            # Normalize rows to sum to 1
            row_sums = attn_with_residual.sum(axis=-1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-8)  # Avoid division by zero
            attn_normalized = attn_with_residual / row_sums

            # Multiply with accumulated rollout
            rollout = rollout @ attn_normalized

        return rollout

    def _handle_zero_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Handle all-zeros heatmap edge case.

        If the heatmap contains all zeros (no gradient signal), logs a warning
        and returns a uniform heatmap with value 0.5.

        Args:
            heatmap: Input heatmap of shape (H, W).

        Returns:
            Original heatmap if non-zero, or uniform 0.5 if all-zeros.
        """
        if np.all(heatmap == 0) or np.max(np.abs(heatmap)) < 1e-10:
            logger.warning(
                "No gradient signal detected. The model may not have learned "
                "meaningful features for this input. Returning uniform heatmap."
            )
            return np.full_like(heatmap, _UNIFORM_HEATMAP_VALUE, dtype=np.float32)
        return heatmap

    @staticmethod
    def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
        """Normalize a heatmap to [0, 1] range.

        Args:
            heatmap: Raw heatmap values.

        Returns:
            Normalized heatmap with values in [0, 1].
        """
        heatmap = heatmap.astype(np.float32)
        min_val = heatmap.min()
        max_val = heatmap.max()

        if max_val - min_val < 1e-10:
            # Constant heatmap
            return np.zeros_like(heatmap)

        return (heatmap - min_val) / (max_val - min_val)
