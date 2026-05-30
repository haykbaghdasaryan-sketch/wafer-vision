"""Unit tests for GradCAMVisualizer (CNN + ViT explainability)."""

import numpy as np
import pytest
import torch

from src.models.backbone import BaseBackbone
from src.visualization.gradcam import GradCAMVisualizer, _UNIFORM_HEATMAP_VALUE


class SimpleCNNBackbone(BaseBackbone):
    """Minimal CNN backbone for testing Grad-CAM."""

    def __init__(self):
        super().__init__(l2_normalize=False)
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(32, 10)

        # Expose last conv layer for Grad-CAM detection
        self.features = self.conv2

    @property
    def embedding_dim(self) -> int:
        return 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TestGradCAMVisualizer:
    """Tests for GradCAMVisualizer with CNN backbone."""

    @pytest.fixture
    def cnn_model(self):
        """Create a simple CNN backbone for testing."""
        model = SimpleCNNBackbone()
        model.eval()
        return model

    @pytest.fixture
    def visualizer(self, cnn_model):
        """Create a GradCAMVisualizer instance."""
        return GradCAMVisualizer(cnn_model, target_layer="features")

    @pytest.fixture
    def sample_image(self):
        """Create a sample image tensor."""
        return torch.rand(1, 3, 64, 64, dtype=torch.float32)

    def test_init_cnn(self, cnn_model):
        """Test initialization with CNN backbone."""
        viz = GradCAMVisualizer(cnn_model, target_layer="features")
        assert not viz.is_vit
        assert viz.model is cnn_model

    def test_compute_explanation_output_shape(self, visualizer, sample_image):
        """Test that compute_explanation returns correct shape."""
        heatmap = visualizer.compute_explanation(sample_image)
        assert heatmap.shape == (64, 64)
        assert heatmap.dtype == np.float32

    def test_compute_explanation_value_range(self, visualizer, sample_image):
        """Test that heatmap values are in [0, 1]."""
        heatmap = visualizer.compute_explanation(sample_image)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_compute_explanation_3d_input(self, visualizer):
        """Test that 3D input (C, H, W) is handled correctly."""
        image = torch.rand(3, 64, 64, dtype=torch.float32)
        heatmap = visualizer.compute_explanation(image)
        assert heatmap.shape == (64, 64)

    def test_compute_explanation_with_target_class(self, visualizer, sample_image):
        """Test computation with explicit target class."""
        heatmap = visualizer.compute_explanation(sample_image, target_class=3)
        assert heatmap.shape == (64, 64)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_compute_batch(self, visualizer):
        """Test batch computation."""
        images = torch.rand(4, 3, 64, 64, dtype=torch.float32)
        heatmaps = visualizer.compute_batch(images)
        assert len(heatmaps) == 4
        for hm in heatmaps:
            assert hm.shape == (64, 64)
            assert hm.min() >= 0.0
            assert hm.max() <= 1.0

    def test_compute_batch_max_size(self, visualizer):
        """Test that batch size > 16 raises ValueError."""
        images = torch.rand(17, 3, 64, 64, dtype=torch.float32)
        with pytest.raises(ValueError, match="exceeds maximum"):
            visualizer.compute_batch(images)

    def test_compute_batch_with_targets(self, visualizer):
        """Test batch computation with target classes."""
        images = torch.rand(3, 3, 64, 64, dtype=torch.float32)
        targets = [0, 5, 2]
        heatmaps = visualizer.compute_batch(images, target_classes=targets)
        assert len(heatmaps) == 3

    def test_compute_batch_target_length_mismatch(self, visualizer):
        """Test that mismatched target_classes raises ValueError."""
        images = torch.rand(3, 3, 64, 64, dtype=torch.float32)
        targets = [0, 1]  # Length mismatch
        with pytest.raises(ValueError, match="must match batch size"):
            visualizer.compute_batch(images, target_classes=targets)

    def test_guided_gradcam(self, visualizer, sample_image):
        """Test guided Grad-CAM computation."""
        heatmap = visualizer.compute_explanation(sample_image, guided=True)
        assert heatmap.shape == (64, 64)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0


class TestCreateOverlay:
    """Tests for the create_overlay static method."""

    def test_overlay_basic(self):
        """Test basic overlay creation."""
        wafer_map = np.random.choice([0, 1, 2], size=(64, 64)).astype(np.uint8)
        heatmap = np.random.rand(64, 64).astype(np.float32)

        overlay = GradCAMVisualizer.create_overlay(wafer_map, heatmap)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8

    def test_overlay_value_range(self):
        """Test that overlay pixel values are in [0, 255]."""
        wafer_map = np.random.choice([0, 1, 2], size=(32, 32)).astype(np.uint8)
        heatmap = np.random.rand(32, 32).astype(np.float32)

        overlay = GradCAMVisualizer.create_overlay(wafer_map, heatmap)
        assert overlay.min() >= 0
        assert overlay.max() <= 255

    def test_overlay_alpha_clamping(self):
        """Test that alpha is clamped to [0.1, 0.9]."""
        wafer_map = np.ones((32, 32), dtype=np.uint8)
        heatmap = np.ones((32, 32), dtype=np.float32) * 0.5

        # alpha=0.0 should be clamped to 0.1
        overlay1 = GradCAMVisualizer.create_overlay(wafer_map, heatmap, alpha=0.0)
        assert overlay1 is not None

        # alpha=1.0 should be clamped to 0.9
        overlay2 = GradCAMVisualizer.create_overlay(wafer_map, heatmap, alpha=1.0)
        assert overlay2 is not None

    def test_overlay_mismatched_dimensions(self):
        """Test overlay with different wafer and heatmap sizes."""
        wafer_map = np.random.choice([0, 1, 2], size=(64, 64)).astype(np.uint8)
        heatmap = np.random.rand(32, 32).astype(np.float32)  # Different size

        overlay = GradCAMVisualizer.create_overlay(wafer_map, heatmap)
        assert overlay.shape == (64, 64, 3)

    def test_overlay_colorization(self):
        """Test that wafer map colorization is correct."""
        wafer_map = np.zeros((10, 10), dtype=np.uint8)
        wafer_map[0, 0] = 1  # Green
        wafer_map[1, 1] = 2  # Red
        heatmap = np.zeros((10, 10), dtype=np.float32)  # No heatmap contribution

        overlay = GradCAMVisualizer.create_overlay(wafer_map, heatmap, alpha=0.1)
        # With low alpha (0.1 heatmap), the wafer colors should dominate
        # Green pixel (0,0): should be mostly green
        assert overlay[0, 0, 1] > overlay[0, 0, 0]  # Green channel > red
        # Red pixel (1,1): should be mostly red
        assert overlay[1, 1, 0] > overlay[1, 1, 1]  # Red channel > green


class TestAllZerosHeatmap:
    """Tests for the all-zeros heatmap edge case."""

    @pytest.fixture
    def cnn_model(self):
        return SimpleCNNBackbone()

    def test_zero_heatmap_returns_uniform(self, cnn_model):
        """Test that all-zeros heatmap returns uniform 0.5."""
        viz = GradCAMVisualizer(cnn_model, target_layer="features")
        # Directly test the handler
        zero_heatmap = np.zeros((64, 64), dtype=np.float32)
        result = viz._handle_zero_heatmap(zero_heatmap)
        assert np.allclose(result, _UNIFORM_HEATMAP_VALUE)

    def test_near_zero_heatmap_returns_uniform(self, cnn_model):
        """Test that near-zero heatmap (< 1e-10) returns uniform 0.5."""
        viz = GradCAMVisualizer(cnn_model, target_layer="features")
        near_zero = np.full((64, 64), 1e-12, dtype=np.float32)
        result = viz._handle_zero_heatmap(near_zero)
        assert np.allclose(result, _UNIFORM_HEATMAP_VALUE)

    def test_nonzero_heatmap_unchanged(self, cnn_model):
        """Test that non-zero heatmaps are returned unchanged."""
        viz = GradCAMVisualizer(cnn_model, target_layer="features")
        heatmap = np.random.rand(64, 64).astype(np.float32)
        result = viz._handle_zero_heatmap(heatmap)
        assert np.array_equal(result, heatmap)


class TestNormalizeHeatmap:
    """Tests for the static normalize method."""

    def test_normalize_basic(self):
        """Test normalization to [0, 1]."""
        heatmap = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = GradCAMVisualizer._normalize_heatmap(heatmap)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_normalize_constant_array(self):
        """Test normalization of constant array returns zeros."""
        heatmap = np.full((10, 10), 5.0, dtype=np.float32)
        result = GradCAMVisualizer._normalize_heatmap(heatmap)
        assert np.all(result == 0.0)

    def test_normalize_already_01(self):
        """Test that [0,1] range stays normalized."""
        heatmap = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
        result = GradCAMVisualizer._normalize_heatmap(heatmap)
        assert result.min() == 0.0
        assert result.max() == 1.0
