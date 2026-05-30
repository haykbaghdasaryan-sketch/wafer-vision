"""Unit tests for data augmentation transforms."""

import pytest
import torch

from src.data.augmentation import WaferAugmentation, EvalTransform, SimCLRAugmentation
from src.exceptions import ConfigValidationError


class TestWaferAugmentation:
    """Tests for WaferAugmentation class."""

    def test_default_construction(self):
        """Test default initialization with medium strength."""
        aug = WaferAugmentation()
        assert aug.strength == "medium"
        assert aug.noise_std == 0.01
        assert aug.rotation is True
        assert aug.flip_h is True
        assert aug.flip_v is True
        assert aug.noise_enabled is True

    def test_strength_presets(self):
        """Test each strength preset configures noise_std correctly."""
        light = WaferAugmentation(strength="light")
        assert light.noise_std == 0.005

        medium = WaferAugmentation(strength="medium")
        assert medium.noise_std == 0.01

        heavy = WaferAugmentation(strength="heavy")
        assert heavy.noise_std == 0.05

    def test_custom_noise_std_overrides_preset(self):
        """Test explicit noise_std overrides preset value."""
        aug = WaferAugmentation(strength="light", noise_std=0.03)
        assert aug.noise_std == 0.03

    def test_invalid_noise_std_raises(self):
        """Test noise_std outside [0.001, 0.1] raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError):
            WaferAugmentation(noise_std=0.0001)
        with pytest.raises(ConfigValidationError):
            WaferAugmentation(noise_std=0.5)

    def test_invalid_strength_raises(self):
        """Test invalid strength raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError):
            WaferAugmentation(strength="extreme")

    def test_output_shape_preserved(self):
        """Test output shape matches input shape."""
        aug = WaferAugmentation()
        tensor = torch.rand(3, 64, 64)
        result = aug(tensor)
        assert result.shape == (3, 64, 64)

    def test_output_values_in_range(self):
        """Test output values are clamped to [0, 1]."""
        aug = WaferAugmentation(noise_std=0.1)
        tensor = torch.rand(3, 64, 64)
        result = aug(tensor)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_no_noise_mode(self):
        """Test with noise disabled produces values from input range."""
        aug = WaferAugmentation(noise_enabled=False)
        tensor = torch.rand(3, 32, 32)
        result = aug(tensor)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_non_square_input(self):
        """Test augmentation works with non-square inputs preserving shape."""
        aug = WaferAugmentation()
        tensor = torch.rand(3, 48, 32)
        result = aug(tensor)
        assert result.shape == (3, 48, 32)

    def test_rotation_only(self):
        """Test with only rotation enabled."""
        aug = WaferAugmentation(rotation=True, flip_h=False, flip_v=False, noise_enabled=False)
        tensor = torch.rand(3, 64, 64)
        result = aug(tensor)
        assert result.shape == (3, 64, 64)

    def test_all_augmentations_disabled(self):
        """Test with all augmentations disabled returns input unchanged."""
        aug = WaferAugmentation(
            rotation=False, flip_h=False, flip_v=False, noise_enabled=False
        )
        tensor = torch.rand(3, 64, 64)
        result = aug(tensor)
        assert torch.equal(result, tensor)


class TestEvalTransform:
    """Tests for EvalTransform class."""

    def test_default_target_size(self):
        """Test default construction with 64×64 target."""
        transform = EvalTransform()
        assert transform.target_size == 64

    def test_output_shape(self):
        """Test output has correct target dimensions."""
        transform = EvalTransform(target_size=32)
        tensor = torch.rand(3, 64, 64)
        result = transform(tensor)
        assert result.shape == (3, 32, 32)

    def test_deterministic(self):
        """Test applying twice gives identical results."""
        transform = EvalTransform(target_size=48)
        tensor = torch.rand(3, 64, 64)
        result1 = transform(tensor)
        result2 = transform(tensor)
        assert torch.equal(result1, result2)

    def test_non_square_input(self):
        """Test center-crop on non-square input."""
        transform = EvalTransform(target_size=32)
        tensor = torch.rand(3, 80, 60)
        result = transform(tensor)
        assert result.shape == (3, 32, 32)

    def test_already_target_size(self):
        """Test input already at target size passes through."""
        transform = EvalTransform(target_size=64)
        tensor = torch.rand(3, 64, 64)
        result = transform(tensor)
        assert result.shape == (3, 64, 64)


class TestSimCLRAugmentation:
    """Tests for SimCLRAugmentation class."""

    def test_default_construction(self):
        """Test default initialization."""
        aug = SimCLRAugmentation()
        assert aug.target_size == 64
        assert aug.crop_scale == (0.6, 1.0)
        assert aug.noise_std == 0.02

    def test_output_two_views(self):
        """Test __call__ returns a tuple of two tensors."""
        aug = SimCLRAugmentation(target_size=32)
        tensor = torch.rand(3, 64, 64)
        view1, view2 = aug(tensor)
        assert isinstance(view1, torch.Tensor)
        assert isinstance(view2, torch.Tensor)

    def test_output_shapes(self):
        """Test both views have correct target shape."""
        aug = SimCLRAugmentation(target_size=48)
        tensor = torch.rand(3, 64, 64)
        view1, view2 = aug(tensor)
        assert view1.shape == (3, 48, 48)
        assert view2.shape == (3, 48, 48)

    def test_output_values_in_range(self):
        """Test both views have values in [0, 1]."""
        aug = SimCLRAugmentation(target_size=32)
        tensor = torch.rand(3, 64, 64)
        view1, view2 = aug(tensor)
        assert view1.min() >= 0.0
        assert view1.max() <= 1.0
        assert view2.min() >= 0.0
        assert view2.max() <= 1.0

    def test_views_are_different(self):
        """Test that the two views are not identical (stochastic)."""
        aug = SimCLRAugmentation(target_size=32)
        # Use a large enough tensor to avoid degenerate case
        tensor = torch.rand(3, 64, 64)
        view1, view2 = aug(tensor)
        # Views should generally be different (extremely unlikely to be identical)
        assert not torch.equal(view1, view2)

    def test_cosine_similarity_below_threshold(self):
        """Test cosine similarity between views is below 0.99."""
        aug = SimCLRAugmentation(target_size=32)
        tensor = torch.rand(3, 64, 64)
        view1, view2 = aug(tensor)
        # Flatten to compute cosine similarity
        v1_flat = view1.flatten()
        v2_flat = view2.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            v1_flat.unsqueeze(0), v2_flat.unsqueeze(0)
        )
        assert cos_sim.item() < 0.99

    def test_non_square_input(self):
        """Test SimCLR augmentation works with non-square inputs."""
        aug = SimCLRAugmentation(target_size=32)
        tensor = torch.rand(3, 80, 60)
        view1, view2 = aug(tensor)
        assert view1.shape == (3, 32, 32)
        assert view2.shape == (3, 32, 32)
