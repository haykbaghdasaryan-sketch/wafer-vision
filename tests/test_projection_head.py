"""Unit tests for ProjectionHead MLP."""

import torch
import pytest

from src.models.projection_head import ProjectionHead


class TestProjectionHead:
    """Tests for ProjectionHead architecture and behavior."""

    def test_output_shape(self):
        """ProjectionHead produces correct output shape."""
        head = ProjectionHead(input_dim=2048)
        x = torch.randn(16, 2048)
        out = head(x)
        assert out.shape == (16, 128)

    def test_custom_dimensions(self):
        """ProjectionHead respects custom hidden and output dims."""
        head = ProjectionHead(input_dim=768, hidden_dim=256, output_dim=64)
        x = torch.randn(8, 768)
        out = head(x)
        assert out.shape == (8, 64)

    def test_output_unit_norm(self):
        """Output vectors always have unit L2 norm."""
        head = ProjectionHead(input_dim=2048)
        head.eval()
        x = torch.randn(32, 2048)
        out = head(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(32), atol=1e-5)

    def test_output_unit_norm_single_sample(self):
        """Single-sample batch still produces unit norm output."""
        head = ProjectionHead(input_dim=512)
        head.eval()
        x = torch.randn(1, 512)
        out = head(x)
        norm = out.norm(dim=1)
        assert torch.allclose(norm, torch.ones(1), atol=1e-5)

    def test_output_unit_norm_large_input(self):
        """Unit norm holds for large-magnitude inputs."""
        head = ProjectionHead(input_dim=1280)
        head.eval()
        x = torch.randn(8, 1280) * 1000.0
        out = head(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_output_unit_norm_small_input(self):
        """Unit norm holds for small-magnitude inputs."""
        head = ProjectionHead(input_dim=1280)
        head.eval()
        x = torch.randn(8, 1280) * 0.001
        out = head(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_architecture_layers(self):
        """Verify the MLP architecture: Linear → BatchNorm → ReLU → Linear."""
        head = ProjectionHead(input_dim=2048, hidden_dim=512, output_dim=128)
        layers = list(head.net.children())
        assert len(layers) == 4
        assert isinstance(layers[0], torch.nn.Linear)
        assert layers[0].in_features == 2048
        assert layers[0].out_features == 512
        assert isinstance(layers[1], torch.nn.BatchNorm1d)
        assert layers[1].num_features == 512
        assert isinstance(layers[2], torch.nn.ReLU)
        assert isinstance(layers[3], torch.nn.Linear)
        assert layers[3].in_features == 512
        assert layers[3].out_features == 128

    def test_export_from_models_package(self):
        """ProjectionHead is importable from src.models."""
        from src.models import ProjectionHead as PH
        assert PH is ProjectionHead

    def test_gradient_flow(self):
        """Gradients flow through the projection head."""
        head = ProjectionHead(input_dim=256)
        x = torch.randn(4, 256, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 256)
