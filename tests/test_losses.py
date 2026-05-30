"""Unit tests for loss functions in src/training/losses.py."""

import pytest
import torch

from src.registry import loss_registry
from src.training.losses import (
    NTXentLoss,
    SupervisedContrastiveLoss,
    TripletMarginLossWithMining,
    WeightedCrossEntropyLoss,
)


class TestTripletMarginLossWithMining:
    """Tests for TripletMarginLossWithMining."""

    def test_basic_forward(self):
        """Loss produces non-negative scalar output."""
        loss_fn = TripletMarginLossWithMining(margin=0.2)
        embeddings = torch.randn(16, 128)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        loss = loss_fn(embeddings, labels)
        assert loss.item() >= 0
        assert loss.ndim == 0  # scalar

    def test_single_class_returns_zero(self):
        """Batch with < 2 classes returns loss=0."""
        loss_fn = TripletMarginLossWithMining(margin=0.2)
        embeddings = torch.randn(8, 64)
        labels = torch.zeros(8, dtype=torch.long)
        loss = loss_fn(embeddings, labels)
        assert loss.item() == 0.0

    def test_gradient_flow(self):
        """Gradients propagate through the loss."""
        loss_fn = TripletMarginLossWithMining(margin=0.2)
        embeddings = torch.randn(12, 64, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        loss = loss_fn(embeddings, labels)
        loss.backward()
        # Gradient should exist (may be zero for some elements)
        assert embeddings.grad is not None

    def test_well_separated_clusters_zero_loss(self):
        """Well-separated clusters produce zero or near-zero loss."""
        loss_fn = TripletMarginLossWithMining(margin=0.2)
        torch.manual_seed(0)
        embeddings = torch.zeros(8, 32)
        # Cluster 0 at origin
        embeddings[0:4] = torch.randn(4, 32) * 0.001
        # Cluster 1 far away
        embeddings[4:8] = torch.randn(4, 32) * 0.001 + 100
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = loss_fn(embeddings, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-3)

    def test_margin_parameter(self):
        """Different margin values affect the loss."""
        embeddings = torch.randn(16, 64)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        torch.manual_seed(42)
        loss_small = TripletMarginLossWithMining(margin=0.1)(embeddings, labels)
        loss_large = TripletMarginLossWithMining(margin=1.0)(embeddings, labels)
        # Larger margin should produce larger loss
        assert loss_large.item() >= loss_small.item()

    def test_two_classes_minimum(self):
        """Works with exactly 2 classes (minimum valid case)."""
        loss_fn = TripletMarginLossWithMining(margin=0.2)
        embeddings = torch.randn(8, 32)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = loss_fn(embeddings, labels)
        assert loss.item() >= 0


class TestSupervisedContrastiveLoss:
    """Tests for SupervisedContrastiveLoss."""

    def test_basic_forward(self):
        """Loss produces non-negative scalar output."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        features = torch.randn(16, 128)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        loss = loss_fn(features, labels)
        assert loss.item() >= 0
        assert loss.ndim == 0

    def test_single_class_returns_zero(self):
        """Batch with < 2 classes returns loss=0."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        features = torch.randn(8, 64)
        labels = torch.zeros(8, dtype=torch.long)
        loss = loss_fn(features, labels)
        assert loss.item() == 0.0

    def test_gradient_flow(self):
        """Gradients propagate through the loss."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        features = torch.randn(12, 64, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        loss = loss_fn(features, labels)
        loss.backward()
        assert features.grad is not None

    def test_temperature_effect(self):
        """Different temperatures produce different loss values."""
        torch.manual_seed(42)
        features = torch.randn(16, 64)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        loss_low_t = SupervisedContrastiveLoss(temperature=0.01)(features, labels)
        loss_high_t = SupervisedContrastiveLoss(temperature=1.0)(features, labels)
        # Different temperature should give different loss
        assert loss_low_t.item() != loss_high_t.item()

    def test_normalized_features(self):
        """Works correctly with L2-normalized features."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        features = torch.nn.functional.normalize(torch.randn(12, 64), p=2, dim=1)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        loss = loss_fn(features, labels)
        assert loss.item() >= 0


class TestNTXentLoss:
    """Tests for NTXentLoss."""

    def test_basic_forward(self):
        """Loss produces non-negative scalar output."""
        loss_fn = NTXentLoss(temperature=0.5)
        z_i = torch.randn(8, 64)
        z_j = torch.randn(8, 64)
        loss = loss_fn(z_i, z_j)
        assert loss.item() >= 0
        assert loss.ndim == 0

    def test_identical_views_low_loss(self):
        """Identical views should produce relatively low loss."""
        loss_fn = NTXentLoss(temperature=0.5)
        z = torch.randn(8, 64)
        # Add small noise so views are nearly identical
        loss = loss_fn(z, z + torch.randn_like(z) * 0.001)
        # With nearly identical positive pairs, loss should be relatively low
        assert loss.item() >= 0
        assert loss.item() < 5.0  # sanity upper bound

    def test_gradient_flow(self):
        """Gradients propagate through the loss."""
        loss_fn = NTXentLoss(temperature=0.5)
        z_i = torch.randn(8, 32, requires_grad=True)
        z_j = torch.randn(8, 32, requires_grad=True)
        loss = loss_fn(z_i, z_j)
        loss.backward()
        assert z_i.grad is not None
        assert z_j.grad is not None

    def test_temperature_effect(self):
        """Different temperatures produce different loss values."""
        torch.manual_seed(42)
        z_i = torch.randn(8, 64)
        z_j = torch.randn(8, 64)
        loss_low_t = NTXentLoss(temperature=0.1)(z_i, z_j)
        loss_high_t = NTXentLoss(temperature=1.0)(z_i, z_j)
        assert loss_low_t.item() != loss_high_t.item()

    def test_batch_size_one(self):
        """Works with batch size 1 (minimal case)."""
        loss_fn = NTXentLoss(temperature=0.5)
        z_i = torch.randn(1, 64)
        z_j = torch.randn(1, 64)
        loss = loss_fn(z_i, z_j)
        assert loss.item() >= 0


class TestWeightedCrossEntropyLoss:
    """Tests for WeightedCrossEntropyLoss."""

    def test_basic_forward(self):
        """Loss produces non-negative scalar output."""
        loss_fn = WeightedCrossEntropyLoss(
            class_counts=[1000, 500, 200, 100, 50, 30, 20, 10, 5], num_classes=9
        )
        logits = torch.randn(16, 9)
        labels = torch.randint(0, 9, (16,))
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0
        assert loss.ndim == 0

    def test_without_weights(self):
        """Works without class weights."""
        loss_fn = WeightedCrossEntropyLoss(num_classes=9, enable_weights=False)
        logits = torch.randn(16, 9)
        labels = torch.randint(0, 9, (16,))
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """Gradients propagate through the loss."""
        loss_fn = WeightedCrossEntropyLoss(
            class_counts=[100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_classes=9,
        )
        logits = torch.randn(8, 9, requires_grad=True)
        labels = torch.randint(0, 9, (8,))
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_correct_prediction_low_loss(self):
        """Correct predictions produce lower loss."""
        loss_fn = WeightedCrossEntropyLoss(num_classes=3)
        # High confidence correct predictions
        logits_correct = torch.tensor([[10.0, -10.0, -10.0], [10.0, -10.0, -10.0]])
        labels = torch.tensor([0, 0])
        loss_correct = loss_fn(logits_correct, labels)

        # Wrong predictions
        logits_wrong = torch.tensor([[-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]])
        loss_wrong = loss_fn(logits_wrong, labels)

        assert loss_correct.item() < loss_wrong.item()

    def test_inverse_frequency_weights(self):
        """Rare classes get higher weights."""
        loss_fn = WeightedCrossEntropyLoss(
            class_counts=[1000, 10, 500], num_classes=3
        )
        # Weight for class 1 (count=10) should be highest
        assert loss_fn.weight is not None
        assert loss_fn.weight[1] > loss_fn.weight[0]
        assert loss_fn.weight[1] > loss_fn.weight[2]

    def test_no_class_counts_no_weight(self):
        """Without class_counts, weight is None."""
        loss_fn = WeightedCrossEntropyLoss(num_classes=9)
        assert loss_fn.weight is None


class TestLossRegistry:
    """Tests for loss registry integration."""

    def test_all_losses_registered(self):
        """All 4 losses are registered."""
        # Import to trigger registration
        import src.training.loss_registry  # noqa: F401

        assert "triplet" in loss_registry
        assert "supcon" in loss_registry
        assert "ntxent" in loss_registry
        assert "crossentropy" in loss_registry

    def test_registry_create_triplet(self):
        """Registry creates TripletMarginLossWithMining."""
        import src.training.loss_registry  # noqa: F401

        loss = loss_registry.create("triplet", margin=0.3)
        assert isinstance(loss, TripletMarginLossWithMining)
        assert loss.margin == 0.3

    def test_registry_create_supcon(self):
        """Registry creates SupervisedContrastiveLoss."""
        import src.training.loss_registry  # noqa: F401

        loss = loss_registry.create("supcon", temperature=0.1)
        assert isinstance(loss, SupervisedContrastiveLoss)
        assert loss.temperature == 0.1

    def test_registry_create_ntxent(self):
        """Registry creates NTXentLoss."""
        import src.training.loss_registry  # noqa: F401

        loss = loss_registry.create("ntxent", temperature=0.3)
        assert isinstance(loss, NTXentLoss)
        assert loss.temperature == 0.3

    def test_registry_create_crossentropy(self):
        """Registry creates WeightedCrossEntropyLoss."""
        import src.training.loss_registry  # noqa: F401

        loss = loss_registry.create("crossentropy", num_classes=5)
        assert isinstance(loss, WeightedCrossEntropyLoss)
        assert loss.num_classes == 5
