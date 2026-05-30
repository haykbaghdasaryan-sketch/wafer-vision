"""Unit tests for LR schedulers: WarmupCosineScheduler and LayerwiseLRDecay."""

import math

import pytest
import torch
import torch.nn as nn

from src.training.scheduler import LayerwiseLRDecay, WarmupCosineScheduler


# --- Fixtures ---


@pytest.fixture
def simple_model():
    """Create a simple multi-layer model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 5),
    )
    return model


@pytest.fixture
def simple_optimizer(simple_model):
    """Create an optimizer for the simple model."""
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


# --- WarmupCosineScheduler Tests ---


class TestWarmupCosineScheduler:
    """Tests for WarmupCosineScheduler."""

    def test_initial_lr_is_zero(self, simple_optimizer):
        """At epoch 0 (start of warmup), LR should be 0."""
        scheduler = WarmupCosineScheduler(
            simple_optimizer, warmup_epochs=5, total_epochs=100
        )
        lrs = scheduler.get_lr()
        assert all(lr == 0.0 for lr in lrs)

    def test_warmup_linear_increase(self, simple_optimizer):
        """During warmup, LR increases linearly from 0 to base_lr."""
        base_lr = 0.01
        warmup_epochs = 5
        scheduler = WarmupCosineScheduler(
            simple_optimizer, warmup_epochs=warmup_epochs, total_epochs=100
        )

        lrs_over_time = []
        for epoch in range(warmup_epochs):
            lrs_over_time.append(scheduler.get_lr()[0])
            scheduler.step()

        # Verify linear increase
        for i, lr in enumerate(lrs_over_time):
            expected = base_lr * (i / warmup_epochs)
            assert abs(lr - expected) < 1e-10, (
                f"Epoch {i}: expected {expected}, got {lr}"
            )

    def test_lr_at_warmup_end_equals_base_lr(self, simple_optimizer):
        """At the end of warmup, LR should equal base_lr."""
        base_lr = 0.01
        warmup_epochs = 5
        scheduler = WarmupCosineScheduler(
            simple_optimizer, warmup_epochs=warmup_epochs, total_epochs=100
        )

        # Step through warmup
        for _ in range(warmup_epochs):
            scheduler.step()

        # At epoch = warmup_epochs, cosine phase starts with full base_lr
        lrs = scheduler.get_lr()
        expected = base_lr  # cos(0) = 1, so factor = 1.0
        assert abs(lrs[0] - expected) < 1e-10

    def test_cosine_annealing_decreases(self, simple_optimizer):
        """After warmup, LR should decrease following cosine schedule."""
        warmup_epochs = 5
        total_epochs = 50
        scheduler = WarmupCosineScheduler(
            simple_optimizer, warmup_epochs=warmup_epochs, total_epochs=total_epochs
        )

        # Step through warmup
        for _ in range(warmup_epochs):
            scheduler.step()

        prev_lr = scheduler.get_lr()[0]
        # Step through cosine phase and verify monotonic decrease
        for _ in range(warmup_epochs, total_epochs - 1):
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
            assert current_lr <= prev_lr + 1e-10, (
                f"LR should decrease: prev={prev_lr}, current={current_lr}"
            )
            prev_lr = current_lr

    def test_lr_at_end_equals_eta_min(self, simple_optimizer):
        """At the final epoch (total_epochs), LR should equal eta_min."""
        base_lr = 0.01
        warmup_epochs = 5
        total_epochs = 50
        eta_min = 1e-6
        scheduler = WarmupCosineScheduler(
            simple_optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            eta_min=eta_min,
        )

        # Step to epoch = total_epochs (progress = 1.0, cos(pi) = -1)
        for _ in range(total_epochs):
            scheduler.step()

        lrs = scheduler.get_lr()
        # At epoch = total_epochs: progress = (50 - 5) / (50 - 5) = 1.0
        # cos(pi * 1.0) = -1, factor = 0.5 * (1 + (-1)) = 0
        assert abs(lrs[0] - eta_min) < 1e-10

    def test_cosine_formula_correctness(self, simple_optimizer):
        """Verify the exact cosine formula at a specific epoch."""
        base_lr = 0.01
        warmup_epochs = 10
        total_epochs = 100
        eta_min = 1e-6
        scheduler = WarmupCosineScheduler(
            simple_optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            eta_min=eta_min,
        )

        # Step to epoch 55 (in the cosine phase)
        for _ in range(55):
            scheduler.step()

        lr = scheduler.get_lr()[0]
        # Epoch 55: progress = (55 - 10) / (100 - 10) = 45/90 = 0.5
        progress = (55 - warmup_epochs) / (total_epochs - warmup_epochs)
        expected = eta_min + (base_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        assert abs(lr - expected) < 1e-10

    def test_multiple_param_groups(self):
        """Scheduler works with multiple parameter groups at different base LRs."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
        optimizer = torch.optim.SGD([
            {"params": model[0].parameters(), "lr": 0.01},
            {"params": model[1].parameters(), "lr": 0.001},
        ])
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=5, total_epochs=50
        )

        # Step to epoch 3 (warmup)
        for _ in range(3):
            scheduler.step()

        lrs = scheduler.get_lr()
        # Both groups should scale proportionally
        assert abs(lrs[0] / lrs[1] - 10.0) < 1e-8  # ratio should be preserved

    def test_invalid_warmup_ge_total_raises(self, simple_optimizer):
        """warmup_epochs >= total_epochs should raise ValueError."""
        with pytest.raises(ValueError, match="warmup_epochs.*must be less than"):
            WarmupCosineScheduler(
                simple_optimizer, warmup_epochs=100, total_epochs=100
            )

    def test_invalid_negative_warmup_raises(self, simple_optimizer):
        """Negative warmup_epochs should raise ValueError."""
        with pytest.raises(ValueError, match="warmup_epochs must be non-negative"):
            WarmupCosineScheduler(
                simple_optimizer, warmup_epochs=-1, total_epochs=100
            )

    def test_invalid_zero_total_raises(self, simple_optimizer):
        """Zero total_epochs should raise ValueError."""
        with pytest.raises(ValueError, match="total_epochs must be positive"):
            WarmupCosineScheduler(
                simple_optimizer, warmup_epochs=0, total_epochs=0
            )

    def test_invalid_negative_eta_min_raises(self, simple_optimizer):
        """Negative eta_min should raise ValueError."""
        with pytest.raises(ValueError, match="eta_min must be non-negative"):
            WarmupCosineScheduler(
                simple_optimizer, warmup_epochs=5, total_epochs=100, eta_min=-0.1
            )

    def test_lr_always_non_negative(self, simple_optimizer):
        """LR should never be negative throughout the full schedule."""
        scheduler = WarmupCosineScheduler(
            simple_optimizer, warmup_epochs=10, total_epochs=100, eta_min=1e-6
        )
        for epoch in range(100):
            lrs = scheduler.get_lr()
            assert all(lr >= 0 for lr in lrs), f"Negative LR at epoch {epoch}: {lrs}"
            scheduler.step()

    def test_lr_never_exceeds_base(self, simple_optimizer):
        """LR should never exceed the base learning rate."""
        base_lr = 0.01
        scheduler = WarmupCosineScheduler(
            simple_optimizer, warmup_epochs=10, total_epochs=100
        )
        for epoch in range(100):
            lrs = scheduler.get_lr()
            assert all(lr <= base_lr + 1e-10 for lr in lrs), (
                f"LR exceeds base at epoch {epoch}: {lrs}"
            )
            scheduler.step()


# --- LayerwiseLRDecay Tests ---


class TestLayerwiseLRDecay:
    """Tests for LayerwiseLRDecay."""

    def test_create_param_groups_basic(self, simple_model):
        """create_param_groups returns correct number of groups."""
        base_lr = 1e-3
        groups = LayerwiseLRDecay.create_param_groups(
            simple_model, base_lr=base_lr, decay_factor=0.9
        )
        # nn.Sequential with Linear, ReLU, Linear, ReLU, Linear
        # ReLU has no parameters, so we expect 3 groups (one per Linear)
        assert len(groups) == 3

    def test_create_param_groups_lr_ordering(self, simple_model):
        """Top layers should have higher LR than lower layers."""
        base_lr = 1e-3
        groups = LayerwiseLRDecay.create_param_groups(
            simple_model, base_lr=base_lr, decay_factor=0.9
        )
        # Last group (topmost layer) should have the highest LR
        lrs = [g["lr"] for g in groups]
        # lrs should be increasing (lower layers first with lower lr)
        assert lrs[-1] > lrs[0], (
            f"Top layer LR ({lrs[-1]}) should be > bottom layer LR ({lrs[0]})"
        )

    def test_create_param_groups_decay_formula(self, simple_model):
        """Verify the exact decay formula for each layer."""
        base_lr = 1e-3
        decay_factor = 0.9
        groups = LayerwiseLRDecay.create_param_groups(
            simple_model, base_lr=base_lr, decay_factor=decay_factor
        )

        n_layers = len(groups)
        for i, group in enumerate(groups):
            depth = n_layers - 1 - i
            expected_lr = base_lr * (decay_factor ** depth)
            assert abs(group["lr"] - expected_lr) < 1e-12, (
                f"Layer {i}: expected lr={expected_lr}, got {group['lr']}"
            )

    def test_create_param_groups_top_layer_gets_base_lr(self, simple_model):
        """The topmost layer should get exactly base_lr (depth=0)."""
        base_lr = 1e-3
        groups = LayerwiseLRDecay.create_param_groups(
            simple_model, base_lr=base_lr, decay_factor=0.9
        )
        # Last group is topmost
        assert abs(groups[-1]["lr"] - base_lr) < 1e-12

    def test_create_param_groups_all_params_included(self, simple_model):
        """All trainable parameters should be present in the groups."""
        groups = LayerwiseLRDecay.create_param_groups(
            simple_model, base_lr=1e-3, decay_factor=0.9
        )
        total_params_in_groups = sum(
            len(g["params"]) for g in groups
        )
        total_model_params = sum(
            1 for _ in simple_model.parameters()
        )
        assert total_params_in_groups == total_model_params

    def test_create_param_groups_decay_factor_1(self, simple_model):
        """With decay_factor=1.0, all layers get the same LR."""
        base_lr = 1e-3
        groups = LayerwiseLRDecay.create_param_groups(
            simple_model, base_lr=base_lr, decay_factor=1.0
        )
        for group in groups:
            assert abs(group["lr"] - base_lr) < 1e-12

    def test_layerwise_decay_applies_to_optimizer(self, simple_model):
        """LayerwiseLRDecay modifies optimizer param group LRs."""
        base_lr = 1e-3
        decay_factor = 0.9
        param_groups = LayerwiseLRDecay.create_param_groups(
            simple_model, base_lr=base_lr, decay_factor=1.0  # Start uniform
        )
        optimizer = torch.optim.Adam(param_groups)

        # Apply layerwise decay
        LayerwiseLRDecay(optimizer, simple_model, decay_factor=decay_factor)

        # Verify LRs were modified
        n_groups = len(optimizer.param_groups)
        for i, group in enumerate(optimizer.param_groups):
            depth = n_groups - 1 - i
            expected_lr = base_lr * (decay_factor ** depth)
            assert abs(group["lr"] - expected_lr) < 1e-12

    def test_invalid_decay_factor_zero_raises(self, simple_model):
        """decay_factor=0 should raise ValueError."""
        with pytest.raises(ValueError, match="decay_factor must be in"):
            LayerwiseLRDecay.create_param_groups(
                simple_model, base_lr=1e-3, decay_factor=0.0
            )

    def test_invalid_decay_factor_negative_raises(self, simple_model):
        """Negative decay_factor should raise ValueError."""
        with pytest.raises(ValueError, match="decay_factor must be in"):
            LayerwiseLRDecay.create_param_groups(
                simple_model, base_lr=1e-3, decay_factor=-0.5
            )

    def test_invalid_decay_factor_gt_1_raises(self, simple_model):
        """decay_factor > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="decay_factor must be in"):
            LayerwiseLRDecay.create_param_groups(
                simple_model, base_lr=1e-3, decay_factor=1.5
            )

    def test_invalid_base_lr_raises(self, simple_model):
        """Non-positive base_lr should raise ValueError."""
        with pytest.raises(ValueError, match="base_lr must be positive"):
            LayerwiseLRDecay.create_param_groups(
                simple_model, base_lr=0.0, decay_factor=0.9
            )

    def test_model_with_no_children_params(self):
        """Model with no children that have params returns single group."""
        # A single linear layer (named_children has no sub-modules with params in this form)
        model = nn.Linear(10, 5)
        groups = LayerwiseLRDecay.create_param_groups(
            model, base_lr=1e-3, decay_factor=0.9
        )
        # nn.Linear has no named_children with parameters, so falls back to all params
        assert len(groups) == 1
        assert groups[0]["lr"] == 1e-3

    def test_create_param_groups_with_resnet_like_model(self):
        """Test with a model that has named children like a backbone."""
        model = nn.Sequential()
        model.add_module("conv1", nn.Conv2d(3, 64, 3))
        model.add_module("layer1", nn.Sequential(nn.Conv2d(64, 128, 3)))
        model.add_module("layer2", nn.Sequential(nn.Conv2d(128, 256, 3)))
        model.add_module("fc", nn.Linear(256, 10))

        base_lr = 1e-3
        decay_factor = 0.8
        groups = LayerwiseLRDecay.create_param_groups(
            model, base_lr=base_lr, decay_factor=decay_factor
        )

        assert len(groups) == 4
        # fc (topmost) should have base_lr
        assert abs(groups[-1]["lr"] - base_lr) < 1e-12
        # conv1 (bottommost) should have base_lr * 0.8^3
        expected_bottom = base_lr * (decay_factor ** 3)
        assert abs(groups[0]["lr"] - expected_bottom) < 1e-12
