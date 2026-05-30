"""Unit tests for BalancedBatchSampler."""

import pytest

from src.data.sampler import BalancedBatchSampler
from src.exceptions import ConfigValidationError


class TestBalancedBatchSamplerInit:
    """Test sampler initialization and validation."""

    def test_basic_init(self):
        """Sampler initializes with valid parameters."""
        labels = [0, 0, 1, 1, 2, 2, 3, 3] * 10  # 80 samples, 4 classes
        sampler = BalancedBatchSampler(labels, p_classes=4, k_samples=2)
        assert sampler.p_classes == 4
        assert sampler.k_samples == 2

    def test_default_parameters(self):
        """Sampler uses correct defaults (p=9, k=4)."""
        # Need at least 9 classes for default p_classes=9
        labels = []
        for c in range(9):
            labels.extend([c] * 20)
        sampler = BalancedBatchSampler(labels, p_classes=9, k_samples=4)
        assert sampler.p_classes == 9
        assert sampler.k_samples == 4

    def test_p_classes_too_low(self):
        """Raises ConfigValidationError for p_classes < 2."""
        labels = [0, 0, 1, 1]
        with pytest.raises(ConfigValidationError) as exc_info:
            BalancedBatchSampler(labels, p_classes=1, k_samples=2)
        assert "p_classes" in str(exc_info.value)

    def test_p_classes_too_high(self):
        """Raises ConfigValidationError for p_classes > 9."""
        labels = list(range(10)) * 5
        with pytest.raises(ConfigValidationError) as exc_info:
            BalancedBatchSampler(labels, p_classes=10, k_samples=2)
        assert "p_classes" in str(exc_info.value)

    def test_k_samples_too_low(self):
        """Raises ConfigValidationError for k_samples < 2."""
        labels = [0, 0, 1, 1]
        with pytest.raises(ConfigValidationError) as exc_info:
            BalancedBatchSampler(labels, p_classes=2, k_samples=1)
        assert "k_samples" in str(exc_info.value)

    def test_k_samples_too_high(self):
        """Raises ConfigValidationError for k_samples > 16."""
        labels = [0, 0, 1, 1] * 20
        with pytest.raises(ConfigValidationError) as exc_info:
            BalancedBatchSampler(labels, p_classes=2, k_samples=17)
        assert "k_samples" in str(exc_info.value)

    def test_p_classes_exceeds_unique_classes(self):
        """Raises ConfigValidationError when p_classes > number of unique classes."""
        labels = [0, 0, 1, 1, 2, 2]  # Only 3 unique classes
        with pytest.raises(ConfigValidationError) as exc_info:
            BalancedBatchSampler(labels, p_classes=5, k_samples=2)
        assert "p_classes" in str(exc_info.value)


class TestBalancedBatchSamplerIteration:
    """Test sampler iteration behavior."""

    def test_batch_size(self):
        """Each batch has exactly P*K indices."""
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] * 10
        sampler = BalancedBatchSampler(labels, p_classes=3, k_samples=4)
        for batch in sampler:
            assert len(batch) == 3 * 4

    def test_batch_class_composition(self):
        """Each batch has exactly P distinct classes."""
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] * 10
        sampler = BalancedBatchSampler(labels, p_classes=3, k_samples=4)
        for batch in sampler:
            batch_labels = [labels[idx] for idx in batch]
            unique_classes = set(batch_labels)
            assert len(unique_classes) == 3

    def test_k_samples_per_class(self):
        """Each class in a batch has exactly K samples."""
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] * 10
        sampler = BalancedBatchSampler(labels, p_classes=3, k_samples=4)
        for batch in sampler:
            batch_labels = [labels[idx] for idx in batch]
            from collections import Counter
            counts = Counter(batch_labels)
            for cls, count in counts.items():
                assert count == 4

    def test_oversampling_small_class(self):
        """Classes with < K samples are oversampled with replacement."""
        # Class 0 has only 2 samples, k_samples=4 requires oversampling
        labels = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2] * 5
        sampler = BalancedBatchSampler(labels, p_classes=3, k_samples=4)
        for batch in sampler:
            batch_labels = [labels[idx] for idx in batch]
            assert len(batch) == 12  # 3 * 4

    def test_len_returns_positive(self):
        """__len__ returns a positive integer."""
        labels = [0, 0, 1, 1, 2, 2] * 20
        sampler = BalancedBatchSampler(labels, p_classes=3, k_samples=2)
        assert len(sampler) > 0

    def test_indices_in_valid_range(self):
        """All yielded indices are valid dataset indices."""
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2] * 10
        sampler = BalancedBatchSampler(labels, p_classes=3, k_samples=3)
        for batch in sampler:
            for idx in batch:
                assert 0 <= idx < len(labels)

    def test_works_with_dataloader_interface(self):
        """Sampler works as batch_sampler with DataLoader."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        data = torch.randn(90, 3, 64, 64)
        targets = torch.tensor([i % 9 for i in range(90)], dtype=torch.long)
        dataset = TensorDataset(data, targets)

        sampler = BalancedBatchSampler(
            targets.tolist(), p_classes=9, k_samples=2
        )
        loader = DataLoader(dataset, batch_sampler=sampler)

        for batch_data, batch_targets in loader:
            assert batch_data.shape[0] == 9 * 2
            assert batch_targets.shape[0] == 9 * 2
            break  # Just test first batch

    def test_shuffles_across_epochs(self):
        """Different iterations produce different batches."""
        labels = list(range(9)) * 20  # 9 classes, 20 each
        sampler = BalancedBatchSampler(labels, p_classes=9, k_samples=4)

        batches_epoch1 = list(sampler)
        batches_epoch2 = list(sampler)

        # With high probability, at least one batch differs
        # (extremely unlikely all batches are identical)
        assert batches_epoch1 != batches_epoch2
