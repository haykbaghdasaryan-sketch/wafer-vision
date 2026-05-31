"""TripletLoss, SupCon, NTXent, and CrossEntropy loss implementations.

All losses produce non-negative scalar output and integrate with
the loss_registry for config-driven instantiation.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TripletMarginLossWithMining(nn.Module):
    """Triplet loss with hard negative mining.

    Selects hardest negatives within batch. Falls back to semi-hard
    when no valid triplets exist (all negatives beyond margin).

    Properties:
        - Output is always non-negative
        - Output = 0 when d(a,p) + margin <= d(a,n) for all triplets

    Args:
        margin: Triplet margin (default 0.2, range [0.05, 1.0]).
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute triplet loss with mining.

        Args:
            embeddings: Shape (B, D), normalized or unnormalized.
            labels: Shape (B,), integer class labels.

        Returns:
            Scalar loss >= 0.
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Edge case: fewer than 2 distinct classes
        unique_classes = labels.unique()
        if unique_classes.numel() < 2:
            logger.warning(
                "Batch has fewer than 2 distinct classes (%d). "
                "Returning zero loss.",
                unique_classes.numel(),
            )
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute pairwise distance matrix
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # For each anchor, find hardest positive and hardest negative
        losses = []
        for i in range(batch_size):
            anchor_label = labels[i]

            # Positive mask: same class, different index
            pos_mask = (labels == anchor_label) & (
                torch.arange(batch_size, device=device) != i
            )
            # Negative mask: different class
            neg_mask = labels != anchor_label

            if not pos_mask.any() or not neg_mask.any():
                continue

            # Hardest positive: max distance among positives
            pos_dists = dist_mat[i][pos_mask]
            hardest_pos_dist = pos_dists.max()

            # Hardest negative: min distance among negatives (hard mining)
            neg_dists = dist_mat[i][neg_mask]
            hardest_neg_dist = neg_dists.min()

            # Try hard negative first
            loss_val = hardest_pos_dist - hardest_neg_dist + self.margin

            if loss_val.item() <= 0:
                # All hard negatives are farther than margin -> semi-hard fallback
                # Semi-hard: negatives closer than positive + margin but farther than positive
                semi_hard_mask = (neg_dists > hardest_pos_dist) & (
                    neg_dists < hardest_pos_dist + self.margin
                )
                if semi_hard_mask.any():
                    semi_hard_neg_dist = neg_dists[semi_hard_mask].min()
                    loss_val = (
                        hardest_pos_dist - semi_hard_neg_dist + self.margin
                    )
                else:
                    # No valid triplets at all for this anchor
                    loss_val = torch.tensor(0.0, device=device)

            losses.append(F.relu(loss_val))

        if not losses:
            logger.warning(
                "No valid triplets found in batch. Returning zero loss."
            )
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon).

    All same-class samples in batch serve as positives.
    Output is always non-negative.

    Args:
        temperature: Scaling temperature (default 0.07, range [0.01, 1.0]).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            features: Shape (B, D), L2-normalized.
            labels: Shape (B,).

        Returns:
            Scalar loss >= 0.
        """
        device = features.device
        batch_size = features.size(0)

        # Edge case: fewer than 2 distinct classes
        unique_classes = labels.unique()
        if unique_classes.numel() < 2:
            logger.warning(
                "Batch has fewer than 2 distinct classes (%d). "
                "Returning zero loss.",
                unique_classes.numel(),
            )
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create masks
        labels_col = labels.unsqueeze(0)  # (1, B)
        labels_row = labels.unsqueeze(1)  # (B, 1)
        positive_mask = (labels_row == labels_col).float()  # (B, B)

        # Remove diagonal (self-similarity)
        identity = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - identity

        # For numerical stability, subtract max from logits
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Mask out self-connections for the denominator
        exp_logits = torch.exp(logits) * (1 - identity)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Compute mean log-likelihood over positive pairs
        # Only compute for anchors that have at least one positive
        num_positives = positive_mask.sum(dim=1)
        mask_valid = num_positives > 0

        if not mask_valid.any():
            logger.warning(
                "No valid positive pairs found in batch. Returning zero loss."
            )
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (
            num_positives + 1e-12
        )

        # Loss is negative mean log-likelihood (only for valid anchors)
        loss = -mean_log_prob[mask_valid].mean()

        return loss


class NTXentLoss(nn.Module):
    """NT-Xent loss for self-supervised SimCLR training.

    Treats paired views as positives, all other 2(B-1) as negatives.
    Output is always non-negative.

    Args:
        temperature: Scaling temperature (default 0.5, range [0.05, 1.0]).
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z_i: First view projections, shape (B, projection_dim).
            z_j: Second view projections, shape (B, projection_dim).

        Returns:
            Scalar loss >= 0.
        """
        device = z_i.device
        batch_size = z_i.size(0)

        # L2 normalize
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # Concatenate both views: shape (2B, D)
        z = torch.cat([z_i, z_j], dim=0)
        n_samples = 2 * batch_size

        # Compute full similarity matrix (2B x 2B)
        similarity_matrix = torch.matmul(z, z.T) / self.temperature

        # Create positive pair mask
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_mask = torch.zeros(n_samples, n_samples, device=device)
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = 1.0
            pos_mask[i + batch_size, i] = 1.0

        # Remove self-similarity from denominator
        identity = torch.eye(n_samples, device=device)

        # For numerical stability
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Denominator: sum over all negatives (everything except self)
        exp_logits = torch.exp(logits) * (1 - identity)
        log_denominator = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Numerator: positive pair similarity
        log_prob = logits - log_denominator

        # Mean over positive pairs
        loss = -(pos_mask * log_prob).sum() / pos_mask.sum()

        return loss


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    Down-weights easy examples and focuses on hard-to-classify samples.
    Particularly effective for minority classes (Loc, Scratch) where the
    model is over-confident on majority classes.

    Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0, range [0.5, 5.0]).
            Higher gamma means stronger focus on hard examples.
        alpha: Per-class weight tensor or None for uniform weights.
            If class_counts is provided, alpha is computed from inverse-frequency.
        class_counts: Per-class sample counts for automatic alpha computation.
        num_classes: Number of classes (default 9).
        reduction: 'mean' or 'sum' (default 'mean').
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        class_counts: Optional[list[int]] = None,
        num_classes: int = 9,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        # Compute alpha from class_counts if provided
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        elif class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
            inv_freq = 1.0 / counts
            # Normalize so weights sum to num_classes
            alpha_tensor = inv_freq / inv_freq.sum() * num_classes
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.register_buffer("alpha", None)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Shape (B, num_classes), raw model outputs.
            labels: Shape (B,), integer class targets.

        Returns:
            Scalar loss >= 0.
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Gather the probability of the true class: p_t
        # labels shape (B,) -> (B, 1) for gather
        labels_one_hot = labels.unsqueeze(1)
        p_t = probs.gather(1, labels_one_hot).squeeze(1)  # (B,)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Log probability (with numerical stability)
        log_p_t = torch.log(p_t + 1e-8)

        # Apply per-class alpha weighting if available
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[labels]  # (B,)
            loss = -alpha_t * focal_weight * log_p_t
        else:
            loss = -focal_weight * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with optional inverse-frequency class weights.

    Args:
        class_counts: Per-class sample counts for weight computation.
        num_classes: Number of classes (default 9).
        enable_weights: Whether to use class weights (default True).
    """

    def __init__(
        self,
        class_counts: Optional[list[int]] = None,
        num_classes: int = 9,
        enable_weights: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.enable_weights = enable_weights

        weight = None
        if class_counts is not None and enable_weights:
            # Compute inverse-frequency weights, then normalize
            counts = torch.tensor(class_counts, dtype=torch.float32)
            # Avoid division by zero for classes with 0 samples
            counts = counts.clamp(min=1.0)
            inv_freq = 1.0 / counts
            # Normalize so weights sum to num_classes (standard practice)
            weight = inv_freq / inv_freq.sum() * num_classes

        self.register_buffer("weight", weight)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted cross-entropy.

        Args:
            logits: Shape (B, num_classes).
            labels: Shape (B,), integer targets.

        Returns:
            Scalar loss >= 0.
        """
        return F.cross_entropy(logits, labels, weight=self.weight)
