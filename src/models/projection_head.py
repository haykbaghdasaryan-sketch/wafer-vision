"""ProjectionHead MLP for metric learning and self-supervised training.

Architecture: Linear(D→512) → BatchNorm1d → ReLU → Linear(512→128) → L2Normalize

The projection head maps backbone embeddings to a lower-dimensional space
where contrastive losses (Triplet, SupCon, NT-Xent) are computed. Final
embeddings for downstream tasks are extracted from the backbone output
(before the projection head).
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProjectionHead(nn.Module):
    """MLP projection head for metric/contrastive learning.

    Maps backbone embeddings (B, D) to L2-normalized projections (B, output_dim).
    Output vectors always have unit L2 norm.

    Architecture:
        Linear(D → hidden_dim) → BatchNorm1d → ReLU → Linear(hidden_dim → output_dim) → L2Normalize

    Args:
        input_dim: Dimensionality of backbone embeddings (D).
        hidden_dim: Hidden layer size. Default 512.
        output_dim: Output projection size. Default 128.

    Example:
        >>> head = ProjectionHead(input_dim=2048)
        >>> x = torch.randn(32, 2048)
        >>> z = head(x)
        >>> z.shape
        torch.Size([32, 128])
        >>> torch.allclose(z.norm(dim=1), torch.ones(32), atol=1e-5)
        True
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project and L2-normalize backbone embeddings.

        Args:
            x: Backbone embedding tensor, shape (B, D).

        Returns:
            L2-normalized projection tensor, shape (B, output_dim).
            Each row vector has unit norm (||v||₂ = 1).
        """
        projected = self.net(x)
        return F.normalize(projected, p=2, dim=1)
