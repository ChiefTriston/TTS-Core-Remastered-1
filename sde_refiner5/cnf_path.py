import torch
import torch.nn as nn
from torch import Tensor

class CNFPath(nn.Module):
    """
    Simple Continuous‐Normalizing‐Flow drift network.
    Takes (t, x), outputs dx/dt for ODE integration.
    """

    def __init__(self, dim: int):
        super().__init__()
        # A small MLP: input x ∈ R^dim → hidden 2*dim → output dim
        self.drift = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            t:   Tensor of shape [B] or [B,1], the current time—ignored here.
            x:   Tensor of shape [B, T, C] or [B*T, C]; we'll treat last dim as features.
        Returns:
            Tensor of same shape as x, representing the drift vector field.
        """
        # The ODE solver will pass x as shape [B, T, C].
        # We want to apply the same MLP to each time‐step/channel vector.
        # First, flatten out the time dimension if needed.
        orig_shape = x.shape
        if x.dim() == 3:
            # x: [B, T, C] → reshape to [B*T, C]
            B, T, C = x.shape
            flat = x.contiguous().view(B * T, C)
            d_flat = self.drift(flat)
            return d_flat.view(B, T, C)
        else:
            # x is already [B*T, C] or [N, C]
            return self.drift(x)
