import torch
import torch.nn as nn

class BetaScheduler(nn.Module):
    """
    Learnable beta(t) schedule via small MLP.

    Maps a scalar diffusion time t to a beta weight in [0,1].
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape [B,1], diffusion time in [0,1].
        Returns:
            Tensor of shape [B,1], beta(t) in [0,1].
        """
        return self.net(t)
