
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .config import AcousticConfig

def stochastic_depth(x: Tensor, p: float, training: bool) -> Tensor:
    if not training or p <= 0.0:
        return x
    keep = 1 - p
    mask = x.new_empty((x.size(0),) + (1,) * (x.dim() - 1)).bernoulli_(keep) / keep
    return x * mask

class ScaleNorm(nn.Module):
    """ScaleNorm: normalize to unit norm then scale by learnable gain."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return x * (self.g / norm)

class ResidualConvBlock(nn.Module):
    """
    One residual block:
      - Input ScaleNorm
      - Causal depthwise + pointwise conv
      - Output ScaleNorm + activation
      - FiLM conditioning
      - Dropout, LayerScale, stochastic depth
    """
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        dropout: float,
        kernel_size: int,
        sd_prob: float,
        ls_init: float
    ):
        super().__init__()
        pad = kernel_size - 1
        self.dw = nn.Conv1d(channels, channels, kernel_size, groups=channels, padding=pad)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm1 = ScaleNorm(channels)
        self.norm2 = ScaleNorm(channels)
        self.film = nn.Sequential(
            nn.Linear(cond_dim, channels), nn.SiLU(), nn.Linear(channels, 2 * channels)
        )
        self.gamma = nn.Parameter(ls_init * torch.ones(channels))
        self.sd_prob = sd_prob
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        res = x
        x_t = x.transpose(1,2)
        x_n = self.norm1(x_t).transpose(1,2)
        y = self.dw(x_n)[..., :res.size(2)]
        y = self.pw(y)
        y_t = y.transpose(1,2)
        y_t = self.norm2(y_t)
        y_t = self.act(y_t)
        scale, shift = self.film(cond).chunk(2, dim=-1)
        y_t = y_t * (1 + scale) + shift
        y = self.drop(y_t).transpose(1,2)
        y = self.gamma.view(1, -1, 1) * y
        y = stochastic_depth(y, self.sd_prob, self.training)
        return res + y