import torch
import torch.nn as nn
import math

class FreqPosEmbed(nn.Module):
    """
    Frequency-position embedding along channel dimension.
    """
    def __init__(self, n_freq: int, dim: int):
        super().__init__()
        freq = torch.arange(n_freq, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000) / dim))
        pe = torch.zeros(n_freq, dim)
        pe[:, 0::2] = torch.sin(freq * div)
        pe[:, 1::2] = torch.cos(freq * div)
        self.register_buffer('pe', pe)

    def forward(self) -> torch.Tensor:
        # returns [n_freq, dim]
        return self.pe