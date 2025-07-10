
import torch
import torch.nn as nn
import math

class FreqPosEmbed(nn.Module):
    def __init__(self, n_freq: int, dim: int):
        super().__init__()
        self.dim = dim
        freq_pos = torch.arange(n_freq).unsqueeze(1).float()  # (n_freq, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))  # (dim/2,)

        pe = torch.zeros(n_freq, dim)
        pe[:, 0::2] = torch.sin(freq_pos * div_term)  # even dims
        pe[:, 1::2] = torch.cos(freq_pos * div_term)  # odd dims

        self.register_buffer('pe', pe)  # non-trainable buffer

    def forward(self):
        """
        Returns:
            Tensor of shape (n_freq, dim) with positional encodings.
        """
        return self.pe
