# 5_sde_refiner/blocks/tf_block.py

import torch
import torch.nn as nn

class TFBlock(nn.Module):
    """
    Temporal-Feature Transformer block with dual attention.
    """
    def __init__(self, ch: int, heads: int = 4, dim_ff: int = 512):
        super().__init__()
        self.t_attn = nn.MultiheadAttention(embed_dim=ch, num_heads=heads, batch_first=True)
        self.f_attn = nn.MultiheadAttention(embed_dim=ch, num_heads=heads, batch_first=True)
        self.ff     = nn.Sequential(
            nn.LayerNorm(ch),
            nn.Linear(ch, dim_ff), nn.GELU(),
            nn.Linear(dim_ff, ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        # Time-attention:
        t = x.permute(0, 2, 1)                  # [B, T, C]
        t_out, _ = self.t_attn(t, t, t)         # [B, T, C]
        t_out = t_out.permute(0, 2, 1)          # [B, C, T]

        # Feature-attention (across channels):
        f = x.permute(0, 2, 1)                  # [B, T, C]
        f_out, _ = self.f_attn(f, f, f)         # [B, T, C]
        f_out = f_out.permute(0, 2, 1)          # [B, C, T]

        # Fuse + feed-forward:
        h = t_out + f_out                       # [B, C, T]
        ff_out = self.ff(h.permute(0, 2, 1))    # [B, T, C]
        ff_out = ff_out.permute(0, 2, 1)        # [B, C, T]

        return x + ff_out

