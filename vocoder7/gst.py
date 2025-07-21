# vocoder/gst.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GANConfig

class GlobalStyleTokens(nn.Module):
    """
    Global Style Tokens module for learning style embeddings via attention.
    """
    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.cfg = cfg
        # Token embeddings: [num_tokens, style_dim]
        self.tokens = nn.Parameter(torch.randn(cfg.num_style_tokens, cfg.style_dim))
        # Attention network: to attend tokens over mel_ref
        self.attn_conv = nn.Sequential(
            nn.Conv1d(cfg.channels, cfg.style_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cfg.style_dim, cfg.num_style_tokens, kernel_size=1)
        )

    def forward(self, mel_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_ref: Refined mel-spectrogram [B, channels, T]
        Returns:
            style: Style embedding [B, style_dim]
        """
        # [B, num_tokens, T]
        logits = self.attn_conv(mel_ref)
        weights = F.softmax(logits, dim=-1)  
        # [B, style_dim] = batch matmul weights @ tokens
        style = torch.einsum('bnt,nd->bd', weights, self.tokens)
        return style
