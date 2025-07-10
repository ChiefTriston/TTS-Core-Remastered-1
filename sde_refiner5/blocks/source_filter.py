# 5_sde_refiner/blocks/source_filter.py

import torch
import torch.nn as nn

class HarmonicSourceFilter(nn.Module):
    """
    Simple harmonic+noise source‐filter module.
    Splits mel into harmonic/noise, applies conv filters, and blends via gating.
    """
    def __init__(self, ch: int = 80, filt_ch: int = 64):
        super().__init__()
        self.harm = nn.Sequential(
            nn.Conv1d(ch, filt_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(filt_ch, ch,  kernel_size=3, padding=1),
        )
        self.noise = nn.Sequential(
            nn.Conv1d(ch, filt_ch, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(filt_ch, ch,  kernel_size=5, padding=2),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [B, C, T]
        returns: [B, C, T]
        """
        h = self.harm(mel)
        n = self.noise(mel)
        gate = torch.sigmoid(h)
        return h * gate + n * (1 - gate)
