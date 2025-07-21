# vocoder/stft.py

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from .config import GANConfig

class LearnableSTFT(nn.Module):
    """
    Learnable filterbank for complex STFT.
    """
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Hann window
        self.register_buffer('window', torch.hann_window(n_fft))
        # Learnable magnitude filterbank
        self.filterbank = nn.Parameter(torch.randn(n_fft // 2 + 1))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, 1, T]
        # Compute complex STFT
        spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            power=None
        )(wav.squeeze(1))  # [B, freq, frames, 2]
        real, imag = spec.unbind(-1)
        mag = torch.sqrt(real**2 + imag**2)
        # Apply learnable filterbank
        return mag * self.filterbank.unsqueeze(-1)

class STFTLoss(nn.Module):
    """
    Multi-resolution STFT loss with learnable STFT modules.
    """
    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.stfts = nn.ModuleList([
            LearnableSTFT(n_fft, cfg.hop_length)
            for n_fft in cfg.stft_sizes
        ])
        self.lambda_stft = cfg.lambda_stft

    def forward(self, wav_fake: torch.Tensor, wav_real: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for stft in self.stfts:
            mag_fake = stft(wav_fake)
            mag_real = stft(wav_real)
            loss += F.l1_loss(mag_fake, mag_real)
        return loss * self.lambda_stft
