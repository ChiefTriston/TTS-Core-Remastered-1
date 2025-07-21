# vocoder/config.py

from dataclasses import dataclass
from typing import List

@dataclass
class GANConfig:
    """
    Configuration for BigVGAN-style vocoder.
    """
    channels: int = 80              # number of mel-spectrogram channels
    cond_dim: int = 128             # conditioning vector dimension
    style_dim: int = 128            # style embedding dimension
    num_bands: int = 4              # number of frequency bands
    upsample_factors: List[int] = None  # upsampling factors per stage
    res_dilations: List[int] = None     # dilations for residual blocks
    disc_periods: List[int] = None      # periods for MultiPeriodDiscriminator
    disc_kernel_sizes: List[int] = None # kernel sizes for MultiScaleDiscriminator
    sr: int = 22050                # sampling rate
    hop_length: int = 256          # hop length for mel-spectrogram
    stft_sizes: List[int] = None       # FFT sizes for STFT loss
    num_style_tokens: int = 10     # number of Global Style Tokens
    dropout_prob: float = 0.1      # classifier-free guidance dropout
    r1_gamma: float = 10.0         # R1 regularization weight
    r1_interval: int = 16          # apply R1 every N steps
    lambda_stft: float = 2.0       # STFT loss weight
    lambda_pitch: float = 1.0      # pitch consistency loss weight
    lambda_dur: float = 1.0        # duration consistency loss weight

    def __post_init__(self):
        if self.upsample_factors is None:
            self.upsample_factors = [8, 8, 2, 2]
        if self.res_dilations is None:
            self.res_dilations = [1, 3, 5]
        if self.disc_periods is None:
            self.disc_periods = [2, 3, 5, 7, 11]
        if self.disc_kernel_sizes is None:
            self.disc_kernel_sizes = [15, 41, 41]
        if self.stft_sizes is None:
            self.stft_sizes = [512, 1024, 2048]
