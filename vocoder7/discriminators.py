# vocoder/discriminators.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GANConfig

class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) for waveform realism.
    """
    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.cfg = cfg
        self.discriminators = nn.ModuleList()
        for period in cfg.disc_periods:
            layers = []
            ch = 1
            # Conv2d stack for each period
            for _ in range(4):
                layers.append(
                    nn.utils.spectral_norm(
                        nn.Conv2d(ch, ch * 4, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))
                    )
                )
                layers.append(nn.LeakyReLU(0.2))
                ch *= 4
            layers.append(
                nn.utils.spectral_norm(
                    nn.Conv2d(ch, 1, kernel_size=(3, 1), padding=(1, 0))
                )
            )
            self.discriminators.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Waveform [B, 1, T]
        Returns:
            outputs: List of discriminator scores for each period
            features: List of lists of intermediate feature maps
        """
        outputs, features = [], []
        B, C, T = x.shape
        for p, disc in zip(self.cfg.disc_periods, self.discriminators):
            if T % p != 0:
                pad = p - (T % p)
                x_p = F.pad(x, (0, pad))
            else:
                x_p = x
            # reshape to [B, 1, T//p, p]
            x_r = x_p.view(B, 1, x_p.size(2) // p, p)
            feat_maps = []
            out = x_r
            for layer in disc:
                out = layer(out)
                feat_maps.append(out)
            outputs.append(out)
            # exclude final output from features
            features.append(feat_maps[:-1])
        return outputs, features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD) for waveform realism.
    """
    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.cfg = cfg
        self.discriminators = nn.ModuleList()
        for ks in cfg.disc_kernel_sizes:
            layers = []
            ch = 1
            for i in range(5):
                stride = 2 if i < 3 else 1
                layers.append(
                    nn.utils.spectral_norm(
                        nn.Conv1d(ch, ch * 4, kernel_size=ks, stride=stride, padding=ks // 2)
                    )
                )
                layers.append(nn.LeakyReLU(0.2))
                ch *= 4
            layers.append(
                nn.utils.spectral_norm(
                    nn.Conv1d(ch, 1, kernel_size=3, padding=1)
                )
            )
            self.discriminators.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Waveform [B, 1, T]
        Returns:
            outputs: List of discriminator scores for each scale
            features: List of lists of intermediate feature maps
        """
        outputs, features = [], []
        scales = [x, F.avg_pool1d(x, 4, 2, 1), F.avg_pool1d(x, 4, 2, 1)]
        for disc, s in zip(self.discriminators, scales):
            out_feats = []
            out = s
            for layer in disc:
                out = layer(out)
                out_feats.append(out)
            outputs.append(out)
            features.append(out_feats[:-1])
        return outputs, features


class MultiBandDiscriminator(nn.Module):
    """
    Multi-Band Discriminator (MBD) for sub-band waveform realism.
    """
    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.cfg = cfg
        self.discriminators = nn.ModuleList()
        for _ in range(cfg.num_bands):
            layers = []
            ch = 1
            for _ in range(4):
                layers.append(
                    nn.utils.spectral_norm(
                        nn.Conv1d(ch, ch * 4, kernel_size=15, stride=2, padding=7)
                    )
                )
                layers.append(nn.LeakyReLU(0.2))
                ch *= 4
            layers.append(
                nn.utils.spectral_norm(
                    nn.Conv1d(ch, 1, kernel_size=3, padding=1)
                )
            )
            self.discriminators.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Waveform [B, 1, T]
        Returns:
            outputs: List of discriminator scores per band
            features: List of lists of intermediate feature maps
        """
        outputs, features = [], []
        bands = torch.chunk(x, self.cfg.num_bands, dim=2)
        for band, disc in zip(bands, self.discriminators):
            out_feats = []
            out = band.squeeze(1) if band.dim() == 3 else band
            out = out.unsqueeze(1)
            for layer in disc:
                out = layer(out)
                out_feats.append(out)
            outputs.append(out)
            features.append(out_feats[:-1])
        return outputs, features
