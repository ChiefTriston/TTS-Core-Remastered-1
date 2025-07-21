# vocoder/generator.py

import torch
import torch.nn as nn
from .config import GANConfig
from .residual import ResidualBlock
from .attention import SelfAttention

class Generator(nn.Module):
    """
    BigVGAN-style multi-band waveform synthesizer.
    """
    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.cfg = cfg
        # Split the mel-spectrogram into bands
        band_size = cfg.channels // cfg.num_bands
        self.band_split = nn.ModuleList([
            nn.Conv1d(band_size, cfg.hidden_dim, kernel_size=7, padding=3)
            for _ in range(cfg.num_bands)
        ])
        # Conditioning projections
        self.cond_prosody = nn.Sequential(
            nn.Linear(18, cfg.cond_dim // 2), nn.SiLU(),
            nn.Linear(cfg.cond_dim // 2, cfg.cond_dim)
        )
        self.style_proj = nn.Linear(cfg.style_dim, cfg.cond_dim)
        self.emotion_proj = nn.Linear(6, cfg.cond_dim)
        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        ch = cfg.hidden_dim
        for i, factor in enumerate(cfg.upsample_factors):
            block = nn.ModuleList()
            # Transposed convolution to upsample
            block.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=factor * 2,
                                   stride=factor, padding=factor // 2)
            )
            # Residual blocks with GLU + FiLM
            for dilation in cfg.res_dilations:
                block.append(ResidualBlock(ch // 2, dilation, cfg.cond_dim))
            # Insert self-attention halfway
            if i == len(cfg.upsample_factors) // 2:
                block.append(SelfAttention(ch // 2))
            self.upsample_blocks.append(block)
            ch //= 2
        # Merge bands back to single-channel waveform
        self.band_merge = nn.Conv1d(ch * cfg.num_bands, 1, kernel_size=7, padding=3)

    def forward(
        self,
        mel: torch.Tensor,        # [B, channels, T]
        prosody: torch.Tensor,    # [B, T, 18]
        style: torch.Tensor,      # [B, style_dim]
        emotion: torch.Tensor,    # [B, 6]
        style_drop: bool = False,
        emo_drop: bool = False,
        w_style: float = 1.0,
        w_emo: float = 1.0
    ) -> torch.Tensor:
        """
        Synthesize multi-band waveform from mel-spectrogram.
        """
        # Conditioning features
        c_pros = self.cond_prosody(prosody)  # [B, T, cond_dim]
        c_sty = self.style_proj(style).unsqueeze(1) * w_style
        if style_drop:
            c_sty = torch.zeros_like(c_sty)
        c_emo = self.emotion_proj(emotion).unsqueeze(1) * w_emo
        if emo_drop:
            c_emo = torch.zeros_like(c_emo)
        cond = c_pros + c_sty + c_emo  # [B, T, cond_dim]
        cond = cond.transpose(1, 2)    # [B, cond_dim, T]

        # Split into bands and apply initial conv
        B, C, T = mel.shape
        band_size = C // self.cfg.num_bands
        bands = [
            self.band_split[i](mel[:, i*band_size:(i+1)*band_size, :])
            for i in range(self.cfg.num_bands)
        ]

        # Upsample each band
        outputs = []
        for x in bands:
            for block in self.upsample_blocks:
                x = block[0](x)
                for layer in block[1:]:
                    if isinstance(layer, ResidualBlock):
                        x = layer(x, cond)
                    else:
                        x = layer(x)
            outputs.append(x)

        # Merge bands and tanh
        x_cat = torch.cat(outputs, dim=1)
        wav = self.band_merge(x_cat)
        return torch.tanh(wav)
