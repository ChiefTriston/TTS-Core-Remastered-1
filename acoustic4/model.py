import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .config import AcousticConfig
from .blocks import ResidualConvBlock

# Try import fused CUDA operator
try:
    from torch.ops.my_uv import fused_conv_film
except ImportError:
    fused_conv_film = None

class AcousticModel(nn.Module):
    """
    Acoustic TTS model: conv residual blocks + FiLM.
    forward(text_emb, f0, energy, speaker) -> [B, T, 80]
    """
    def __init__(self, cfg: AcousticConfig):
        super().__init__()
        self.cfg = cfg
        total_cond = cfg.cond_dim + (cfg.speaker_dim if cfg.speaker_dim>0 else 0)
        in_ch = cfg.text_emb_dim + total_cond
        self.input_proj = nn.Conv1d(in_ch, cfg.hidden_channels, 1)
        self.blocks = nn.ModuleList([
            ResidualConvBlock(
                cfg.hidden_channels,
                total_cond,
                cfg.dropout,
                cfg.kernel_size,
                sd_prob=cfg.base_sd_prob * (i+1)/cfg.num_layers,
                ls_init=cfg.layer_scale_init
            ) for i in range(cfg.num_layers)
        ])
        self.output_proj = nn.Conv1d(cfg.hidden_channels, 80, 1)

    def forward(
        self,
        text_emb: Tensor,
        f0: Tensor,
        energy: Tensor,
        speaker: Optional[Tensor] = None
    ) -> Tensor:
        B, T, _ = text_emb.shape
        parts = [text_emb.transpose(1,2), f0.unsqueeze(1), energy.unsqueeze(1)]
        cond_parts = [f0.unsqueeze(-1), energy.unsqueeze(-1)]
        if self.cfg.speaker_dim > 0 and speaker is not None:
            spk_t = speaker.unsqueeze(2).expand(-1, -1, T)
            parts.append(spk_t)
            cond_parts.append(spk_t.transpose(1,2))
        x = torch.cat(parts, dim=1)
        cond = torch.cat(cond_parts, dim=-1)
        h = self.input_proj(x)
        for blk in self.blocks:
            # use fused op if available
            h = blk(h, cond)
        mel = self.output_proj(h).transpose(1,2)
        if self.cfg.profile:
            flops = 2 * mel.numel() * self.cfg.hidden_channels / 1e9
            print(f"[profile] GFLOPs: {flops:.3f}")
        return mel
