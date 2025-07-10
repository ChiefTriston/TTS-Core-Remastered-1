#!/usr/bin/env python3
# sde_refiner5/model.py

import json
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchdiffeq import odeint
from torch.nn.utils import spectral_norm

# 1) Bring in our config loader and dataclass
from .config import ScoreSDEConfig, load_score_sde_config

# 2) Local utilities
from .scheduler           import BetaScheduler
from .band_split_merge    import BandSplitMerge
from .sde_freq_pos_embed  import FreqPosEmbed
from .cnf_path            import CNFPath

# 3) Re-exported building blocks
from .blocks import RevBlock, TFBlock, GumbelMoE, HarmonicSourceFilter
from .blocks.hier_vq import HierVQ
from .blocks.hsf     import HSFLayer

# 4) Import the acoustic & S4 components via pipeline sys.path
from acoustic.model import AcousticModel
from s4             import S4


class ScoreSDERefinerV15(nn.Module):
    """
    State-of-the-art Score-SDE refiner for TTS.
    """
    def __init__(
        self,
        acoustic_cfg: AcousticModel.__init__.__annotations__['cfg'],
        cfg: ScoreSDEConfig
    ):
        super().__init__()

        # a) Step-4 acoustic model
        self.acoustic = AcousticModel(acoustic_cfg)

        self.cfg     = cfg
        self.profile = cfg.profile
        total_dim    = cfg.cnf_dim  # this is your actual channel count (e.g., 80)

        # ─── Voice-cloning blocks ────────────────────────────
        self.hsf = HSFLayer(
            channels     = total_dim,
            hidden       = cfg.hsflayer_hidden,
            layers       = cfg.hsflayer_layers,
            kernel_size  = cfg.hsflayer_kernel
        )

        # Determine VQ dims vs. total_dim
        dims = cfg.vq_dims
        if sum(dims) != total_dim:
            dims = [total_dim]
        codes = cfg.vq_codes if isinstance(cfg.vq_codes, list) else [cfg.vq_codes]
        if len(codes) != len(dims):
            codes = [codes[0]] * len(dims)

        # 6) Source-filter + VQ
        self.vq = HierVQ(dims, codes)
        # ──────────────────────────────────────────────────────

        # b) Band split / merge
        self.split = BandSplitMerge(input_dim=total_dim, bands=cfg.bands)
        self.merge = BandSplitMerge(input_dim=total_dim, bands=cfg.bands)

        # c) Time-dependent beta scheduler
        self.beta_sched = BetaScheduler(cfg.beta_hidden)

        # d) Conditioning projections
        self.cond_f0 = nn.Sequential(
            nn.Linear(1, cfg.cond_dim // 2), nn.SiLU(),
            nn.Linear(cfg.cond_dim // 2, cfg.cond_dim)
        )
        self.cond_en = nn.Sequential(
            nn.Linear(1, cfg.cond_dim // 2), nn.SiLU(),
            nn.Linear(cfg.cond_dim // 2, cfg.cond_dim)
        )
        self.style_proj = nn.Linear(cfg.style_dim, cfg.cond_dim)

        # Infer text_emb_dim from acoustic_cfg
        text_dim = getattr(acoustic_cfg, "text_emb_dim", None)
        if text_dim is None:
            raise ValueError("text_emb_dim missing in acoustic_cfg")
        self.seg_proj = nn.Linear(text_dim, cfg.cond_dim)

        self.freq_emb = FreqPosEmbed(n_freq=total_dim, dim=cfg.cond_dim)

        # Learned PE fusion per band
        self.pe_proj = nn.ModuleList([
            nn.Linear(cfg.cond_dim * band_size, cfg.cond_dim)
            for band_size in cfg.bands
        ])

        # e) Per-band networks
        self.nets = nn.ModuleList()
        for band_size in cfg.bands:
            layers = []
            ch = band_size + cfg.cond_dim
            for _ in range(cfg.levels):
                layers.append(RevBlock(S4(ch, **cfg.s4)))
                layers.append(RevBlock(GumbelMoE(ch, ch * 2, style_dim=cfg.style_dim)))
                layers.append(TFBlock(ch * 2, heads=cfg.s4['heads'], dim_ff=cfg.cond_dim))
                ch *= 2
            layers.append(RevBlock(S4(ch, **cfg.s4)))
            for _ in range(cfg.levels):
                layers.append(RevBlock(nn.ConvTranspose1d(ch, ch // 2, 4, 2, 1)))
                layers.append(TFBlock(ch // 2, heads=cfg.s4['heads'], dim_ff=cfg.cond_dim))
                layers.append(RevBlock(S4(ch // 2, **cfg.s4)))
                ch //= 2
            layers.append(RevBlock(spectral_norm(nn.Conv1d(ch, band_size, 3, padding=1))))
            self.nets.append(nn.ModuleList(layers))

    def forward(
        self,
        text_emb: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
        speaker: Optional[torch.Tensor],
        style: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) Coarse acoustic mel (B,C,T)
        mel0 = self.acoustic(text_emb, f0, energy, speaker)
        x = mel0.transpose(1, 2)
        B, C, T = x.shape

        # 2) Diffusion time & beta(t)
        if t is None:
            t = torch.rand(B, 1, device=x.device)
        beta = self.beta_sched(t)

        # 3) Conditioning vectors
        c_f0  = self.cond_f0(f0.unsqueeze(-1))
        c_eng = self.cond_en(energy.unsqueeze(-1))
        c_sty = self.style_proj(style).unsqueeze(1)
        c_seg = self.seg_proj(text_emb.mean(dim=1)).unsqueeze(1)
        cond  = c_f0 + c_eng + c_sty + c_seg

        # 4) Frequency positional encodings
        pe = self.freq_emb().to(x.device)
        pe = pe.unsqueeze(0).expand(B, -1, -1)
        pe = pe.unsqueeze(2).expand(-1, -1, T, -1)

        # 5) Band-wise processing
        outs, offset = [], 0
        for i, band in enumerate(self.split.split(x)):
            bsz = band.size(1)
            pe_blk   = pe[:, offset:offset + bsz, :, :]
            pe_flat  = pe_blk.permute(0, 2, 3, 1).reshape(B, T, self.cfg.cond_dim * bsz)
            pe_slice = self.pe_proj[i](pe_flat).permute(0, 2, 1)
            y = torch.cat([band, pe_slice], dim=1)
            for layer in self.nets[i]:
                y = layer(y, cond=cond, style=style) if isinstance(layer, RevBlock) else layer(y)
            outs.append(y)
            offset += bsz

        merged = self.merge.merge(outs)

        # 6) Source-filter + VQ
        hn      = self.hsf(merged)
        z       = merged + beta.view(B, 1, 1) * hn
        zq, vql = self.vq(z)

        # if the time-axis changed, restore original length T
        if zq.size(2) != T:
            zq = F.interpolate(zq, size=T, mode="linear", align_corners=False)

        # transpose back to (B, T, C) for pipeline
        mel_ref = zq.transpose(1, 2)
        return mel_ref, t, vql, None, None

