#!/usr/bin/env python3
"""
ProsodyPredictorV15: Predicts f0 and energy from text embeddings.

All defaults and hyperparameters are loaded from config.prosody.json located in the same folder.
Includes a sanity-check stub under __main__.

Constructor kwargs (all from CFG["predictor"]):
  emb_dim (int): Text embedding dimension
  cond_dim (int): Conditioning dimension
  n_layers (int): Number of S4 layers
  use_amp (bool): Whether to use automatic mixed precision
  n_freq (int): Number of frequency bins for positional embedding

Methods:
  forward(text_emb: Tensor[B, T, emb_dim]) -> Dict[str, Tensor]:
    Returns dict with keys:
      "f0": Tensor[B, T]
      "energy": Tensor[B, T]
"""
import os
import sys
# ensure our local 3_prosody folder is on the path exactly once
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from s4 import S4
from freq_pos_embed import FreqPosEmbed

# Load hyperparameters from JSON
CFG_PATH = Path(__file__).parent.joinpath("config.prosody.json")
CFG      = json.load(CFG_PATH.open())
P        = CFG["predictor"]

class ProsodyPredictorV15(nn.Module):
    def __init__(
        self,
        emb_dim:  int  = P["emb_dim"],
        cond_dim: int  = P["cond_dim"],
        n_layers: int  = P["n_layers"],
        use_amp:  bool = P.get("use_amp", False),
        n_freq:   int  = P["n_freq"]
    ):
        super().__init__()
        self.use_amp = use_amp and torch.cuda.is_available()

        # Positional frequency embedding
        self.freq_emb = FreqPosEmbed(n_freq=n_freq, dim=cond_dim)
        # Project text embeddings to conditioning dimension
        self.input_proj = nn.Linear(emb_dim, cond_dim)

        # Stacked S4 layers with JSON-specified parameters
        s4_cfg = CFG.get("s4", {})
        self.s4_layers = nn.ModuleList([
            S4(d_model=cond_dim, **s4_cfg)
            for _ in range(n_layers)
        ])

        # Prediction heads
        self.f0_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 1)
        )
        self.energy_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 1)
        )

    def forward(self, text_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_emb: Tensor of shape [B, T, emb_dim]

        Returns:
            dict with:
              "f0": Tensor[B, T]
              "energy": Tensor[B, T]
        """
        with autocast(enabled=self.use_amp):
            x    = self.input_proj(text_emb)                              # [B, T, C]
            freq = self.freq_emb().to(x.device)                           # [n_freq, C]
            freq = freq.unsqueeze(0).permute(0,2,1).expand(x.size(0), -1, -1)
            T = x.size(1)
            if freq.size(2) != T:
                if freq.size(2) > T:
                    freq = freq[:, :, :T]
                else:
                    pad = freq[:, :, -1:].repeat(1,1, T - freq.size(2))
                    freq = torch.cat([freq, pad], dim=2)
            freq = freq.permute(0,2,1)                                    # [B, T, C]
            h    = x + freq                                               # [B, T, C]

            # —— HERE’S THE ONLY CHANGE ——                
            # Each S4 wants [B, C, T] → [B, C, T], so we permute before+after:
            for s4 in self.s4_layers:
                h = h.permute(0,2,1)   # [B, T, C] → [B, C, T]
                h = s4(h)
                h = h.permute(0,2,1)   # [B, C, T] → [B, T, C]

            f0_pred     = self.f0_head(h).squeeze(-1)
            energy_pred = self.energy_head(h).squeeze(-1)

        return {"f0": f0_pred, "energy": energy_pred}

# Sanity check stub
if __name__ == "__main__":
    import torch
    pp    = ProsodyPredictorV15()
    dummy = torch.randn(2, 100, P["emb_dim"])
    out   = pp(dummy)
    assert set(out.keys()) == {"f0", "energy"}
    assert out["f0"].shape    == (2, 100)
    print("✅ ProsodyPredictorV15 smoke-test passed")

