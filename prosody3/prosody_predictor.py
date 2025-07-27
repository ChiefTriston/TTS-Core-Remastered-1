
#!/usr/bin/env python3
"""
ProsodyPredictorV15: Predicts prosodic features (F0, energy, pitch variance, speech rate, pause duration, MFCCs) from mel-spectrograms.

All defaults and hyperparameters are loaded from config.prosody.json located in the same folder.

Constructor kwargs (all from CFG["predictor"]):
  mel_dim (int): Mel-spectrogram dimension
  cond_dim (int): Conditioning dimension
  n_layers (int): Number of S4 layers
  use_amp (bool): Whether to use automatic mixed precision
  n_freq (int): Number of frequency bins for positional embedding
  sample_rate (int): Audio sample rate
  n_mels (int): Number of mel bins
  window_size (int): FFT window size
  hop_length (int): Hop length for mel-spectrogram

Methods:
  forward(mel: Tensor[B, n_mels, T]) -> Dict[str, Tensor]:
    Returns dict with keys:
      "f0": Tensor[B, T] (mean pitch)
      "energy": Tensor[B, T]
      "pitch_var": Tensor[B, T] (pitch variance)
      "speech_rate": Tensor[B, 1] (syllables per second)
      "pause_dur": Tensor[B, 1] (average pause duration)
      "mfcc": Tensor[B, 13] (13 MFCC coefficients)
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
import torchaudio
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
        mel_dim:    int  = P.get("mel_dim", 80),
        cond_dim:   int  = P["cond_dim"],
        n_layers:   int  = P["n_layers"],
        use_amp:    bool = P.get("use_amp", False),
        n_freq:     int  = P["n_freq"],
        sample_rate: int = P.get("sample_rate", 22050),
        n_mels:     int  = P.get("n_mels", 80),
        window_size: int = P.get("window_size", 1024),
        hop_length:  int = P.get("hop_length", 256)
    ):
        super().__init__()
        self.use_amp = use_amp and torch.cuda.is_available()
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Positional frequency embedding
        self.freq_emb = FreqPosEmbed(n_freq=n_freq, dim=cond_dim)
        # Project mel-spectrogram to conditioning dimension
        self.input_proj = nn.Linear(mel_dim, cond_dim)

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
        self.pitch_var_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 1)
        )
        self.speech_rate_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 1)
        )
        self.pause_dur_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 1)
        )
        self.mfcc_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 13)  # 13 MFCC coefficients
        )

        # MelSpectrogram for validation
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=window_size, hop_length=hop_length, n_mels=n_mels
        )

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: Tensor of shape [B, n_mels, T]

        Returns:
            dict with:
              "f0": Tensor[B, T]
              "energy": Tensor[B, T]
              "pitch_var": Tensor[B, T]
              "speech_rate": Tensor[B, 1]
              "pause_dur": Tensor[B, 1]
              "mfcc": Tensor[B, 13]
        """
        with autocast(enabled=self.use_amp):
            # mel: [B, 80, T]  -> [B, T, 80] so Linear sees 80
            x = mel.transpose(1, 2)                 # [B, T, 80]
            x = self.input_proj(x)                  # [B, T, 256]

            freq = self.freq_emb().to(x.device)     # [n_freq, 256]
            freq = freq.unsqueeze(0).expand(x.size(0), -1, -1)        # [B, n_freq, 256]
            if freq.size(1) != x.size(1):
                if freq.size(1) > x.size(1):
                    freq = freq[:, :x.size(1)]
                else:
                    pad_len = x.size(1) - freq.size(1)
                    pad = freq[:, -1:].repeat(1, pad_len, 1)
                    freq = torch.cat([freq, pad], dim=1)
            h = x + freq                                           # [B, T, 256]

            # Process through S4 layers (they expect [B, cond_dim, T])
            for s4 in self.s4_layers:
                h = h.transpose(1, 2)   # [B, T, cond_dim] -> [B, cond_dim, T]
                h = s4(h)
                h = h.transpose(1, 2)   # back to [B, T, cond_dim]

            # Predict prosodic features
            f0_pred = self.f0_head(h).squeeze(-1)  # [B, T]
            energy_pred = self.energy_head(h).squeeze(-1)  # [B, T]
            pitch_var = self.pitch_var_head(h).squeeze(-1)  # [B, T]
            speech_rate = self.speech_rate_head(h.mean(dim=1)).unsqueeze(1)  # [B, 1]
            pause_dur = self.pause_dur_head(h.mean(dim=1)).unsqueeze(1)  # [B, 1]
            mfcc = self.mfcc_head(h.mean(dim=1))  # [B, 13]

        return {
            "f0": f0_pred,
            "energy": energy_pred,
            "pitch_var": pitch_var,
            "speech_rate": speech_rate,
            "pause_dur": pause_dur,
            "mfcc": mfcc
        }