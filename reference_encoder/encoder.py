#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Wav2Vec2Model
except ImportError:
    Wav2Vec2Model = None

try:
    from res2net import Res2NetBlock
except ImportError:
    Res2NetBlock = None

try:
    from espnet2.asr.frontend.conformer import ConformerLayer
except ImportError:
    ConformerLayer = None

# Guard ECAPA import
try:
    from speechbrain.lobes.models.ecapa_tdnn import ECAPA_TDNN
except ImportError:
    ECAPA_TDNN = None


class StatsPooling(nn.Module):
    def forward(self, x):
        # x: (B, T, D)
        return torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)


class MultiHeadAttentivePooling(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            ) for _ in range(heads)
        ])
        self.fuse = nn.Linear(heads * dim, dim)

    def forward(self, x):
        # x: (B, T, D)
        outs = []
        for head in self.attn:
            w = F.softmax(head(x), dim=1)       # (B, T, 1)
            outs.append((x * w).sum(dim=1))     # (B, D)
        return self.fuse(torch.cat(outs, dim=1))  # (B, D)


class ReferenceEncoder(nn.Module):
    """
    Produces a fixed-dimensional speaker embedding from raw audio or mel inputs.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # backbone for feature extraction
        if cfg.backbone == 'wav2vec2':
            if Wav2Vec2Model is None:
                raise ImportError("Please install transformers for wav2vec2 support")
            # load SSL model onto CPU and enable gradient checkpointing
            self.ssl = Wav2Vec2Model.from_pretrained(cfg.wav2vec2_name).to('cpu')
            self.ssl.gradient_checkpointing_enable()
            # freeze SSL weights
            for p in self.ssl.parameters():
                p.requires_grad = False

            self.proj_input = nn.Linear(self.ssl.config.hidden_size, cfg.speaker_dim)
            self.pooling = lambda h: h.mean(dim=1)

        elif cfg.backbone == 'res2net':
            if Res2NetBlock is None:
                raise ImportError("Install res2net-pytorch for res2net support")
            self.backbone = nn.Sequential(
                nn.Conv1d(cfg.n_mels, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                Res2NetBlock(64, scale=4),
                nn.Conv1d(64, cfg.speaker_dim, 1)
            )

        elif cfg.backbone == 'conformer':
            if ConformerLayer is None:
                raise ImportError("Install espnet-model-zoo for conformer support")
            layers = [
                ConformerLayer(dim_model=cfg.n_mels, num_heads=4, dim_ff=256)
                for _ in range(4)
            ]
            self.backbone = nn.Sequential(*layers)

        elif cfg.backbone == 'ecapa_tdnn':
            if ECAPA_TDNN is None:
                raise ImportError("Install speechbrain>=0.5 for ecapa_tdnn support")
            self.backbone = ECAPA_TDNN(channels=1024, lin_neurons=cfg.speaker_dim)

        else:
            raise ValueError(f"Unknown backbone '{cfg.backbone}'")

        # pooling layer for mel branch
        if cfg.pooling == 'self_attentive':
            self.pool = nn.Sequential(
                nn.Linear(cfg.speaker_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        elif cfg.pooling == 'multi_head_attentive':
            self.pool = MultiHeadAttentivePooling(cfg.speaker_dim)
        else:
            self.pool = StatsPooling()
            self.pool_proj = nn.Linear(cfg.speaker_dim * 2, cfg.speaker_dim)

        # final projection
        self.proj = nn.Sequential(
            nn.Linear(cfg.speaker_dim, cfg.speaker_dim),
            nn.BatchNorm1d(cfg.speaker_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.speaker_dim, cfg.speaker_dim)
        )

    def forward(self, x):
        """
        x: either raw waveform (B, N_samples) or mel (B, n_mels, T)
        """
        if self.cfg.backbone == "wav2vec2":
            # raw waveform branch → run the entire SSL call on CPU
            wav = x if x.dim() == 2 else x.squeeze(1)        # (B, N) on GPU
            device = wav.device
            wav_cpu = wav.detach().cpu()                     # move to CPU
            with torch.no_grad():
                hssl = self.ssl(wav_cpu).last_hidden_state   # (B, L, H) on CPU
            hssl = hssl.to(device)                           # back to GPU
            pooled = self.pooling(hssl)                      # (B, H) on GPU
            emb = self.proj_input(pooled)                    # (B, speaker_dim) on GPU
        else:
            # mel-spectrogram branch (pad_collate ensures x is (B, n_mels, T))
            mel = x
            h = self.backbone(mel)                           # (B, speaker_dim, T) or (B, speaker_dim)
            if h.dim() == 3:
                h = self.pool(h.transpose(1, 2))             # (B, speaker_dim)
                if hasattr(self, 'pool_proj'):
                    h = self.pool_proj(h)
            emb = h                                         # (B, speaker_dim)

        emb = self.proj(emb)                                # (B, speaker_dim)
        return F.normalize(emb, p=2, dim=1)                 # unit-length embeddings
