# vocoder/trainer.py

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from .config import GANConfig
from .generator import Generator
from .discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiBandDiscriminator
from .stft import STFTLoss
from .losses import compute_gan_loss
from .gst import GlobalStyleTokens

class VocoderTrainer:
    def __init__(self, cfg: GANConfig, device='cuda'):
        self.cfg = cfg
        self.device = device

        # Models
        self.g = Generator(cfg).to(device)
        self.d_mpd = MultiPeriodDiscriminator(cfg).to(device)
        self.d_msd = MultiScaleDiscriminator(cfg).to(device)
        self.d_mbd = MultiBandDiscriminator(cfg).to(device)
        self.gst = GlobalStyleTokens(cfg).to(device)

        # Optimizers
        self.opt_g = optim.Adam(self.g.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.opt_d = optim.Adam(
            list(self.d_mpd.parameters()) +
            list(self.d_msd.parameters()) +
            list(self.d_mbd.parameters()), lr=1e-4, betas=(0.9, 0.999)
        )

        # Schedulers
        self.sch_g = optim.lr_scheduler.CosineAnnealingLR(self.opt_g, T_max=200000)
        self.sch_d = optim.lr_scheduler.CosineAnnealingLR(self.opt_d, T_max=200000)

        # Mixed precision
        self.scaler = GradScaler()

        # Loss modules
        self.stft_loss = STFTLoss(cfg)

        # EMA
        self.ema_g = Generator(cfg).to(device)
        self.ema_decay = 0.999
        self._init_ema()

    def _init_ema(self):
        self.ema_g.load_state_dict(self.g.state_dict())
        for p in self.ema_g.parameters():
            p.requires_grad_(False)

    def update_ema(self):
        for ema_p, p in zip(self.ema_g.parameters(), self.g.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def training_step(self, batch, refiner=None, step=0):
        """
        batch: (text_emb, prosody, style_id, speaker, mel_target, wav_real, vader_scores)
        refiner: optional ScoreSDERefinerV15 to produce mel_ref
        step: global step counter
        """
        text_emb, prosody, style_id, speaker, mel_target, wav_real, vader_scores = batch

        # 1) Generate mel_ref via refiner if provided
        if refiner:
            with torch.no_grad():
                mel_ref, _, _ = refiner(text_emb, prosody, style_id, speaker, mel_target, vader_scores)
        else:
            mel_ref = mel_target

        # 2) Style embedding
        style = self.gst(mel_ref.transpose(1,2).to(self.device))
        # 3) Generator forward
        with autocast():
            wav_fake = self.g(
                mel_ref.transpose(1,2).to(self.device),
                prosody.to(self.device),
                style,
                vader_scores if vader_scores is not None else torch.zeros_like(style),
                style_drop=(torch.rand(1).item()<self.cfg.dropout_prob),
                emo_drop=(torch.rand(1).item()<self.cfg.dropout_prob),
                w_style=1.0, w_emo=1.0
            )
        # 4) Discriminator forward on fake
        mpd_out, mpd_feat = self.d_mpd(wav_fake)
        msd_out, msd_feat = self.d_msd(wav_fake)
        mbd_out, mbd_feat = self.d_mbd(wav_fake)
        # 5) Discriminator forward on real
        mpd_out_r, mpd_feat_r = self.d_mpd(wav_real)
        msd_out_r, msd_feat_r = self.d_msd(wav_real)
        mbd_out_r, mbd_feat_r = self.d_mbd(wav_real)

        # 6) Compute losses (no speaker model)
        g_loss, d_loss = compute_gan_loss(
            mpd_out, mpd_feat,
            msd_out, msd_feat,
            mbd_out, mbd_feat,
            wav_fake, wav_real,
            predicted_pitch=None, true_pitch=None,
            predicted_duration=None, true_duration=None,
            cfg=self.cfg, step=step
        )

        # 7) Backprop discriminator
        self.opt_d.zero_grad()
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.opt_d)
        self.scaler.update()

        # 8) Backprop generator
        self.opt_g.zero_grad()
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.opt_g)
        self.scaler.update()

        # 9) EMA & schedulers
        self.update_ema()
        self.sch_g.step()
        self.sch_d.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
        }
