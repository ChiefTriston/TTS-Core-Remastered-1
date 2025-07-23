# train/blocks/vocoder.py
from typing import Dict
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from ..adapters.vocoder_adapter import vocoder_adapter
from vocoder.trainer import VocoderTrainer
from ..config_schemas import UnifiedTrainerConfig
from .base import TrainBlock
import logging
import os

logger = logging.getLogger(__name__)

class VocoderBlock(TrainBlock):
    """Vocoder block with GAN-based waveform generation."""
    def __init__(self, config: UnifiedTrainerConfig, device: torch.device):
        model = vocoder_adapter(VocoderTrainer()).to(device)
        if config.vocoder.model_path and os.path.exists(config.vocoder.model_path):
            try:
                model.load_state_dict(torch.load(config.vocoder.model_path, map_location=device))
            except Exception as e:
                logger.warning(f"Failed to load vocoder model: {e}")
        optim_g = AdamW(model.generator.parameters(), lr=config.vocoder.optim_g.lr, betas=config.vocoder.optim_g.betas)
        optim_d = AdamW(model.discriminators.parameters(), lr=config.vocoder.optim_d.lr, betas=config.vocoder.optim_d.betas)
        scheduler_g = OneCycleLR(optim_g, max_lr=config.vocoder.scheduler_g.max_lr, total_steps=config.trainer.max_steps, pct_start=config.vocoder.scheduler_g.pct_start)
        scheduler_d = OneCycleLR(optim_d, max_lr=config.vocoder.scheduler_d.max_lr, total_steps=config.trainer.max_steps, pct_start=config.vocoder.scheduler_d.pct_start)
        scaler = GradScaler(enabled=config.vocoder.amp)
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
        super().__init__(model, [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, device)
        self.dynamic_gan = config.trainer.novel.dynamic_gan.enabled
        self.d_loss_ema = 0.0
        self.g_loss_ema = 0.0
        self.gan_d_steps = config.vocoder.gan_d_steps
        self.gan_g_steps = config.vocoder.gan_g_steps
        self.ema_alpha = config.trainer.novel.dynamic_gan.ema_alpha
        self._cached_metrics = {}

    def forward(self, batch: Dict[str, torch.Tensor], mel_ref: torch.Tensor, **cond) -> Dict[str, torch.Tensor]:
        self._cached_metrics = {}
        with autocast(enabled=self.scaler().enabled):
            wav_fake = self.model.generator(mel_ref)
            disc_out = self.model.discriminators(wav_fake)
        return {'wav_fake': wav_fake, 'disc_out': disc_out}

    def disc_step(self, mel_ref: torch.Tensor, wav_gt: torch.Tensor, scaler: GradScaler, use_amp: bool) -> Dict:
        """Discriminator step, updates every batch regardless of grad_accum_steps for GAN stability."""
        self._cached_metrics = {}
        with autocast(enabled=use_amp):
            wav_fake = self.model.generator(mel_ref)
            assert wav_fake.shape == wav_gt.shape, f"Mismatch: wav_fake {wav_fake.shape}, wav_gt {wav_gt.shape}"
            disc_out_fake = self.model.discriminators(wav_fake.detach())
            disc_out_real = self.model.discriminators(wav_gt)
            d_loss = self.model.compute_discriminator_loss(disc_out_fake, disc_out_real)
            metrics = self.model.compute_discriminator_metrics(disc_out_fake, disc_out_real)
        scaler.scale(d_loss).backward()
        scaler.unscale_(self.optimizers()[1])
        torch.nn.utils.clip_grad_norm_(self.model.discriminators.parameters(), 1.0)
        scaler.step(self.optimizers()[1])
        scaler.update()
        self.optimizers()[1].zero_grad(set_to_none=True)
        if self.dynamic_gan:
            self.d_loss_ema = self.ema_alpha * self.d_loss_ema + (1 - self.ema_alpha) * d_loss.item()
        return {'metrics': metrics, 'd_loss': d_loss}

    def gen_step(self, mel_ref: torch.Tensor, wav_gt: torch.Tensor, scaler: GradScaler, use_amp: bool) -> Dict:
        """Generator step, updates every batch regardless of grad_accum_steps for GAN stability."""
        self._cached_metrics = {}
        with autocast(enabled=use_amp):
            wav_fake = self.model.generator(mel_ref)
            disc_out = self.model.discriminators(wav_fake)
            g_loss, stft_loss, fm_loss = self.model.compute_generator_loss(wav_fake, wav_gt, disc_out)
            total_loss = g_loss + stft_loss + fm_loss
        scaler.scale(total_loss).backward()
        scaler.unscale_(self.optimizers()[0])
        torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), 1.0)
        scaler.step(self.optimizers()[0])
        scaler.update()
        self.optimizers()[0].zero_grad(set_to_none=True)
        if self.dynamic_gan:
            self.g_loss_ema = self.ema_alpha * self.g_loss_ema + (1 - self.ema_alpha) * g_loss.item()
        metrics = {'g_loss': g_loss.item(), 'stft_loss': stft_loss.item(), 'fm_loss': fm_loss.item()}
        self._cached_metrics = metrics
        return {'metrics': metrics, 'g_loss': g_loss}

    def metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self._cached_metrics:
            return self._cached_metrics
        with torch.no_grad():
            wav_gt = batch['wav_gt'].to(self.device)
            g_loss, stft_loss, fm_loss = self.model.compute_generator_loss(outputs['wav_fake'], wav_gt, outputs['disc_out'])
            return {'g_loss': g_loss.item(), 'stft_loss': stft_loss.item(), 'fm_loss': fm_loss.item()}

    def update_gan_ratio(self, step: int) -> None:
        """Adjusts GAN steps based on EMA-smoothed losses."""
        if (not self.dynamic_gan) or (step % 100 != 0):
            return
        if self.d_loss_ema > self.g_loss_ema * 1.5 and self.gan_d_steps < 5:
            self.gan_d_steps = min(self.gan_d_steps + 1, 5)
            self.gan_g_steps = max(self.gan_g_steps - 1, 1)
        elif self.g_loss_ema > self.d_loss_ema * 1.5 and self.gan_g_steps < 5:
            self.gan_g_steps = min(self.gan_g_steps + 1, 5)
            self.gan_d_steps = max(self.gan_d_steps - 1, 1)
        logger.info(f"Adjusted GAN steps: D={self.gan_d_steps}, G={self.gan_g_steps}")