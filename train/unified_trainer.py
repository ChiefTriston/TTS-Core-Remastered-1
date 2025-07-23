```python
# unified_trainer.py

import logging
import contextlib
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from acoustic.model import AcousticConfig, AcousticModel
from sde_refiner.config import ScoreSDEConfig
from sde_refiner.refiner import ScoreSDERefinerV15
from vocoder.config import GANConfig
from vocoder.trainer import VocoderTrainer
from vocoder.losses import compute_gan_loss

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@contextlib.contextmanager
def eval_mode(*models):
    """Context manager to temporarily switch models to eval mode."""
    prev = [m.training for m in models]
    try:
        for m in models:
            m.eval()
        yield
    finally:
        for m, was_train in zip(models, prev):
            m.train(was_train)

class SchedulerCallback:
    """Callback to step schedulers at epoch end."""
    def on_epoch_end(self, trainer, epoch_idx, metrics):
        trainer.sch_ac.step()
        trainer.sch_rf.step()
        # Vocoder schedulers if present
        if hasattr(trainer.vocoder_trainer, 'sch_g'):
            trainer.vocoder_trainer.sch_g.step()
            trainer.vocoder_trainer.sch_d.step()

class UnifiedTrainer:
    """
    Unified trainer for AcousticModel, ScoreSDERefinerV15, and BigVGAN Vocoder.

    Callbacks must implement:
      - on_step_end(trainer, step, metrics: Dict[str, float])
      - on_epoch_end(trainer, epoch_idx, metrics: List[Dict[str, float]])
      - on_validation_end(trainer, val_metrics: Dict[str, float])
    """
    def __init__(
        self,
        acoustic_cfg: AcousticConfig,
        refine_cfg: ScoreSDEConfig,
        gan_cfg: GANConfig,
        device: str = 'cuda',
        callbacks: list = None
    ):
        self.device = device
        # Initialize callbacks, include SchedulerCallback by default
        self.callbacks = [SchedulerCallback()]
        if callbacks:
            self.callbacks += callbacks

        # Instantiate core models
        self.acoustic = AcousticModel(acoustic_cfg).to(device)
        self.refiner = ScoreSDERefinerV15(acoustic_cfg, refine_cfg).to(device)
        self.vocoder_trainer = VocoderTrainer(gan_cfg, device=device)
        # Extract generator and discriminators
        gen = self.vocoder_trainer.g
        d_mpd = self.vocoder_trainer.d_mpd
        d_msd = self.vocoder_trainer.d_msd
        d_mbd = self.vocoder_trainer.d_mbd

        # Optional DDP & torch.compile
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        if dist.is_available() and dist.is_initialized():
            # Compile models first
            try:
                self.acoustic = torch.compile(self.acoustic)
                self.refiner = torch.compile(self.refiner)
                gen = torch.compile(gen)
            except Exception:
                pass
            # Wrap
            self.acoustic = DDP(self.acoustic, device_ids=[torch.cuda.current_device()])
            self.refiner = DDP(self.refiner, device_ids=[torch.cuda.current_device()])
            gen = DDP(gen, device_ids=[torch.cuda.current_device()])
            d_mpd = DDP(d_mpd, device_ids=[torch.cuda.current_device()])
            d_msd = DDP(d_msd, device_ids=[torch.cuda.current_device()])
            d_mbd = DDP(d_mbd, device_ids=[torch.cuda.current_device()])

        # Assign back wrapped modules
        self.vocoder_trainer.g = gen
        self.vocoder_trainer.d_mpd = d_mpd
        self.vocoder_trainer.d_msd = d_msd
        self.vocoder_trainer.d_mbd = d_mbd

        # Optimizers
        self.opt_ac = optim.AdamW(self.acoustic.parameters(), lr=acoustic_cfg.lr)
        self.opt_rf = optim.AdamW(self.refiner.parameters(), lr=refine_cfg.lr)

        # Single AMP scaler
        self.scaler = GradScaler()

        # Learning rate schedulers (epoch-level via callbacks)
        self.sch_ac = optim.lr_scheduler.OneCycleLR(
            self.opt_ac,
            max_lr=acoustic_cfg.lr,
            total_steps=acoustic_cfg.scheduler_steps,
            pct_start=0.1
        )
        self.sch_rf = optim.lr_scheduler.OneCycleLR(
            self.opt_rf,
            max_lr=refine_cfg.lr,
            total_steps=refine_cfg.scheduler_steps,
            pct_start=0.1
        )

    def _to_device(self, batch: tuple) -> tuple:
        return tuple(
            b.to(self.device) if torch.is_tensor(b) else b
            for b in batch
        )

    def _step_acoustic(
        self,
        text_emb: Tensor,
        prosody: Tensor,
        style_id: Tensor,
        speaker: Tensor,
        mel_tgt: Tensor
    ) -> tuple[Tensor, float]:
        self.acoustic.train()
        self.opt_ac.zero_grad()
        with autocast():
            mel_pred, *_ = self.acoustic(text_emb, prosody, style_id, speaker)
            loss = F.l1_loss(mel_pred, mel_tgt)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt_ac)
        torch.nn.utils.clip_grad_norm_(self.acoustic.parameters(), 1.0)
        self.scaler.step(self.opt_ac)
        self.scaler.update()
        return mel_pred.detach(), loss.item()

    def _step_refiner(
        self,
        text_emb: Tensor,
        prosody: Tensor,
        style_id: Tensor,
        speaker: Tensor,
        mel_pred: Tensor,
        vader: Tensor
    ) -> tuple[Tensor, float]:
        self.refiner.train()
        self.opt_rf.zero_grad()
        with autocast():
            mel_ref, _, _ = self.refiner(text_emb, prosody, style_id, speaker, vader_scores=vader)
            loss = F.l1_loss(mel_ref, mel_pred)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt_rf)
        torch.nn.utils.clip_grad_norm_(self.refiner.parameters(), 1.0)
        self.scaler.step(self.opt_rf)
        self.scaler.update()
        return mel_ref.detach(), loss.item()

    def _step_vocoder(
        self,
        text_emb: Tensor,
        prosody: Tensor,
        style_id: Tensor,
        speaker: Tensor,
        mel_ref: Tensor,
        wav_real: Tensor,
        vader: Tensor,
        step: int
    ) -> dict[str, float]:
        try:
            # Ensure train mode
            self.vocoder_trainer.g.train()
            self.vocoder_trainer.d_mpd.train()
            self.vocoder_trainer.d_msd.train()
            self.vocoder_trainer.d_mbd.train()
            return self.vocoder_trainer.training_step((text_emb, prosody, style_id, speaker, mel_ref, wav_real, vader), refiner=None, step=step)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                logger.warning('OOM in vocoder step')
                return {}
            raise

    def training_step(self, batch: tuple, step: int) -> dict[str, float]:
        text_emb, prosody, style_id, speaker, mel_tgt, wav_real, vader = self._to_device(batch)

        # Acoustic
        mel_pred, loss_ac = self._step_acoustic(text_emb, prosody, style_id, speaker, mel_tgt)
        logger.info(f"Step {step} | L_acoustic: {loss_ac:.4f}")

        # Refiner
        mel_ref, loss_rf = self._step_refiner(text_emb, prosody, style_id, speaker, mel_pred, vader)
        logger.info(f"Step {step} | L_refiner: {loss_rf:.4f}")

        # Vocoder
        voc_stats = self._step_vocoder(text_emb, prosody, style_id, speaker, mel_ref, wav_real, vader, step)
        logger.info(f"Step {step} | Vocoder: {voc_stats}")

        metrics = {'loss_acoustic': loss_ac, 'loss_refiner': loss_rf, **voc_stats}
        for cb in self.callbacks:
            cb.on_step_end(self, step, metrics)
        return metrics

    def train_epoch(self, dataloader, epoch_idx: int) -> dict[str, float]:
        metrics = [self.training_step(batch, epoch_idx * len(dataloader) + i) for i, batch in enumerate(dataloader)]
        for cb in self.callbacks:
            cb.on_epoch_end(self, epoch_idx, metrics)
        avg = {k: sum(m.get(k, 0.0) for m in metrics) / len(metrics) for k in metrics[0]}
        logger.info(f"Epoch {epoch_idx} avg: {avg}")
        return avg

    def validation_step(self, batch) -> dict[str, float]:
        text_emb, prosody, style_id, speaker, mel_tgt, wav_real, vader = self._to_device(batch)
        models = [self.acoustic, self.refiner, self.vocoder_trainer.g,
                  self.vocoder_trainer.d_mpd, self.vocoder_trainer.d_msd, self.vocoder_trainer.d_mbd]
        with torch.no_grad(), eval_mode(*models):
            mel_pred, *_ = self.acoustic(text_emb, prosody, style_id, speaker)
            mel_ref, _, _ = self.refiner(text_emb, prosody, style_id, speaker, vader_scores=vader)
            wav_fake = self.vocoder_trainer.g(mel_ref.transpose(1,2), prosody, style_id, speaker=speaker, emotion_probs=None)
            mpd_out, mpd_feat = self.vocoder_trainer.d_mpd(wav_fake)
            msd_out, msd_feat = self.vocoder_trainer.d_msd(wav_fake)
            mbd_out, mbd_feat = self.vocoder_trainer.d_mbd(wav_fake)
            g_loss, d_loss = compute_gan_loss(mpd_out, mpd_feat, msd_out, msd_feat, mbd_out, mbd_feat, wav_fake, wav_real, None, None, None, None, cfg=self.vocoder_trainer.cfg, step=0)
        val_stats = {
            'val_loss_acoustic': F.l1_loss(mel_pred, mel_tgt).item(),
            'val_loss_refiner': F.l1_loss(mel_ref, mel_pred).item(),
            'val_loss_vocoder_g': g_loss.item(),
            'val_loss_vocoder_d': d_loss.item()
        }
        for cb in self.callbacks:
            cb.on_validation_end(self, val_stats)
        return val_stats
```
