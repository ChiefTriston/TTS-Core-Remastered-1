# train/engine/loop.py
from typing import Dict
import torch
from torch.cuda.amp import autocast
from ..blocks.refiner import RefinerBlock
from ..blocks.vocoder import VocoderBlock
from ..utils.tensor_ops import move_to_device
import logging

logger = logging.getLogger(__name__)

class TrainingLoop:
    """Manages training and validation loops, including gradient accumulation."""
    def __init__(self, trainer: 'UnifiedTrainer'):
        self.trainer = trainer
        self.config = trainer.config
        self.blocks = trainer.blocks
        self.callbacks = trainer.callbacks
        self.state = trainer.state
        self.device = trainer.device

    def run(self) -> None:
        """Runs the full training loop."""
        for callback in self.callbacks:
            callback.on_train_start(self.trainer)
        while self.state.global_step < self.config.trainer.max_steps:
            for callback in self.callbacks:
                callback.on_epoch_start(self.trainer)
            metrics = {}
            for batch in self.trainer.train_loader:
                saved_this_step = False
                if self.state.global_step >= self.config.trainer.max_steps:
                    break
                batch = move_to_device(batch, self.device)
                metrics = self.train_step(batch)
                for callback in self.callbacks:
                    callback.on_batch_end(self.state.global_step, metrics, trainer=self.trainer)
                self.state.global_step += 1
                if self.state.global_step % self.config.trainer.val_freq == 0:
                    val_metrics = self._run_validation()
                    for callback in self.callbacks:
                        callback.on_val_end(self.state.global_step, val_metrics, trainer=self.trainer)
                    if val_metrics:  # Skip checkpoint if no validation metrics
                        self.trainer.save_checkpoint(self.state.global_step, val_metrics)
                        saved_this_step = True
                    if 'refiner' in self.blocks and isinstance(self.blocks['refiner'], RefinerBlock):
                        self.blocks['refiner'].update_noise_schedule(val_metrics.get('val_l1_loss', float('inf')))
                if self.state.global_step % self.config.trainer.checkpoint_freq == 0 and not saved_this_step:
                    self.trainer.save_checkpoint(self.state.global_step, metrics)
            for callback in self.callbacks:
                callback.on_epoch_end(self.state.global_step, metrics, trainer=self.trainer)
        for callback in self.callbacks:
            callback.on_train_end(self.trainer)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executes a single training step with gradient accumulation.
        Modulo checks use pre-increment global_step, aligned with +1 logic in run()."""
        if 'mel_gt' not in batch:
            raise ValueError("batch['mel_gt'] is required for training")
        if 'vocoder' in self.blocks and 'wav_gt' not in batch:
            raise ValueError("batch['wav_gt'] is required when vocoder is enabled")
        metrics = {}
        batch = self._pre_forward_hook(batch)
        mel_pred = None
        mel_ref = None

        if 'acoustic' in self.blocks:
            with autocast(enabled=self.config.acoustic.amp):
                out_ac = self.blocks['acoustic'].forward(batch)
                losses_ac = self.blocks['acoustic'].loss(out_ac, batch)
            self.blocks['acoustic'].scaler().scale(losses_ac['l1_loss']).backward()
            if (self.state.global_step + 1) % self.config.trainer.grad_accum_steps == 0:
                self.blocks['acoustic'].step(self.config.trainer.grad_clip_norm)
                for scheduler in self.blocks['acoustic'].schedulers():
                    scheduler.step()
            metrics.update(self.blocks['acoustic'].metrics(out_ac, batch))
            mel_pred = out_ac['mel_pred']

        if 'refiner' in self.blocks and (self.state.global_step + 1) % self.config.refiner.update_freq == 0:
            with autocast(enabled=self.config.refiner.amp):
                out_rf = self.blocks['refiner'].forward(batch, mel_pred=mel_pred or batch['mel_gt'])
                losses_rf = self.blocks['refiner'].loss(out_rf, batch)
            self.blocks['refiner'].scaler().scale(losses_rf['total_loss']).backward()
            if (self.state.global_step + 1) % self.config.trainer.grad_accum_steps == 0:
                self.blocks['refiner'].step(self.config.trainer.grad_clip_norm)
                for scheduler in self.blocks['refiner'].schedulers():
                    scheduler.step()
            metrics.update(self.blocks['refiner'].metrics(out_rf, batch))
            mel_ref = out_rf['mel_ref']

        if 'vocoder' in self.blocks and self.state.global_step >= self.config.vocoder.freeze_until:
            with autocast(enabled=self.config.vocoder.amp):
                d_metrics = {}
                for _ in range(self.blocks['vocoder'].gan_d_steps):
                    d_out = self.blocks['vocoder'].disc_step(mel_ref or mel_pred or batch['mel_gt'], batch['wav_gt'], self.blocks['vocoder'].scaler(), self.config.vocoder.amp)
                    for k, v in d_out['metrics'].items():
                        d_metrics[k] = d_metrics.get(k, 0.0) + v / self.blocks['vocoder'].gan_d_steps
                g_out = self.blocks['vocoder'].gen_step(mel_ref or mel_pred or batch['mel_gt'], batch['wav_gt'], self.blocks['vocoder'].scaler(), self.config.vocoder.amp)
                for scheduler in self.blocks['vocoder'].schedulers():
                    scheduler.step()
            metrics.update(d_metrics)
            metrics.update(g_out['metrics'])
            if isinstance(self.blocks['vocoder'], VocoderBlock):
                self.blocks['vocoder'].update_gan_ratio(self.state.global_step)

        return metrics

    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executes a single validation step."""
        if 'mel_gt' not in batch:
            raise ValueError("batch['mel_gt'] is required for validation")
        if 'vocoder' in self.blocks and 'wav_gt' not in batch:
            raise ValueError("batch['wav_gt'] is required when vocoder is enabled")
        metrics = {}
        batch = self._pre_forward_hook(batch)
        mel_pred = None
        mel_ref = None

        with torch.no_grad():
            if 'acoustic' in self.blocks:
                with autocast(enabled=self.config.acoustic.amp):
                    out_ac = self.blocks['acoustic'].forward(batch)
                    metrics.update(self.blocks['acoustic'].metrics(out_ac, batch))
                    mel_pred = out_ac['mel_pred']
            
            if 'refiner' in self.blocks:
                with autocast(enabled=self.config.refiner.amp):
                    out_rf = self.blocks['refiner'].forward(batch, mel_pred=mel_pred or batch['mel_gt'])
                    metrics.update(self.blocks['refiner'].metrics(out_rf, batch))
                    mel_ref = out_rf['mel_ref']
            
            if 'vocoder' in self.blocks:
                with autocast(enabled=self.config.vocoder.amp):
                    out_voc = self.blocks['vocoder'].forward(batch, mel_ref=mel_ref or mel_pred or batch['mel_gt'])
                    metrics.update(self.blocks['vocoder'].metrics(out_voc, batch))
        
        return metrics

    def _run_validation(self) -> Dict[str, float]:
        """Runs validation loop, prefixing all metrics with 'val_'."""
        metrics = {}
        count = 0
        for callback in self.callbacks:
            if hasattr(callback, 'swap_to_ema') and self.config.trainer.novel.ema_swap_validate.enabled:
                callback.swap_to_ema()
        try:
            for batch in self.trainer.val_loader:
                batch = move_to_device(batch, self.device)
                batch_metrics = self.val_step(batch)
                count += 1
                for k, v in batch_metrics.items():
                    metrics[k] = metrics.get(k, 0.0) + v
            if count:
                for k in metrics:
                    metrics[k] /= count
            else:
                logger.warning("Validation loader is empty; skipping validation metrics")
                return {}
        finally:
            for callback in self.callbacks:
                if hasattr(callback, 'swap_to_train') and self.config.trainer.novel.ema_swap_validate.enabled:
                    callback.swap_to_train()
        return {f"val_{k}": v for k, v in metrics.items()}

    def _pre_forward_hook(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Applies observer pre-forward hook if enabled."""
        if self.trainer.observer:
            for stage in ['acoustic', 'refiner', 'vocoder']:
                batch = self.trainer.observer.pre_forward(stage, batch)
        return batch