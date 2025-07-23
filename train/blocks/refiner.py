# train/blocks/refiner.py
from typing import Dict
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sde_refiner.refiner import ScoreSDERefinerV15
from ..config_schemas import UnifiedTrainerConfig
from .base import TrainBlock
import logging
import os

logger = logging.getLogger(__name__)

class RefinerBlock(TrainBlock):
    """Score-SDE refiner block for mel spectrogram enhancement."""
    def __init__(self, config: UnifiedTrainerConfig, device: torch.device):
        model = ScoreSDERefinerV15().to(device)
        if config.refiner.model_path and os.path.exists(config.refiner.model_path):
            try:
                model.load_state_dict(torch.load(config.refiner.model_path, map_location=device))
            except Exception as e:
                logger.warning(f"Failed to load refiner model: {e}")
        optimizer = AdamW(model.parameters(), lr=config.refiner.optim.lr, betas=config.refiner.optim.betas)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.refiner.scheduler.T_max)
        scaler = GradScaler(enabled=config.refiner.amp)
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
        super().__init__(model, [optimizer], [scheduler], scaler, device)
        self.noise_annealing = config.trainer.novel.sde_noise_annealing.enabled
        self.noise_schedule = config.trainer.novel.sde_noise_annealing.initial_sigma
        self.l1_weight = config.trainer.novel.sde_noise_annealing.initial_l1_weight
        self.l1_plateau_count = 0
        self.prev_val_l1 = float('inf')
        self.sde_loss_missing_warned = False

    def forward(self, batch: Dict[str, torch.Tensor], mel_pred: torch.Tensor, **cond) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.scaler().enabled):
            mel_ref = self.model(mel_pred, batch.get('prosody_stats'), batch.get('speaker_id'), batch.get('style_id'), batch.get('vader'), sigma=self.noise_schedule if self.noise_annealing else None)
        return {'mel_ref': mel_ref}

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mel_gt = batch['mel_gt'].to(self.device)
        assert outputs['mel_ref'].shape == mel_gt.shape, f"Mismatch: mel_ref {outputs['mel_ref'].shape}, mel_gt {mel_gt.shape}"
        l1_loss = torch.nn.functional.l1_loss(outputs['mel_ref'], mel_gt)
        sde_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.model, 'sde_loss'):
            sde_loss = self.model.sde_loss(outputs['mel_ref'], mel_gt)
        elif not self.sde_loss_missing_warned:
            logger.warning("Model does not support sde_loss; using L1 loss only")
            self.sde_loss_missing_warned = True
        total_loss = l1_loss * self.l1_weight + sde_loss * (1.0 - self.l1_weight)
        return {'l1_loss': l1_loss, 'sde_loss': sde_loss, 'total_loss': total_loss}

    def metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        mel_gt = batch['mel_gt'].to(self.device)
        l1_loss = torch.nn.functional.l1_loss(outputs['mel_ref'], mel_gt)
        sde_loss = 0.0
        if hasattr(self.model, 'sde_loss'):
            sde_loss = self.model.sde_loss(outputs['mel_ref'], mel_gt).item()
        elif not self.sde_loss_missing_warned:
            logger.warning("Model does not support sde_loss; reporting L1 loss only")
            self.sde_loss_missing_warned = True
        return {'l1_loss': l1_loss.item(), 'sde_loss': sde_loss}

    def update_noise_schedule(self, val_l1: float) -> None:
        """Updates noise schedule based on validation L1 loss."""
        if not self.noise_annealing:
            return
        if val_l1 >= self.prev_val_l1 * 0.99:
            self.l1_plateau_count += 1
            if self.l1_plateau_count >= 3:
                self.noise_schedule = max(self.noise_schedule * 0.9, 0.1)
                self.l1_weight = min(self.l1_weight * 1.1, 0.9)
                self.l1_plateau_count = 0
                logger.info(f"Annealed SDE noise to {self.noise_schedule}, L1 weight to {self.l1_weight}")
        else:
            self.l1_plateau_count = 0
        self.prev_val_l1 = val_l1