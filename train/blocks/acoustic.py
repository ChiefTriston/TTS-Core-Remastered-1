# train/blocks/acoustic.py
from typing import Dict
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from acoustic.model import AcousticModel
from ..config_schemas import UnifiedTrainerConfig
from .base import TrainBlock
import logging
import os

logger = logging.getLogger(__name__)

class AcousticBlock(TrainBlock):
    """Acoustic model block for text-to-mel spectrogram generation."""
    def __init__(self, config: UnifiedTrainerConfig, device: torch.device):
        model = AcousticModel().to(device)
        if config.acoustic.model_path and os.path.exists(config.acoustic.model_path):
            try:
                model.load_state_dict(torch.load(config.acoustic.model_path, map_location=device))
            except Exception as e:
                logger.warning(f"Failed to load acoustic model: {e}")
        optimizer = AdamW(model.parameters(), lr=config.acoustic.optim.lr, betas=config.acoustic.optim.betas, weight_decay=config.acoustic.optim.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=config.acoustic.scheduler.max_lr, total_steps=config.trainer.max_steps, pct_start=config.acoustic.scheduler.pct_start)
        scaler = GradScaler(enabled=config.acoustic.amp)
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
        super().__init__(model, [optimizer], [scheduler], scaler, device)

    def forward(self, batch: Dict[str, torch.Tensor], **cond) -> Dict[str, torch.Tensor]:
        with autocast(enabled=self.scaler().enabled):
            mel_pred = self.model(batch['text_ids'], batch['phonemes'], batch.get('prosody_stats'), batch.get('speaker_id'), batch.get('style_id'))
        return {'mel_pred': mel_pred}

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mel_gt = batch['mel_gt'].to(self.device)
        assert outputs['mel_pred'].shape == mel_gt.shape, f"Mismatch: mel_pred {outputs['mel_pred'].shape}, mel_gt {mel_gt.shape}"
        l1_loss = torch.nn.functional.l1_loss(outputs['mel_pred'], mel_gt)
        return {'l1_loss': l1_loss}

    def metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        mel_gt = batch['mel_gt'].to(self.device)
        l1_loss = torch.nn.functional.l1_loss(outputs['mel_pred'], mel_gt)
        esr = (outputs['mel_pred'] - mel_gt).pow(2).mean() / (mel_gt.pow(2).mean() + 1e-8)
        return {'l1_loss': l1_loss.item(), 'esr': esr.item()}