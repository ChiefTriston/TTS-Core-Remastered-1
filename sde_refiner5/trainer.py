# sde_refiner/trainer.py

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from .config import ScoreSDEConfig
from .refiner import ScoreSDERefinerV15

class RefinerTrainer:
    """
    Trainer for ScoreSDERefinerV15 using denoising score matching.
    """
    def __init__(
        self,
        acoustic_cfg: ScoreSDEConfig,
        refiner_cfg: ScoreSDEConfig,
        device: str = 'cuda'
    ):
        # Device
        self.device = device

        # Model
        self.refiner = ScoreSDERefinerV15(acoustic_cfg, refiner_cfg).to(device)

        # Optimizer
        self.opt = optim.Adam(self.refiner.parameters(),
                              lr=refiner_cfg.lr,
                              betas=(0.9, 0.999))

        # Scheduler (e.g., cosine annealing)
        self.sch = optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=refiner_cfg.scheduler_steps
        )

        # Mixed-precision scaler
        self.scaler = GradScaler()

    def training_step(self, batch, step: int):
        """
        Perform one training step.

        batch: (text_emb, prosody, style_id, speaker, mel_target, vader_scores)
        step: global step counter
        """
        text_emb, prosody, style_id, speaker, mel_target, vader_scores = batch

        self.opt.zero_grad()
        with autocast():
            # Forward pass through the refiner
            mel_ref, duration, pitch = self.refiner(
                text_emb.to(self.device),
                prosody.to(self.device),
                style_id.to(self.device),
                speaker.to(self.device) if speaker is not None else None,
                t=None,
                vader_scores=vader_scores.to(self.device) if vader_scores is not None else None
            )

            # Reconstruction loss between refined and target mel
            loss = nn.functional.l1_loss(
                mel_ref,
                mel_target.to(self.device)
            )

        # Backward and optimizer step
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        # Scheduler update
        self.sch.step()

        return {"refiner_loss": loss.item()}
