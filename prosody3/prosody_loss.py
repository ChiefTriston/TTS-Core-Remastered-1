
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProsodyLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=None):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred: predicted prosody (f0 or energy) [B, T]
        target: ground truth prosody [B, T]
        mask: optional boolean mask tensor [B, T] to ignore padding in loss

        Returns weighted SmoothL1 loss averaged over valid elements.
        """

        loss = self.loss_fn(pred, target)  # [B, T]

        if mask is not None:
            loss = loss * mask.float()
            if self.reduction == 'mean':
                return loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
