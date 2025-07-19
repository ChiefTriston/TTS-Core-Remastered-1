
```python
#!/usr/bin/env python3
"""
ProsodyLoss: Computes weighted SmoothL1 loss for prosodic features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProsodyLoss(nn.Module):
    def __init__(self, f0_weight=1.0, energy_weight=1.0, pitch_var_weight=1.0,
                 speech_rate_weight=1.0, pause_dur_weight=1.0, mfcc_weight=1.0,
                 reduction='mean', ignore_index=None):
        super().__init__()
        self.weights = {
            'f0': f0_weight,
            'energy': energy_weight,
            'pitch_var': pitch_var_weight,
            'speech_rate': speech_rate_weight,
            'pause_dur': pause_dur_weight,
            'mfcc': mfcc_weight
        }
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred: dict, target: dict, mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred: Dict with keys ['f0', 'energy', 'pitch_var', 'speech_rate', 'pause_dur', 'mfcc']
        target: Dict with matching keys
        mask: Optional boolean mask tensor [B, T] for f0, energy, pitch_var; [B, 1] for speech_rate, pause_dur; None for mfcc

        Returns weighted SmoothL1 loss averaged over valid elements.
        """
        total_loss = 0.0
        for key in ['f0', 'energy', 'pitch_var']:
            loss = self.loss_fn(pred[key], target[key])  # [B, T]
            if mask is not None:
                loss = loss * mask.float()
            total_loss += self.weights[key] * (loss.sum() / mask.sum().clamp(min=1) if mask is not None else loss.mean())

        for key in ['speech_rate', 'pause_dur']:
            loss = self.loss_fn(pred[key], target[key])  # [B, 1]
            if mask is not None:
                loss = loss * mask.float()
            total_loss += self.weights[key] * (loss.sum() / mask.sum().clamp(min=1) if mask is not None else loss.mean())

        # MFCC loss (vectorized)
        loss = self.loss_fn(pred['mfcc'], target['mfcc'])  # [B, 13]
        total_loss += self.weights['mfcc'] * loss.mean()

        if self.reduction == 'mean':
            return total_loss / len(self.weights)
        elif self.reduction == 'sum':
            return total_loss
        else:
            return total_loss
```