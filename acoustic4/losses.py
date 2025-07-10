
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MelSpectrogramLoss(nn.Module):
    """
    Combines L1 mel-spectrogram loss with spectral convergence.
    """
    def __init__(self, power=2.0, eps=1e-10):
        super().__init__()
        self.power = power
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # pred, target: [B, T, 80]
        # L1 loss
        l1 = F.l1_loss(pred, target)
        # Spectral convergence
        sc = torch.norm(target - pred, p='fro') / (torch.norm(target, p='fro') + self.eps)
        return l1 + sc

class FrameWiseMSELoss(nn.Module):
    """
    Mean Squared Error over frames.
    """
    def __init__(self):
        super().__init__()
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(pred, target)

class CompositeLoss(nn.Module):
    """
    Composite loss weighting mel and frame-wise MSE.
    """
    def __init__(self, mel_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.mel_loss = MelSpectrogramLoss()
        self.mse_loss = FrameWiseMSELoss()
        self.mel_w = mel_weight
        self.mse_w = mse_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.mel_w * self.mel_loss(pred, target) + self.mse_w * self.mse_loss(pred, target)