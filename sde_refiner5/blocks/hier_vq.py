# 5_sde_refiner/blocks/hier_vq.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Single‐layer Vector Quantizer with commitment loss and straight‐through estimator.
    Args:
      dim      : embedding dimension of each code
      num_codes: number of embedding vectors in codebook
    """
    def __init__(self, dim: int, num_codes: int):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, dim)
        self.embedding.weight.data.uniform_(-1/num_codes, 1/num_codes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C, T]
        returns (quantized: [B, C, T], loss: scalar)
        """
        B, C, T = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, C)  # [B*T, C]
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        encode_idx = torch.argmin(dist, dim=1)
        quant = self.embedding(encode_idx).view(B, T, C).permute(0, 2, 1)  # [B, C, T]

        # Loss
        loss = F.mse_loss(quant.detach(), x) + F.mse_loss(quant, x.detach())
        quantized = x + (quant - x).detach()
        return quantized, loss

class HierVQ(nn.Module):
    """
    Chains multiple VectorQuantizer layers in sequence.
    dims: list of embedding dims for each layer
    codes:list of code‐book sizes for each layer
    """
    def __init__(self, dims: list[int], codes: list[int]):
        super().__init__()
        assert len(dims) == len(codes), "dims and codes length must match."
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(d, c) for d, c in zip(dims, codes)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_loss = 0.0
        out = x
        for vq in self.vq_layers:
            out, loss = vq(out)
            total_loss += loss
        return out, total_loss
