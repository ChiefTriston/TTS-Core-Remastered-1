# 5_sde_refiner/blocks/gumbel_moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelMoE(nn.Module):
    """
    Style‐conditioned Mixture‐of‐Experts with Gumbel‐Softmax routing.
    Args:
      input_dim : number of input channels
      output_dim: number of output channels
      num_experts: how many experts to route between
      style_dim : dimension of style‐conditioning vector
      dropout   : dropout probability on gate weights
    """
    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 4,
                 style_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(style_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:     [B, D_in, T] or [B, T, D_in]
          style: [B, style_dim]
        Returns:
          [B, D_out, T] or [B, T, D_out], matching input shape.
        """
        # Detect if x is [B, D, T], convert to [B, T, D]
        transposed = False
        if x.dim() == 3 and x.shape[1] != style.shape[0] and x.shape[1] != style.shape[1]:
            x = x.transpose(1, 2)  
            transposed = True

        B, T, D_in = x.shape

        # Gate logits → (B, num_experts)
        gate_logits = self.gate(style)
        gate_weights = F.gumbel_softmax(gate_logits, tau=1.0, hard=False)  # [B, E]
        gate_weights = self.dropout(gate_weights).unsqueeze(1).unsqueeze(1)  # [B,1,1,E]

        # Expert outputs: list of [B, T, D_out]
        expert_outs = [expert(x) for expert in self.experts]  # each is [B, T, D_out]
        stacked = torch.stack(expert_outs, dim=-1)            # [B, T, D_out, E]
        out = (stacked * gate_weights).sum(dim=-1)            # [B, T, D_out]

        if transposed:
            out = out.transpose(1, 2)  # back to [B, D_out, T]
        return out
