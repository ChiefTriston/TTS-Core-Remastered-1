
#!/usr/bin/env python3
"""
S4: State-Space Sequence Model layer

Defaults for hyperparameters are loaded from config.prosody.json in the same folder.
"""
import json
from pathlib import Path
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import spectral_norm
import torch.amp  # use new AMP API
from typing import Optional, Tuple

# Load S4-specific config
_CFG_PATH = Path(__file__).parent / "config.prosody.json"
_CFG = json.load(_CFG_PATH.open())
S4_CFG = _CFG.get("s4", {})

__all__ = ["S4"]

def _init_ssm(d: int, r: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    A = -torch.linspace(1.0, d, d, dtype=torch.float32) / d
    U = torch.randn(d, r, dtype=torch.float32) * d**-0.5
    V = torch.randn(d, r, dtype=torch.float32) * d**-0.5
    B = torch.ones(d, dtype=torch.float32)
    return A, U, V, B

class S4(nn.Module):
    """
    Multi-head S4 V7.0
    """
    def __init__(
        self,
        d_model: int,
        heads:         int   = S4_CFG.get("heads", 4),
        l_max:         int   = S4_CFG.get("l_max", 1024),
        rank:          int   = S4_CFG.get("rank", 1),
        dropout:     float  = S4_CFG.get("dropout", 0.1),
        norm_groups:  int   = S4_CFG.get("norm_groups", 8),
        use_fft:     bool   = S4_CFG.get("use_fft", False),
        causal:      bool   = S4_CFG.get("causal", False),
        cache_decay: bool   = S4_CFG.get("cache_decay", True),
        fft_threshold:float = S4_CFG.get("fft_threshold", 2.0),
    ):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.H, self.d = heads, d_model // heads
        self.l_max = l_max
        self.use_fft = use_fft
        self.causal = causal
        self.cache_decay = cache_decay
        self.fft_threshold = fft_threshold
        self.use_amp = _CFG.get("use_amp", False) and torch.cuda.is_available()

        # Norm layers
        self.ln = nn.LayerNorm(d_model)
        self.gn = nn.GroupNorm(norm_groups, d_model)

        # SwiGLU gating + dropout
        self.gate = weight_norm(nn.Conv1d(d_model, 2*d_model, 1, bias=True))
        self.dropout = nn.Dropout(dropout)

        # Pos bias + small local conv
        self.pos_bias = nn.Parameter(torch.zeros(heads, l_max))
        self.local_conv = nn.Conv1d(d_model, d_model, 3, padding=1, groups=heads)

        # Low-rank residual + scale
        self.lr_drop = nn.Dropout1d(dropout)
        self.alpha   = nn.Parameter(torch.ones(heads, 1, 1))

        # SSM params per head
        As, Us, Vs, Bs = zip(*[_init_ssm(self.d, rank) for _ in range(self.H)])
        self.register_buffer('A_diag', torch.stack(As, dim=0))   # [H, d]
        self.U = nn.Parameter(torch.stack(Us, dim=0))           # [H, d, r]
        self.V = nn.Parameter(torch.stack(Vs, dim=0))           # [H, d, r]
        self.register_buffer('B', torch.stack(Bs, dim=0))       # [H, d]

        # Factorized C & bias D
        self.C1 = nn.Parameter(torch.randn(self.H, self.d, rank) * 0.02)
        self.C2 = nn.Parameter(torch.randn(self.H, rank, self.d) * 0.02)
        spectral_norm(self.local_conv)  # only on conv
        self.C0 = nn.Parameter(torch.zeros(self.H, self.d))
        self.D  = nn.Parameter(torch.zeros(self.H, self.d))

        # Precompute decay buffer
        t = torch.arange(l_max, dtype=torch.float32).view(1, l_max, 1)
        decay_arg = (self.A_diag.unsqueeze(1) * t).clamp(-50.0, 50.0)
        expAB = torch.exp(decay_arg) * self.B.unsqueeze(1) if cache_decay else torch.empty(0)
        self.register_buffer('decay_buf', expAB)
        self.register_buffer('time_idx', t.squeeze(-1).to(torch.long))

        # Precompute full C
        with torch.no_grad():
            Cfull = (self.C1 @ self.C2) + torch.diag_embed(self.C0)
        self.register_buffer('C_full_buf', Cfull)
        self._cache: Optional[tuple] = None
        self.register_forward_pre_hook(self._update_C_full)

    @torch.jit.ignore
    def _update_C_full(self, module, *args):
        buf = (self.C1 @ self.C2 + torch.diag_embed(self.C0)).detach()
        self.C_full_buf.copy_(buf)

    @torch.jit.ignore
    def _fft_conv(self, x: Tensor, w: Tensor) -> Tensor:
        B, C, T = x.shape
        L = w.shape[-1]
        n = T + L - 1
        Xf = torch.fft.rfft(x, n=n)
        Kf = torch.fft.rfft(F.pad(w, (0, n-L)).squeeze(1), n=n)
        y = torch.fft.irfft(Xf.unsqueeze(1) * Kf.unsqueeze(0), n=n)
        s = (L-1)//2
        return y[..., s:s+T]

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass under AMP guard"""
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # [B, C, T] canonical
            if x.ndim == 3 and x.shape[-1] == self.H*self.d:
                x = x.permute(0,2,1)
            B, C, T = x.shape
            x = self.ln(x.permute(0,2,1)).permute(0,2,1)

            # SSM conv
            if self.cache_decay:
                expAB_t = self.decay_buf[:, :T, :].to(x.dtype)
            else:
                t = self.time_idx[:T].to(x.device).unsqueeze(-1).to(x.dtype)
                decay_arg = (self.A_diag.unsqueeze(1) * t).clamp(-50.0, 50.0)
                expAB_t = torch.exp(decay_arg) * self.B.unsqueeze(1)

            key = (T, x.dtype, x.device)
            if self._cache and self._cache[:3] == key:
                Kt = self._cache[3]
            else:
                Kt = torch.einsum('htd,hde->hte', expAB_t, self.C_full_buf)
                self._cache = (T, x.dtype, x.device, Kt)

            L = Kt.size(1)
            w = Kt.transpose(1,2).reshape(self.H*self.d, 1, L)

            pad_l = L-1 if self.causal else (L-1)//2
            pad_r = 0   if self.causal else L-1-pad_l
            x_p = x if pad_l==pad_r else F.pad(x, (pad_l,pad_r))
            y = F.conv1d(x_p, w,
                         padding=pad_l if pad_l==pad_r else 0,
                         groups=self.H*self.d)
            if self.causal:
                y = y[..., :T]

            # pos-bias expansion
            if T <= self.l_max:
                pb = self.pos_bias[:, :T]
            else:
                tail = self.pos_bias[:, -1:].expand(-1, T-self.l_max)
                pb = torch.cat([self.pos_bias, tail], dim=1)
            bias = pb.unsqueeze(1).expand(-1, self.d, -1).reshape(self.H*self.d, T)
            y = y + bias.unsqueeze(0)

            # local conv + SwiGLU
            y = self.local_conv(y)
            a, b = self.gate(y).chunk(2,1)
            y = a * F.silu(b)
            y = self.dropout(y)

            # ====== PATCHED LOW-RANK RESIDUAL START ======
            x_flat = x.permute(0, 2, 1).reshape(B * T, C)
            r = self.U.size(-1)
            U_flat = self.U.view(C, r)
            V_flat = self.V.view(C, r)
            res_flat = x_flat @ V_flat
            res_flat = res_flat @ U_flat.t()
            residual = res_flat.view(B, T, C).permute(0, 2, 1)
            residual = self.lr_drop(residual)
            y = y + residual
            # ====== PATCHED LOW-RANK RESIDUAL END ======

            out = self.gn(y)
            return out

    def __repr__(self) -> str:
        return f"S4(d_model={self.H*self.d}, heads={self.H}, l_max={self.l_max})"
