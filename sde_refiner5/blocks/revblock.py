import inspect
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class RevBlock(nn.Module):
    """
    Memory-efficient reversible block via gradient checkpointing.
    Wraps any nn.Module and applies checkpoint to trade compute for memory.
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, *, cond=None, style=None) -> torch.Tensor:
        """
        Always call checkpoint(self.module, x, **kw), where kw
        only contains the args its forward actually wants.
        """
        sig = inspect.signature(self.module.forward)
        kws = {}
        # if module.forward signature has a 'cond' param, forward our cond
        if 'cond' in sig.parameters and cond is not None:
            kws['cond'] = cond
        # if it has a 'style' param, forward our style
        if 'style' in sig.parameters and style is not None:
            kws['style'] = style
        return checkpoint(lambda inp, _kws=kws: self.module(inp, **_kws), x)
