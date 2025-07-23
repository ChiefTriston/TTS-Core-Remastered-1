# train/blocks/base.py
from typing import Dict, List, Optional
import torch
from torch.cuda.amp import GradScaler

class TrainBlock:
    """Base class for training blocks."""
    def __init__(self, model: torch.nn.Module, optimizers: List[torch.optim.Optimizer], schedulers: List, scaler: GradScaler, device: torch.device):
        self.model = model
        self._optimizers = optimizers
        self._schedulers = schedulers
        self._scaler = scaler
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor], **cond) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        raise NotImplementedError()

    def step(self, grad_clip_norm: Optional[float] = None) -> None:
        """Steps single optimizer with gradient clipping. GAN or multi-optimizer blocks (e.g., VocoderBlock) must implement custom step methods (e.g., gen_step, disc_step)."""
        if len(self._optimizers) != 1:
            raise RuntimeError(f"{self.__class__.__name__}.step() supports only one optimizer; use specialized step methods for multi-optimizer blocks")
        self._scaler.unscale_(self._optimizers[0])
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
        self._scaler.step(self._optimizers[0])
        self._scaler.update()
        self._optimizers[0].zero_grad(set_to_none=True)

    def scaler(self) -> GradScaler:
        return self._scaler

    def optimizers(self) -> List[torch.optim.Optimizer]:
        return self._optimizers

    def schedulers(self) -> List:
        return self._schedulers