# train/callbacks/ema.py
from typing import Dict, Optional, TYPE_CHECKING
from .base import Callback
if TYPE_CHECKING:
    from ..engine.trainer import UnifiedTrainer
from ..config_schemas import UnifiedTrainerConfig

class EMACallback(Callback):
    """Callback for exponential moving average of model parameters."""
    def __init__(self, config: UnifiedTrainerConfig):
        self.model = None
        self.decay = config.vocoder.ema_g.decay if config.vocoder.ema_g.enabled else 0.999
        self.swap_validate = config.trainer.novel.ema_swap_validate.enabled
        self.shadow_params = {}
        self.original_params = {}

    def on_train_start(self, trainer: 'UnifiedTrainer', **kwargs) -> None:
        if trainer.blocks.get('vocoder') and trainer.config.vocoder.ema_g.enabled:
            self.model = trainer.blocks['vocoder'].model.generator
        if self.model:
            self.shadow_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def on_batch_end(self, step: int, metrics: Dict[str, float], trainer: Optional['UnifiedTrainer'] = None, **kwargs) -> None:
        if self.model:
            for name, param in self.model.named_parameters():
                self.shadow_params[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def swap_to_ema(self) -> None:
        if self.model and self.swap_validate:
            self.original_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
            for name, param in self.model.named_parameters():
                param.data.copy_(self.shadow_params[name])

    def swap_to_train(self) -> None:
        if self.model and self.swap_validate and self.original_params:
            for name, param in self.model.named_parameters():
                param.data.copy_(self.original_params[name])
            self.original_params = {}