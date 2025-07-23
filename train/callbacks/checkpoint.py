# train/callbacks/checkpoint.py
from typing import Dict, Optional, TYPE_CHECKING
import os
from .base import Callback
from ..utils.checkpoint import save_checkpoint
if TYPE_CHECKING:
    from ..engine.trainer import UnifiedTrainer
from ..config_schemas import UnifiedTrainerConfig

class CheckpointCallback(Callback):
    """Callback for saving checkpoints."""
    def __init__(self, config: UnifiedTrainerConfig):
        self.checkpoint_dir = config.trainer.checkpoint_dir
        self.checkpoint_freq = config.trainer.checkpoint_freq
        self.best_metric = config.trainer.best_metric
        self.best_value = float('inf')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_checkpoint(self, step: int, metrics: Dict[str, float], state: Dict, trainer: Optional['UnifiedTrainer'] = None, **kwargs) -> None:
        if step % self.checkpoint_freq == 0:
            is_best = self.best_metric in metrics and metrics[self.best_metric] < self.best_value
            if is_best:
                self.best_value = metrics[self.best_metric]
            save_checkpoint(state, step, metrics, self.checkpoint_dir, self.best_metric, is_best=is_best)