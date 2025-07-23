# train/callbacks/scheduler.py
from typing import Dict
from ..config_schemas import UnifiedTrainerConfig
from .base import Callback
import logging

logger = logging.getLogger(__name__)

class SchedulerCallback(Callback):
    """Handles learning rate scheduling."""
    def __init__(self, _config: UnifiedTrainerConfig):
        pass

    def on_batch_end(self, step: int, metrics: Dict[str, float], trainer: 'UnifiedTrainer') -> None:
        """Steps schedulers after batch; moved to train_step for grad accumulation alignment."""
        pass

    def on_val_end(self, step: int, metrics: Dict[str, float], trainer: 'UnifiedTrainer') -> None:
        pass

    def on_epoch_start(self, trainer: 'UnifiedTrainer') -> None:
        pass

    def on_epoch_end(self, step: int, metrics: Dict[str, float], trainer: 'UnifiedTrainer') -> None:
        pass

    def on_train_start(self, trainer: 'UnifiedTrainer') -> None:
        pass

    def on_train_end(self, trainer: 'UnifiedTrainer') -> None:
        pass

    def on_checkpoint(self, step: int, metrics: Dict[str, float], state: Dict[str, object], trainer: 'UnifiedTrainer') -> None:
        pass