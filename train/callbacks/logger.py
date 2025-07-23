# train/callbacks/logger.py
from typing import Dict, Optional, TYPE_CHECKING
from .base import Callback
from ..utils.logging import Logger
from ..config_schemas import UnifiedTrainerConfig
if TYPE_CHECKING:
    from ..engine.trainer import UnifiedTrainer

class LoggerCallback(Callback):
    """Callback for logging metrics."""
    def __init__(self, config: UnifiedTrainerConfig):
        self.logger = Logger(config.trainer.log_dir, config.logging.tensorboard, config.logging.csv)

    def on_batch_end(self, step: int, metrics: Dict[str, float], trainer: Optional['UnifiedTrainer'] = None, **kwargs) -> None:
        self.logger.log_train_metrics(step, metrics)

    def on_val_end(self, step: int, metrics: Dict[str, float], trainer: Optional['UnifiedTrainer'] = None, **kwargs) -> None:
        self.logger.log_val_metrics(step, metrics)

    def on_train_end(self, trainer: 'UnifiedTrainer', **kwargs) -> None:
        self.logger.close()