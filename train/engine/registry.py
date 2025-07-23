# train/engine/registry.py
from typing import Dict, Type
import torch
from ..blocks.base import TrainBlock
from ..blocks.acoustic import AcousticBlock
from ..blocks.refiner import RefinerBlock
from ..blocks.vocoder import VocoderBlock
from ..callbacks.base import Callback
from ..callbacks.checkpoint import CheckpointCallback
from ..callbacks.logger import LoggerCallback
from ..callbacks.ema import EMACallback
from ..config_schemas import UnifiedTrainerConfig

class BlockRegistry:
    """Registry for training blocks."""
    def __init__(self):
        self.blocks: Dict[str, Type[TrainBlock]] = {
            "acoustic": AcousticBlock,
            "refiner": RefinerBlock,
            "vocoder": VocoderBlock
        }

    def register_block(self, name: str, block_class: Type[TrainBlock]) -> None:
        """Registers a new block type."""
        self.blocks[name] = block_class

    def instantiate(self, name: str, config: UnifiedTrainerConfig, device: torch.device) -> TrainBlock:
        """Instantiates a block by name."""
        if name not in self.blocks:
            raise ValueError(f"Unknown block: {name}")
        return self.blocks[name](config, device)

class CallbackRegistry:
    """Registry for callbacks. Note: 'scheduler' callback is deprecated; scheduler stepping moved to train_step."""
    def __init__(self):
        self.callbacks = {
            "checkpoint": CheckpointCallback,
            "logger": LoggerCallback,
            # "scheduler": SchedulerCallback,  # Deprecated; scheduler stepping handled in train_step
            "ema": EMACallback,
        }

    def register_callback(self, name: str, cb_class: Type[Callback]) -> None:
        self.callbacks[name] = cb_class

    def instantiate(self, name: str, config: UnifiedTrainerConfig) -> Callback:
        if name not in self.callbacks:
            raise ValueError(f"Unknown callback: {name}")
        return self.callbacks[name](config)