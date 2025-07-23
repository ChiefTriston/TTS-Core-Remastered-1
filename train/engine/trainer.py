# train/engine/trainer.py
from typing import Dict, List
import torch
import numpy as np
import random
import os
import logging
from .registry import BlockRegistry, CallbackRegistry
from .state import TrainingState
from .loop import TrainingLoop
from ..config_schemas import UnifiedTrainerConfig
from ..blocks.base import TrainBlock
from ..callbacks.base import Callback
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class UnifiedTrainer:
    """Orchestrates training by managing blocks, callbacks, and state."""
    def __init__(self, config: UnifiedTrainerConfig, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        self.config = config
        self.device = device
        self.state = TrainingState(config.trainer.max_steps)
        self.block_registry = BlockRegistry()
        self.callback_registry = CallbackRegistry()
        self.blocks: Dict[str, TrainBlock] = {}
        self.callbacks: List[Callback] = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loop = TrainingLoop(self)
        self.observer = None
        self._setup()

    def _setup(self) -> None:
        """Initializes blocks and callbacks from config."""
        if self.config.acoustic.enabled:
            self.blocks['acoustic'] = self.block_registry.instantiate("acoustic", self.config, self.device)
        if self.config.refiner.enabled:
            self.blocks['refiner'] = self.block_registry.instantiate("refiner", self.config, self.device)
        if self.config.vocoder.enabled:
            self.blocks['vocoder'] = self.block_registry.instantiate("vocoder", self.config, self.device)
        
        self.callbacks.extend([
            self.callback_registry.instantiate("checkpoint", self.config),
            self.callback_registry.instantiate("logger", self.config),
            self.callback_registry.instantiate("ema", self.config)
            # Note: 'scheduler' callback is deprecated; scheduler stepping moved to train_step
        ])
        
        if self.config.observer.enabled:
            from ..observer import Observer
            self.observer = Observer(self.config.observer.module_path, self.config.observer.policy)
        if self.config.trainer.resume:
            best_path = f"{self.config.trainer.checkpoint_dir}/best.pt"
            if os.path.exists(best_path):
                try:
                    self.load_checkpoint(best_path)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")

    def train(self) -> None:
        """Starts the training process."""
        self.loop.run()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executes a single training step."""
        return self.loop.train_step(batch)

    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executes a single validation step."""
        return self.loop.val_step(batch)

    def save_checkpoint(self, step: int, metrics: Dict[str, float]) -> None:
        """Saves checkpoint via callback."""
        state = self._get_state()
        for cb in self.callbacks:
            cb.on_checkpoint(step, metrics, state, trainer=self)

    def load_checkpoint(self, path: str) -> None:
        """Loads checkpoint and updates state."""
        from ..utils.checkpoint import load_checkpoint
        checkpoint = load_checkpoint(path, self.device)
        self.state.global_step = checkpoint.get('global_step', 0)
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
        if 'cuda_rng_state_all' in checkpoint and torch.cuda.is_available():
            try:
                saved_states = checkpoint['cuda_rng_state_all']
                if not saved_states:
                    logger.warning("Empty cuda_rng_state_all in checkpoint")
                else:
                    current_devices = torch.cuda.device_count()
                    if len(saved_states) >= current_devices:
                        torch.cuda.set_rng_state_all(saved_states[:current_devices])
                    else:
                        logger.warning(f"CUDA RNG state mismatch: saved {len(saved_states)} devices, but {current_devices} available")
                        torch.cuda.set_rng_state_all(saved_states + [saved_states[-1]] * (current_devices - len(saved_states)))
            except (IndexError, RuntimeError) as e:
                logger.warning(f"Failed to restore CUDA RNG state: {e}")
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'py_rng_state' in checkpoint:
            random.setstate(checkpoint['py_rng_state'])
        for name, block in self.blocks.items():
            sd = checkpoint.get(name) or checkpoint.get(f"{name}")
            if sd:
                block.model.load_state_dict(sd, strict=False)
            # optimizers
            for i, opt in enumerate(block.optimizers()):
                key = f"{name}_opt_{i}"
                if key in checkpoint:
                    opt.load_state_dict(checkpoint[key])
            # schedulers
            for i, sch in enumerate(block.schedulers()):
                key = f"{name}_sch_{i}"
                if key in checkpoint:
                    sch.load_state_dict(checkpoint[key])
            # scaler
            key = f"{name}_scaler"
            if key in checkpoint:
                block.scaler().load_state_dict(checkpoint[key])
        # EMA
        if 'ema_g' in checkpoint:
            for cb in self.callbacks:
                if hasattr(cb, 'shadow_params'):
                    cb.shadow_params = checkpoint['ema_g']
                    break

    def _get_state(self) -> Dict[str, object]:
        """Build a flat checkpoint dict."""
        state: Dict[str, object] = {
            'global_step': self.state.global_step,
            'rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate()
        }
        if torch.cuda.is_available():
            state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        for name, block in self.blocks.items():
            state[name] = block.model.state_dict()
            for i, opt in enumerate(block.optimizers()):
                state[f"{name}_opt_{i}"] = opt.state_dict()
            for i, sch in enumerate(block.schedulers()):
                state[f"{name}_sch_{i}"] = sch.state_dict()
            state[f"{name}_scaler"] = block.scaler().state_dict()
        # EMA (grab from EMA callback if present)
        for cb in self.callbacks:
            if hasattr(cb, 'shadow_params'):
                state['ema_g'] = cb.shadow_params
                break
        return state