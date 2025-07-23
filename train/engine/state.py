# train/engine/state.py
from typing import Optional
import torch

class TrainingState:
    """Tracks global training state."""
    def __init__(self, max_steps: int):
        self.global_step: int = 0
        self.max_steps: int = max_steps
        self.rng_state: Optional[torch.Tensor] = None
        self.oom_count: int = 0