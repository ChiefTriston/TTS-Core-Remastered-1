
from dataclasses import dataclass

@dataclass
class AcousticConfig:
    """
    Configuration for AcousticModel.

    Attributes:
        text_emb_dim: dim of input text embeddings
        cond_dim: dim of base conditioning (f0 + energy)
        speaker_dim: dim of speaker embedding (0 to disable)
        hidden_channels: number of conv channels
        num_layers: number of residual blocks
        dropout: dropout prob per block
        kernel_size: width of depthwise conv
        layer_scale_init: init value for LayerScale
        base_sd_prob: base stochastic depth drop prob
        ci_latency_factor: max allowed factor vs MLP in CI
        profile: toggle profiling prints
    """
    text_emb_dim: int
    cond_dim: int = 2
    speaker_dim: int = 0
    hidden_channels: int = 384
    num_layers: int = 4
    dropout: float = 0.1
    kernel_size: int = 5
    layer_scale_init: float = 1e-4
    base_sd_prob: float = 0.1
    ci_latency_factor: float = 20.0
    profile: bool = False