```python
from dataclasses import dataclass

@dataclass
class AcousticConfig:
    """
    Configuration for SOTA AcousticModel.

    Attributes:
        text_emb_dim: dim of input text embeddings
        cond_dim: dim of conditioning (f0, energy, pitch_var, speech_rate, pause_dur, mfcc)
        speaker_dim: dim of speaker embedding
        emotion_dim: dim of emotion probabilities
        hidden_channels: number of conv channels
        num_layers: number of Conformer layers
        dropout: dropout prob per block
        kernel_size: width of depthwise conv
        layer_scale_init: init value for LayerScale
        base_sd_prob: base stochastic depth drop prob
        ci_latency_factor: max allowed factor vs MLP in CI
        profile: toggle profiling prints
        use_amp: enable mixed precision for GPU
        attention_heads: number of attention heads
        transformer_dim: dimension for Conformer feedforward
        emotion_intensity: enable intensity scaling for emotions
        diffusion_steps: number of diffusion steps
        conformer_layers: number of Conformer layers
        prune_ratio: ratio of weights to prune
    """
    text_emb_dim: int
    cond_dim: int = 18  # f0, energy, pitch_var, speech_rate, pause_dur (5) + mfcc (13)
    speaker_dim: int = 16
    emotion_dim: int = 6  # joy, sadness, anger, neutral, surprise, fear
    hidden_channels: int = 256  # Reduced for RTX 2050
    num_layers: int = 6  # Conformer layers
    dropout: float = 0.1
    kernel_size: int = 5
    layer_scale_init: float = 1e-4
    base_sd_prob: float = 0.1
    ci_latency_factor: float = 20.0
    profile: bool = False
    use_amp: bool = True
    attention_heads: int = 4  # Reduced for efficiency
    transformer_dim: int = 512
    emotion_intensity: bool = True
    diffusion_steps: int = 10
    conformer_layers: int = 6
    prune_ratio: float = 0.2
```