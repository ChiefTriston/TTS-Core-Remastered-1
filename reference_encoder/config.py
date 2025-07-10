from dataclasses import dataclass

@dataclass
class RefEncConfig:
    # Audio parameters
    sample_rate: int = 22050       # sampling rate in Hz
    n_mels: int      = 80          # mel filter banks
    window_size: int = 1024        # STFT window size
    hop_length: int  = 256         # STFT hop (stride)

    # Speaker embedding dimension
    speaker_dim: int = 256         # output embedding size

    # Backbone for spectral feature extraction: ecapa_tdnn, res2net, conformer, wav2vec2
    backbone: str = "wav2vec2"
    wav2vec2_name: str = "facebook/wav2vec2-base-960h"

    # Pooling strategy: self_attentive, multi_head_attentive, stats
    pooling: str = "self_attentive"

    # Loss configuration
    loss_type: str = "arcface"     # arcface or ge2e
    margin: float = 0.3
    scale: float  = 30.0
    use_margin_schedule: bool = False

    # Training parameters
    optimizer: str    = "AdamW"
    lr: float         = 1e-4
    weight_decay: float = 1e-5
    batch_size: int   = 8          # micro-batch size to fit GPU memory
    accumulation_steps: int = 16   # accumulate grads to simulate effective batch of 128
    max_epochs: int   = 50
    warmup_steps: int = 5000
    total_steps: int  = 200000
    gradient_clip: float = 3.0

    # Data augmentation flags
    augment_noise: bool   = True
    augment_reverb: bool  = True
    speed_perturb: bool   = True
    spec_augment: bool    = True
    mixup: bool           = False
    speaker_batch_utterances: int = 4

    # Checkpointing & evaluation
    checkpoint_steps: int = 5000
    eval_steps: int       = 5000
    output_dir: str       = "./checkpoints"
    onnx_export: bool     = True
