import time
import torch
from .config import AcousticConfig
from .model import AcousticModel
from prosody.prosody_predictor import ProsodyPredictorV15


def benchmark_acoustic_model(cfg: AcousticConfig, B: int = 2, T: int = 128) -> float:
    """
    Measures average forward-pass time of AcousticModel over 20 runs.
    Returns time per run in seconds.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AcousticModel(cfg).to(device)
    # Prepare dummy inputs
    text = torch.randn(B, T, cfg.text_emb_dim, device=device)
    f0 = torch.randn(B, T, device=device)
    energy = torch.randn(B, T, device=device)
    speaker = (torch.randn(B, cfg.speaker_dim, device=device)
               if cfg.speaker_dim > 0 else None)
    # Warmup
    _ = model(text, f0, energy, speaker)
    # Timing
    start = time.time()
    for _ in range(20):
        _ = model(text, f0, energy, speaker)
    return (time.time() - start) / 20


def benchmark_prosody_predictor(B: int = 2, T: int = 128, emb_dim: int = 384) -> float:
    """
    Measures average forward-pass time of ProsodyPredictorV15 over 20 runs.
    Returns time per run in seconds.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = ProsodyPredictorV15().to(device)
    text_emb = torch.randn(B, T, emb_dim, device=device)
    # Warmup
    _ = predictor(text_emb)
    # Timing
    start = time.time()
    for _ in range(20):
        _ = predictor(text_emb)
    return (time.time() - start) / 20