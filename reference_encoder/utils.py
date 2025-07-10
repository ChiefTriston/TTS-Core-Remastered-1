import torch
import torchaudio
from .config import RefEncConfig


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    """
    Load a waveform from `path`, resample if needed, and convert to mono.
    Returns a Tensor of shape (1, N_samples).
    """
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    # convert to mono
    if wav.ndim > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


# reuse a single MelSpectrogram instance for efficiency
_mel_extractor = None

def compute_mel(wav: torch.Tensor, cfg: RefEncConfig) -> torch.Tensor:
    """
    Compute a normalized mel-spectrogram from a mono waveform.
    Input: wav Tensor of shape (1, N)
    Output: Tensor of shape (T_frames, n_mels)
    """
    global _mel_extractor
    if _mel_extractor is None:
        _mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.window_size,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels
        )
    mel = _mel_extractor(wav)
    # normalize per mel bin
    mel = (mel - mel.mean(dim=-1, keepdim=True)) / (mel.std(dim=-1, keepdim=True) + 1e-9)
    # convert (1, n_mels, T) -> (T, n_mels)
    mel = mel.squeeze(0).transpose(0, 1)
    return mel