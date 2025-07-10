import random
import torch
import torchaudio
import torchaudio.functional as F

# List of filepaths for augmentation sources; populate as needed
NOISE_FILES = []  # e.g. ['noise1.wav', 'noise2.wav']
RIR_FILES   = []  # e.g. ['rir1.wav', 'rir2.wav']

def add_noise(wav: torch.Tensor, snr_range=(5, 20)) -> torch.Tensor:
    """
    Add background noise at a random SNR between snr_range.
    wav: Tensor (1, N)
    """
    if not NOISE_FILES:
        return wav
    noise, sr = torchaudio.load(random.choice(NOISE_FILES))
    # convert to mono if needed
    if noise.ndim > 1:
        noise = noise.mean(dim=0, keepdim=True)
    # resample noise if sr mismatches
    target_sr = wav.shape[1]  # assuming wav length == sample rate here
    if sr != target_sr and hasattr(F, 'resample'):
        noise = F.resample(noise, sr, target_sr)
    # ensure noise length >= wav length
    if noise.size(1) < wav.size(1):
        repeat = (wav.size(1) // noise.size(1)) + 1
        noise = noise.repeat(1, repeat)
    noise = noise[:, :wav.size(1)]
    # compute scaling factor for desired SNR
    sig_p = wav.pow(2).mean()
    noise_p = noise.pow(2).mean()
    snr = random.uniform(*snr_range)
    factor = (sig_p / (noise_p * 10**(snr / 10))).sqrt()
    return wav + noise * factor

def add_reverb(wav: torch.Tensor) -> torch.Tensor:
    """
    Convolve waveform with a random room impulse response (RIR).
    """
    if not RIR_FILES:
        return wav
    rir, sr = torchaudio.load(random.choice(RIR_FILES))
    if rir.ndim > 1:
        rir = rir.mean(dim=0, keepdim=True)
    rir = rir / rir.norm(p=2)
    # apply convolution
    return torch.nn.functional.conv1d(
        wav.unsqueeze(0), rir.unsqueeze(1), padding=rir.size(1) // 2
    ).squeeze(0)

def speed_perturb(wav: torch.Tensor, rates=(0.9, 1.1)) -> torch.Tensor:
    """
    Speed perturb by resampling via sox effects.
    """
    rate = random.choice(rates)
    effects = [['speed', str(rate)], ['rate', '22050']]
    wav_sp, _ = torchaudio.sox_effects.apply_effects_tensor(wav, 22050, effects)
    return wav_sp
