import random
from torch.utils.data import Dataset, Sampler
from .utils import load_audio, compute_mel
from .augment import add_noise, add_reverb, speed_perturb
from torchaudio.transforms import FrequencyMasking, TimeMasking
from pathlib import Path
from typing import List, Tuple

class RefEncDataset(Dataset):
    def __init__(self, files, cfg, is_train=True):
        """
        files: list of tuples (audio_path, speaker_id)
        cfg:   RefEncConfig
        is_train: whether to apply data augmentation
        """
        self.files = files
        self.cfg = cfg
        self.is_train = is_train
        # build a mapping from speaker name → integer label
        all_spks = sorted({spk for _, spk in files})
        self.spk2idx = {spk: i for i, spk in enumerate(all_spks)}

    def __len__(self):
        return len(self.files)

    def _apply_spec_augment(self, mel):
        # Frequency and time masking
        m = mel.transpose(0,1).unsqueeze(0)
        m = FrequencyMasking(freq_mask_param=15)(m)
        m = TimeMasking(time_mask_param=35)(m)
        return m.squeeze(0).transpose(0,1)

    def _sample_another(self):
        idx = random.randrange(len(self.files))
        path, spk = self.files[idx]
        wav = load_audio(path, self.cfg.sample_rate)
        return wav, spk

    def __getitem__(self, idx):
        path, spk_str = self.files[idx]
        wav = load_audio(path, self.cfg.sample_rate)
        spk = self.spk2idx[spk_str]

        # ── for wav2vec2 backbone we never compute mels ──
        if self.cfg.backbone == 'wav2vec2':
            # wav has shape (1, N) → squeeze to (N,)
            return wav.squeeze(0), spk

        # training-time augmentations
        if self.is_train:
            # random crop between 2-4 seconds
            total_len = wav.size(1)
            seg_len = random.randint(2*self.cfg.sample_rate, 4*self.cfg.sample_rate)
            if total_len > seg_len:
                start = random.randint(0, total_len - seg_len)
                wav = wav[:, start:start+seg_len]
            if self.cfg.augment_noise:
                wav = add_noise(wav)
            if self.cfg.augment_reverb:
                wav = add_reverb(wav)
            if self.cfg.speed_perturb:
                wav = speed_perturb(wav)

        # extract mel features
        mel = compute_mel(wav, self.cfg)

        # optional spec augment on mel
        if self.cfg.spec_augment and self.is_train:
            mel = self._apply_spec_augment(mel)

        # optional mixup augmentation
        if self.cfg.mixup and self.is_train:
            wav2, spk2_str = self._sample_another()
            m2 = compute_mel(wav2, self.cfg)
            alpha = random.betavariate(0.4, 0.4)
            mel = alpha * mel + (1 - alpha) * m2
            spk2 = self.spk2idx[spk2_str]
            return mel, (spk, spk2, alpha)

        return mel, spk


class SpeakerBalancedSampler(Sampler):
    """
    Yields batches of indices balanced per speaker.
    """
    def __init__(self, files, spk_per_batch, utts_per_spk):
        # group indices by speaker
        self.by_spk = {}
        for i, (_, spk) in enumerate(files):
            self.by_spk.setdefault(spk, []).append(i)
        self.speakers = list(self.by_spk.keys())
        self.spk_per_batch = spk_per_batch
        self.utts_per_spk = utts_per_spk

    def __iter__(self):
        spks = self.speakers.copy()
        random.shuffle(spks)
        while len(spks) >= self.spk_per_batch:
            batch = []
            selected = spks[:self.spk_per_batch]
            spks = spks[self.spk_per_batch:]
            for spk in selected:
                batch += random.sample(self.by_spk[spk], self.utts_per_spk)
            yield batch

    def __len__(self):
        # number of batches per epoch
        return sum(len(idxs) // self.utts_per_spk for idxs in self.by_spk.values())


def load_file_list(data_dir: str) -> List[Tuple[str, str]]:
    """
    Walks data_dir/<speaker>/*.wav and returns list of (wav_path, speaker_name).
    """
    files = []
    for wav in Path(data_dir).rglob("*.wav"):
        spk = wav.parent.name
        files.append((str(wav), spk))
    if not files:
        raise FileNotFoundError(f"No .wav files found under {data_dir}")
    return files
