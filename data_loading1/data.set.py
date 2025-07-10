#!/usr/bin/env python3
"""
v15-grade TTSDataset (auto-discover speakers/domains/styles)

Return keys:
  wav_path, wav, mel, f0, energy,
  text_ids, text_emb, transcript,
  speaker_id, domain_id, style_id,
  text_length, frame_length
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
import torchaudio

from utils.frontend     import PhonemeFrontend
from utils.audio        import extract_f0_energy, extract_mel_spectrogram
from utils.text_encoder import TextEncoder

logger = logging.getLogger(__name__)

class TTSDataset(Dataset):
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config["audio_root"]      (str)
            config["transcript_root"] (str)
            config["include_speakers"](List[str], optional)
            config["sample_rate"]     (int, optional)
            config["resample"]        (bool, optional)
            config["mel_transform"]   (callable, optional)
            config["frontend"]        (Dict[str,int])
            config["text_encoder"]    (str)
            config["max_frames"]      (int, optional)
        """
        self.audio_root      = Path(config["audio_root"])
        self.transcript_root = Path(config["transcript_root"])
        if not self.audio_root.is_dir():
            raise FileNotFoundError(f"audio_root not found: {self.audio_root}")
        if not self.transcript_root.is_dir():
            raise FileNotFoundError(f"transcript_root not found: {self.transcript_root}")

        self.sample_rate      = config.get("sample_rate", 22050)
        self.resample_enabled = config.get("resample", True)
        self.resampler        = None
        self.mel_transform    = config.get("mel_transform", None)

        self.frontend     = PhonemeFrontend(config.get("frontend", {}))
        self.text_encoder = TextEncoder(config.get("text_encoder",
                                                   "sentence-transformers/all-MiniLM-L6-v2"))
        # NEW: optional cap on how many mel-frames to load per sample
        self.max_frames = config.get("max_frames", None)

        speaker_dirs = [p.name for p in self.audio_root.iterdir() if p.is_dir()]
        self.spk2id = {spk: i for i, spk in enumerate(sorted(speaker_dirs))}
        logger.info(f"Discovered speakers: {sorted(speaker_dirs)}")

        all_domains, all_styles = set(), set()
        for spk in speaker_dirs:
            base = self.audio_root / spk
            for dom in (base.iterdir() if base.exists() else []):
                if dom.is_dir():
                    all_domains.add(dom.name)
                    for sty in dom.iterdir():
                        if sty.is_dir():
                            all_styles.add(sty.name)
        self.dom2id = {d: i for i, d in enumerate(sorted(all_domains))}
        self.sty2id = {s: i for i, s in enumerate(sorted(all_styles))}
        logger.info(f"Discovered domains: {sorted(all_domains)}")
        logger.info(f"Discovered styles : {sorted(all_styles)}")

        self.items: List[Tuple[Path, Path, Tuple[str, ...]]] = []
        for wav_path in sorted(self.audio_root.rglob("*.wav")):
            rel = wav_path.relative_to(self.audio_root)
            txt_path = (self.transcript_root / rel).with_suffix(".txt")
            if txt_path.is_file():
                self.items.append((wav_path, txt_path, rel.parts))
            else:
                logger.warning(f"Missing transcript: {wav_path}")
        if not self.items:
            raise RuntimeError("No valid (wav, txt) pairs found!")
        logger.info(f"Loaded {len(self.items)} examples.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wav_path, txt_path, parts = self.items[idx]

        waveform, sr = torchaudio.load(wav_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        if self.resample_enabled and sr != self.sample_rate:
            if not self.resampler or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self.resampler(waveform)

        f0, energy = extract_f0_energy(waveform, self.sample_rate)
        mel = (self.mel_transform(waveform) if self.mel_transform
               else extract_mel_spectrogram(waveform, self.sample_rate))
        T = mel.shape[0]

        # NEW: if sample is longer than max_frames, cut it down
        if self.max_frames is not None and T > self.max_frames:
            mel    = mel[:self.max_frames]
            f0     = f0[:self.max_frames]
            energy = energy[:self.max_frames]
            T      = self.max_frames

        f0     = self._pad_or_truncate(f0, T)
        energy = self._pad_or_truncate(energy, T)

        raw_text  = txt_path.read_text(encoding="utf-8").strip()
        phonemes  = self.frontend.text_to_phonemes(raw_text)
        token_ids = self.frontend.phonemes_to_sequence(phonemes)
        text_ids  = torch.LongTensor(token_ids)
        text_emb  = self.text_encoder.encode(raw_text)
        text_len  = text_ids.size(0)
        frame_len = T

        speaker = parts[0]
        if speaker not in self.spk2id:
            raise ValueError(f"Unknown speaker '{speaker}' in {wav_path}")
        speaker_id = torch.tensor(self.spk2id[speaker], dtype=torch.long)

        subparts = parts[:-1]
        domain   = subparts[1] if len(subparts) > 1 else None
        style    = subparts[2] if len(subparts) > 2 else None

        domain_id = torch.tensor(self.dom2id.get(domain, 0), dtype=torch.long)
        style_id  = torch.tensor(self.sty2id.get(style, 0),  dtype=torch.long)

        return {
            "wav_path":     wav_path,
            "wav":          waveform,
            "mel":          mel,
            "f0":           f0,
            "energy":       energy,
            "text_ids":     text_ids,
            "text_emb":     text_emb,
            "transcript":   raw_text,
            "speaker_id":   speaker_id,
            "domain_id":    domain_id,
            "style_id":     style_id,
            "text_length":  text_len,
            "frame_length": frame_len,
        }

    @staticmethod
    def _pad_or_truncate(x: torch.Tensor, length: int) -> torch.Tensor:
        if x.size(0) < length:
            return torch.cat([x, x.new_zeros(length - x.size(0))], dim=0)
        return x[:length]


if __name__ == "__main__":
    cfg = {
        "audio_root":      "C:/.../Remastered TTS Final Version/audio",
        "transcript_root": "C:/.../Remastered TTS Final Version/transcripts",
        "frontend":        {"pad_id": 0, "bos_id": 1, "eos_id": 2},
        "text_encoder":    "sentence-transformers/all-MiniLM-L6-v2",
        "max_frames":      200,
    }
    ds = TTSDataset(cfg)
    for i in range(3):
        s = ds[i]
        print({k: (v.shape if isinstance(v, torch.Tensor) else v) for k, v in s.items()})

