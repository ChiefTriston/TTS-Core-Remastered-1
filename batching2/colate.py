#!/usr/bin/env python3

"""
v15-grade TTSCollator: Production-ready, thread-safe, and extensible

Features:
  - Length-bucketed batching via BucketBatchSampler (import from utils.batch_samplers)
  - On-the-fly waveform augments (speed, noise) and SpecAugment batch transform
  - Thread-safe dynamic feature caching with flexible cache_key_fn
  - Mixed-precision casting when AMP & CUDA available
  - Structured logging for collate profiling
  - Reproducible batch-level augment seeding
  - Exposes collate_time metric
  - Optional async pre-augmentation via ThreadPoolExecutor

Sampler example:
    from utils.batch_samplers import BucketBatchSampler
    sampler = BucketBatchSampler(
        dataset, batch_size, key_fn=lambda i: dataset.items[i][2]  # frame_length
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
"""
import time
import threading
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

class AugmentationPipeline(Module):
    """Compose and randomize multiple augment transforms each call."""
    def __init__(self, transforms: List[Module]):
        super().__init__()
        self.transforms = transforms

    def forward(self, x: Tensor) -> Tensor:
        # Randomize order for maximal diversity
        transforms = random.sample(self.transforms, k=len(self.transforms))
        for t in transforms:
            x = t(x)
        return x

class TTSCollator:
    def __init__(
        self,
        pad_id: int = 0,
        use_amp: bool = False,
        cache_enabled: bool = False,
        cache_key_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        wav_augment: Optional[Module] = None,
        noise_augment: Optional[Module] = None,
        spec_augment: Optional[Module] = None,
        seed_fn: Optional[Callable[[int, int], int]] = None,
        async_pre: bool = False
    ):
        self.pad_id = pad_id
        self.use_amp = use_amp and torch.cuda.is_available()
        self.cache_enabled = cache_enabled
        self.cache_key_fn = cache_key_fn or (lambda s: s.get('wav_path'))
        self.wav_augment = wav_augment
        self.noise_augment = noise_augment
        self.spec_augment = spec_augment
        self.seed_fn = seed_fn
        self.async_pre = async_pre
        self._cache: Dict[Any, Dict[str, Tensor]] = {}
        self._lock = threading.Lock()
        if async_pre:
            self._executor = ThreadPoolExecutor(max_workers=4)

    def __call__(
        self,
        batch: List[Dict[str, Any]],
        epoch: int = 0,
        batch_idx: int = 0
    ) -> Dict[str, Union[Tensor, List[str], float]]:
        """Batch and return {Tensor, List[str], float}"""
        if not batch:
            raise ValueError("Empty batch passed to TTSCollator")

        # Profiling timer
        if self.use_amp:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
        else:
            start_time = time.time()

        # Reproducible augment seeding
        if self.seed_fn:
            seed = self.seed_fn(epoch, batch_idx)
            torch.manual_seed(seed)

        # Sample-level augment & caching (sync or async)
        def process(sample: Dict[str, Any]):
            key = self.cache_key_fn(sample)
            if self.cache_enabled:
                with self._lock:
                    cached = self._cache.get(key)
                if cached:
                    sample.update(cached)
                    return
            wav: Tensor = sample['wav']
            aug_transforms = []
            if self.wav_augment:
                aug_transforms.append(self.wav_augment)
            if self.noise_augment:
                aug_transforms.append(self.noise_augment)
            if aug_transforms:
                wav = AugmentationPipeline(aug_transforms)(wav)
            sample['wav'] = wav
            if self.cache_enabled and key is not None:
                with self._lock:
                    self._cache[key] = {
                        'wav': wav,
                        'mel': sample['mel'],
                        'f0': sample['f0'],
                        'energy': sample['energy']
                    }

        if self.async_pre:
            list(self._executor.map(process, batch))
        else:
            for sample in batch:
                process(sample)

        # Collect tensors
        wavs = [s['wav'] for s in batch]
        wav_lengths = torch.tensor([w.size(0) for w in wavs], dtype=torch.int64)
        wavs_padded = pad_sequence(wavs, batch_first=True)

        mels = [s['mel'] for s in batch]
        frame_lengths = torch.tensor([m.size(0) for m in mels], dtype=torch.int64)
        mels_padded = pad_sequence(mels, batch_first=True)
        if self.spec_augment:
            mels_padded = self.spec_augment(mels_padded)

        f0s = [s['f0'] for s in batch]
        f0_padded = pad_sequence(f0s, batch_first=True).unsqueeze(-1)
        energies = [s['energy'] for s in batch]
        energy_padded = pad_sequence(energies, batch_first=True).unsqueeze(-1)

        text_ids_list = [s['text_ids'] for s in batch]
        text_lengths = torch.tensor([t.size(0) for t in text_ids_list], dtype=torch.int64)
        text_ids_padded = pad_sequence(
            text_ids_list, batch_first=True, padding_value=self.pad_id
        )
        text_mask = text_ids_padded.ne(self.pad_id)

        text_embs = torch.stack([s['text_emb'] for s in batch], dim=0)
        speaker_ids = torch.cat([s['speaker_id'].view(1) for s in batch], dim=0)
        domain_ids = torch.cat([s['domain_id'].view(1) for s in batch], dim=0)
        style_ids = torch.cat([s['style_id'].view(1) for s in batch], dim=0)
        transcripts = [s['transcript'] for s in batch]

        # Mixed-precision
        if self.use_amp:
            mels_padded = mels_padded.half()
            f0_padded = f0_padded.half()
            energy_padded = energy_padded.half()
            text_embs = text_embs.half()

        # End profiling
        if self.use_amp:
            end_evt.record()
            torch.cuda.synchronize()
            collate_time = start_evt.elapsed_time(end_evt) / 1000.0
        else:
            collate_time = time.time() - start_time
        logger.debug("Batch collated in %.3fs", collate_time)

        return {
            'wav': wavs_padded,
            'wav_length': wav_lengths,
            'mel': mels_padded,
            'f0': f0_padded,
            'energy': energy_padded,
            'frame_length': frame_lengths,
            'text_ids': text_ids_padded,
            'text_mask': text_mask,
            'text_length': text_lengths,
            'text_emb': text_embs,
            'speaker_id': speaker_ids,
            'domain_id': domain_ids,
            'style_id': style_ids,
            'transcripts': transcripts,
            'collate_time': collate_time,
        }