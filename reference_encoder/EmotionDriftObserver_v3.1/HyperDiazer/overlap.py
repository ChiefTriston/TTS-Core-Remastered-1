# overlap.py
"""
Module for overlap detection with multi-feature checks and intra-slice analysis.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import torch
import torch.nn as nn
import logging
import yaml

SAMPLE_RATE = 16000

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class OverlapClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3)  # Improved
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32 * ((SAMPLE_RATE//10)//2), 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

overlap_model = OverlapClassifier()  # TODO: Train and load weights

def detect_overlaps(audio, slices, labels, embs):
    try:
        per_speaker_energy = {label: np.mean(np.concatenate([audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)] for idx, (s, e, _) in enumerate(slices) if labels[idx] == label])**2) for label in np.unique(labels)}
        overlap_energy_thresh = 0.3 * np.median(list(per_speaker_energy.values()))  # Dynamic
        
        overlaps = []
        for i in range(len(slices)):
            s, e, _ = slices[i]
            slice_audio = audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
            if len(slice_audio) < SAMPLE_RATE * 0.1:
                continue
            
            mfcc = librosa.feature.mfcc(y=slice_audio, sr=SAMPLE_RATE, n_mfcc=13)
            delta = librosa.feature.delta(mfcc)
            spectral_flux = librosa.onset.onset_strength(y=slice_audio, sr=SAMPLE_RATE)
            if np.mean(np.abs(delta)) > 0.5 or np.mean(spectral_flux) > 0.5:
                audio_t = torch.from_numpy(slice_audio).float().unsqueeze(0).unsqueeze(0)
                overlap_prob = overlap_model(audio_t).item()
                if overlap_prob > 0.5:
                    overlaps.append((s, e, labels[i], -1, overlap_prob))
            
            if i < len(slices) - 1:
                gap_start, gap_end = slices[i][1], slices[i+1][0]
                gap_audio = audio[int(gap_start * SAMPLE_RATE):int(gap_end * SAMPLE_RATE)]
                gap_energy = np.mean(gap_audio**2)
                norm_energy = gap_energy / max(per_speaker_energy[labels[i]], per_speaker_energy[labels[i+1]])
                if norm_energy > overlap_energy_thresh:
                    sim = cosine_similarity([embs[i]], [embs[i+1]])[0][0]
                    if sim < config['sim_overlap_thresh']:
                        conf = norm_energy * (1 - sim)
                        overlaps.append((gap_start, gap_end, labels[i], labels[i+1], conf))

        return overlaps
    except Exception as e:
        logging.error(f"Error in detect_overlaps: {e}")
        return []