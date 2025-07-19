# dataset.py
from torch.utils.data import Dataset
import json
import torch
import random
from utils import load_audio, compute_mel

class RefEncDataset(Dataset):
    def __init__(self, json_file, cfg, is_train=True):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.cfg = cfg
        self.is_train = is_train
        self.spk2idx = {item['speaker_id']: i for i, item in enumerate(self.data)}
        self.emotion_labels = [None for _ in self.data]  # Placeholder
    
    def __getitem__(self, idx):
        item = self.data[idx]
        wav = load_audio(item['audio_path'], self.cfg.sample_rate)
        mel = compute_mel(wav, self.cfg) if self.cfg.backbone != 'wav2vec2' else wav.squeeze(0)
        vader_scores = torch.tensor(list(item['vader_scores'].values()), dtype=torch.float32)
        prosody_features = torch.tensor([
            item['prosody_features']['f0'],
            item['prosody_features']['energy'],
            item['prosody_features']['pitch_var'],
            item['prosody_features']['speech_rate'][0],
            item['prosody_features']['pause_dur'][0],
            *item['prosody_features']['mfcc']
        ], dtype=torch.float32)
        spk_idx = self.spk2idx[item['speaker_id']]
        emotions = torch.zeros(6) if self.emotion_labels[idx] is None else torch.tensor(self.emotion_labels[idx])
        if self.is_train and self.cfg.mixup:
            idx2 = random.randrange(len(self.data))
            item2 = self.data[idx2]
            mel2 = compute_mel(load_audio(item2['audio_path'], self.cfg.sample_rate), self.cfg)
            alpha = random.betavariate(0.4, 0.4)
            vader2 = torch.tensor(list(item2['vader_scores'].values()), dtype=torch.float32)
            prosody2 = torch.tensor([
                item2['prosody_features']['f0'],
                item2['prosody_features']['energy'],
                item2['prosody_features']['pitch_var'],
                item2['prosody_features']['speech_rate'][0],
                item2['prosody_features']['pause_dur'][0],
                *item2['prosody_features']['mfcc']
            ], dtype=torch.float32)
            return mel, (spk_idx, emotions, vader_scores, prosody_features, vader2, prosody2, alpha)
        return mel, (spk_idx, emotions, vader_scores, prosody_features)
    
    def __len__(self):
        return len(self.data)