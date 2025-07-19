```python
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler
from tqdm import trange, tqdm
from acoustic4.config import AcousticConfig
from acoustic4.model import AcousticModel
from acoustic4.losses import CompositeLoss
from prosody3.dataset import RefEncDataset
from prosody3.pad_collate import pad_collate
from prosody3.observer_module import ObserverModule
from utils import load_audio, compute_mel

def train_acoustic(json_files, cfg, num_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = [RefEncDataset(json_file, cfg, is_train=True) for json_file in json_files]
    dataset = ConcatDataset(datasets)
    sampler = SpeakerBalancedSampler(
        [(item['audio_path'], item['speaker_id'], None) for ds in datasets for item in ds.data],
        spk_per_batch=cfg.batch_size // cfg.speaker_batch_utterances,
        utts_per_spk=cfg.speaker_batch_utterances
    )
    loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler, cfg.batch_size, drop_last=True),
                        collate_fn=pad_collate, num_workers=4)
    
    acoustic = AcousticModel(cfg).to(device)
    observer = ObserverModule().to(device)
    observer.load_state_dict(torch.load('observer_module.pt', map_location=device))
    loss_fn = CompositeLoss(mel_weight=1.0, mse_weight=1.0, disc_weight=0.1, diff_weight=0.1, emo_weight=0.1)
    optimizer = AdamW(acoustic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    for epoch in trange(num_epochs, desc="Epochs"):
        total_loss = 0
        acoustic.train()
        observer.eval()
        for batch_idx, (mel, data) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            if cfg.mixup:
                spk, _, vader, prosody, vader2, prosody2, alpha = data
                vader, prosody = vader.to(device), prosody.to(device)
            else:
                spk, _, vader, prosody = data
                vader, prosody = vader.to(device), prosody.to(device)
            
            # Get emotion probabilities
            with torch.no_grad():
                emotion_probs, _, _ = observer(vader, prosody)
            
            # Prepare inputs
            text_emb = torch.randn(len(spk), mel.size(1), cfg.text_emb_dim).to(device)
            speaker = torch.randn(len(spk), cfg.speaker_dim).to(device)
            mel = mel.to(device)
            
            # Forward pass
            pred_mel, real_logits, fake_logits, real_features, fake_features, noise_pred, duration, pitch, energy = \
                acoustic(text_emb, prosody, emotion_probs, mel, speaker)
            
            # Compute loss
            noise_true = torch.randn_like(noise_pred)
            loss = loss_fn(pred_mel, mel, real_logits, fake_logits, real_features, fake_features, noise_pred, noise_true, emotion_probs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
    
    torch.save(acoustic.state_dict(), 'acoustic_model.pt')
    return acoustic

if __name__ == "__main__":
    cfg = AcousticConfig(text_emb_dim=128, speaker_dim=16)
    cfg.batch_size = 8
    cfg.lr = 1e-4
    cfg.weight_decay = 1e-5
    cfg.speaker_batch_utterances = 2
    acoustic = train_acoustic(
        [r'baseline_vader_prosody.json', r'golden_vader_prosody.json'],
        cfg
    )
```