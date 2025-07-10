#!/usr/bin/env python3
import torch
import numpy as np
from sklearn.metrics import roc_curve
from .config import RefEncConfig
from .encoder import ReferenceEncoder
from .dataset import RefEncDataset, load_file_list
from .pad_collate import pad_collate
from torch.utils.data import DataLoader

def evaluate_eer(model, loader, device):
    # put model in eval mode (SSL backbone stays on CPU if configured that way)
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for mel, spk in loader:
            mel = mel.to(device)
            emb = model(mel)
            embeddings.append(emb.cpu())
            if isinstance(spk, torch.Tensor):
                labels.extend(spk.cpu().numpy().tolist())
            else:
                labels.extend(spk)
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = np.array(labels)

    # build pairwise labels and scores
    y_true, y_scores = [], []
    import itertools
    for i, j in itertools.combinations(range(len(labels)), 2):
        y_true.append(int(labels[i] == labels[j]))
        y_scores.append(np.dot(embeddings[i], embeddings[j]))  # cosine similarity

    # safe ROC/EER computation: if roc_curve or nanargmin fails, return inf
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        return (fpr[idx] + fnr[idx]) / 2
    except ValueError:
        # e.g. all positive or all negative pairs → cannot compute ROC/EER
        return float('inf')

if __name__ == '__main__':
    cfg = RefEncConfig()
    model = ReferenceEncoder(cfg)
    ckpt = torch.load('best_model.pt', map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    files = load_file_list('./data/test')
    ds = RefEncDataset(files, cfg, is_train=False)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate
    )

    eer = evaluate_eer(model, loader, torch.device('cpu'))
    print(f"EER: {eer*100:.2f}%")
