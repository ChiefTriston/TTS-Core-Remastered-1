# embedding.py
"""
Module for dual speaker embedding extraction and fusion with parallel processing and learned weights.
"""

import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier
from resemblyzer import VoiceEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear
from concurrent.futures import ThreadPoolExecutor
import logging
import yaml

SAMPLE_RATE = 16000

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 4e9 else 'cpu'

ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb").to(device)
resemblyzer = VoiceEncoder(device=device)

# Learned fusion (train separately)
fusion_linear = Linear(2, 2).to(device)  # Placeholder, load trained weights

transformer_layer = TransformerEncoderLayer(d_model=ecapa.embedding_dim + 256, nhead=4)  # Dynamic dims
transformer_encoder = TransformerEncoder(transformer_layer, num_layers=2).to(device)

def extract_emb(audio, slices, noise_amp=None):
    try:
        def extract_single_ecapa(slice_audio):
            slice_t = torch.tensor(slice_audio).unsqueeze(0).unsqueeze(0).float().to(device)
            emb = ecapa.encode_batch(slice_t).squeeze().cpu().numpy()
            emb /= np.linalg.norm(emb) + 1e-6
            return emb

        def extract_single_res(slice_audio):
            try:
                emb = resemblyzer.embed_utterance(slice_audio)
                emb /= np.linalg.norm(emb) + 1e-6
                return emb
            except Exception as e:
                logging.error(f"Resemblyzer failed: {e}")
                return None  # Skip or fallback

        with ThreadPoolExecutor() as executor:
            ecapa_futures = [executor.submit(extract_single_ecapa, audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]) for s, e, _ in slices]
            res_futures = [executor.submit(extract_single_res, audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]) for s, e, _ in slices]
            
            batch_ecapa = [f.result() for f in ecapa_futures]
            batch_res = []
            for f in res_futures:
                res = f.result()
                if res is None:
                    res = batch_ecapa[len(batch_res)]  # Fallback to ECAPA
                batch_res.append(res)

        embs = []
        for ec, res in zip(batch_ecapa, batch_res):
            conf_ec = np.linalg.norm(ec)  # Better metric?
            conf_res = np.linalg.norm(res)
            weights = torch.softmax(fusion_linear(torch.tensor([conf_ec, conf_res]).to(device)), dim=0).cpu().numpy()
            fused = np.concatenate([ec * weights[0], res * weights[1]])
            embs.append(fused)

        embs = np.array(embs)
        embs_t = torch.from_numpy(embs).float().to(device).unsqueeze(1)
        contextual_embs = transformer_encoder(embs_t).squeeze(1).cpu().numpy()
        
        return contextual_embs
    except Exception as e:
        logging.error(f"Error in extract_emb: {e}")
        return np.zeros((len(slices), ecapa.embedding_dim + 256))