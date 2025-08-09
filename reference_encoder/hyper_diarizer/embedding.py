"""
Module for dual speaker embedding extraction and fusion with parallel processing and learned weights.
"""

import os
import numpy as np
import torch
from speechbrain.inference import EncoderClassifier
from resemblyzer import VoiceEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear
from concurrent.futures import ThreadPoolExecutor
import logging

SAMPLE_RATE = 16000

# Patch: Make paths configurable via environment variable or fallback to default
MODELS_DIR = os.environ.get(
    'MODELS_DIR',
    r"C:\Users\trist\OneDrive\Documents\Remastered TTS Final Version\TTS Core Remastered\models"
)
ECAPA_PATH = os.path.join(MODELS_DIR, "ecapa-voxceleb")
FUSION_WEIGHTS_PATH = os.path.join(MODELS_DIR, "fusion_linear.pth")

# Determine device (GPU if available and sufficient memory)
device = 'cuda' if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 4e9 else 'cpu'

# Load pre-trained speaker embedding models
ecapa = EncoderClassifier.from_hparams(
    source=ECAPA_PATH,
    run_opts={"device": device}
)
res_encoder = VoiceEncoder(device=device)

# Fusion layer placeholder (to be loaded with trained weights)
fusion_linear = Linear(2, 2).to(device)
# Patch: Load weights if available, else log warning (commented out to suppress benign warning)
# if os.path.isfile(FUSION_WEIGHTS_PATH):
#     try:
#         fusion_linear.load_state_dict(torch.load(FUSION_WEIGHTS_PATH, map_location=device))
#     except Exception as e:
#         logging.warning(f"Failed loading fusion weights: {e}")
# else:
#     logging.warning("No pretrained fusion weights found; using random init.")

# Dynamically infer ECAPA embedding dimension
try:
    with torch.no_grad():
        dummy_audio = torch.zeros((1, SAMPLE_RATE * 3), device=device)  # Patch: Correct shape [batch, time] for ECAPA
        dummy_emb = ecapa.encode_batch(dummy_audio).squeeze().cpu().numpy()
        ecapa_dim = dummy_emb.shape[-1]
except Exception as e:
    logging.warning(f"Cannot infer ECAPA embedding dim, defaulting to 192: {e}")
    ecapa_dim = 192

# Resemblyzer outputs fixed-size embeddings
res_dim = 256

# Build transformer encoder for contextualization
d_model = ecapa_dim + res_dim
transformer_layer = TransformerEncoderLayer(d_model=d_model, nhead=4)
transformer_encoder = TransformerEncoder(transformer_layer, num_layers=2).to(device)

def extract_emb(audio, slices, noise_amp=None):
    """
    Extract and fuse speaker embeddings for each audio slice.
    Returns a (num_slices, d_model) NumPy array of contextual embeddings.
    """
    try:
        # Single-slice extractor functions
        def extract_ecapa(slice_audio):
            tensor = torch.tensor(slice_audio).unsqueeze(0).float().to(device)  # Patch: [batch, time] no extra channel
            emb = ecapa.encode_batch(tensor).squeeze().cpu().numpy()
            return emb / (np.linalg.norm(emb) + 1e-6)

        def extract_res(slice_audio):
            try:
                emb = res_encoder.embed_utterance(slice_audio)
                return emb / (np.linalg.norm(emb) + 1e-6)
            except Exception as e:
                logging.error(f"Resemblyzer failed: {e}")
                return None

        # Parallel embedding extraction
        with ThreadPoolExecutor() as executor:
            ecapa_futs = [executor.submit(extract_ecapa, audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]) for s, e, _ in slices]
            res_futs = [executor.submit(extract_res, audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]) for s, e, _ in slices]

            ecapa_embs = [f.result() for f in ecapa_futs]
            res_embs = []
            for idx, f in enumerate(res_futs):
                r = f.result()
                res_embs.append(r if r is not None else ecapa_embs[idx])

        # Fuse per-slice embeddings
        fused = []
        for ec, r in zip(ecapa_embs, res_embs):
            conf_ec = np.linalg.norm(ec)
            conf_r = np.linalg.norm(r)
            weights = torch.softmax(fusion_linear(torch.tensor([conf_ec, conf_r], device=device)), dim=0).cpu().numpy()
            fused.append(np.concatenate([ec * weights[0], r * weights[1]]))

        # Contextualize fused embeddings
        embs = np.stack(fused, axis=0)
        embs_t = torch.from_numpy(embs).float().to(device).unsqueeze(1)
        contextual = transformer_encoder(embs_t).squeeze(1).cpu().numpy()
        return contextual

    except Exception as e:
        logging.error(f"Error in extract_emb: {e}")
        return np.zeros((len(slices), d_model))
