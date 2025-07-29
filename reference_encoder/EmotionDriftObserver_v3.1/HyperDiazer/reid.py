# reid.py
"""
Module for speaker re-identification using adaptive memory bank with anomaly detection.
"""

import numpy as np
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from bidict import bidict
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import logging
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

EMB_DIM = 448  # Update dynamically if possible
VOICEPRINT_THRESH = config['voiceprint_thresh']
MEMORY_SIZE = config['memory_size']

class ReIDMemory:
    def __init__(self, thresh=VOICEPRINT_THRESH, memory_size=MEMORY_SIZE):
        self.memory = {}
        self.label_map = bidict()
        self.thresh = thresh
        self.memory_size = memory_size
        transformer_layer = TransformerEncoderLayer(d_model=EMB_DIM, nhead=4)
        self.transformer = TransformerEncoder(transformer_layer, num_layers=1)  # Reduced layers

    def re_id(self, embs, labels, use_transformer=True):
        try:
            if not labels.size or np.any(labels < 0):
                logging.error("Invalid labels")
                return np.zeros_like(labels), {}, np.zeros(len(labels))
            
            unique_labels = np.unique(labels)
            new_labels = np.copy(labels)
            certainties = np.zeros(len(labels))
            for label in unique_labels:
                mask = labels == label
                slice_embs = embs[mask]
                if len(slice_embs) > 1:
                    clf = IsolationForest(contamination=0.1)
                    outliers = clf.fit_predict(slice_embs) == -1
                    slice_embs = slice_embs[~outliers]
                avg_emb = np.mean(slice_embs, axis=0) if len(slice_embs) > 0 else np.zeros(EMB_DIM)
                
                best_id = None
                best_score = -1
                for mem_id, mem_embs in self.memory.items():
                    mem_avg = np.mean(mem_embs, axis=0)
                    score = cosine_similarity([avg_emb], [mem_avg])[0][0]
                    if score > best_score:
                        best_score = score
                        best_id = mem_id
                if best_id and best_score > self.thresh:
                    existing_label = self.label_map.inverse.get(best_id)
                    new_labels[mask] = existing_label
                    self.memory[best_id].extend(slice_embs.tolist())
                    self.memory[best_id] = self.memory[best_id][-self.memory_size:]
                    certainties[mask] = best_score
                else:
                    new_uuid = str(uuid.uuid4())
                    self.memory[new_uuid] = list(slice_embs)
                    self.label_map[label] = new_uuid
                    certainties[mask] = 1.0

            if use_transformer:
                embs_t = torch.from_numpy(embs).float().unsqueeze(1)
                smoothed_embs = self.transformer(embs_t).squeeze(1).numpy()
                embs = smoothed_embs

            return new_labels, dict(self.label_map), certainties
        except Exception as e:
            logging.error(f"Error in re_id: {e}")
            return labels, {}, np.zeros(len(labels))