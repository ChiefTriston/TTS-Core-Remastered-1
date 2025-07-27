# modules/anomaly/anomaly.py
"""
Anomaly flagging for hallucinations and VADER outliers.
Injects into drift_vector.json, updates drift_log.json.
"""

import json
import numpy as np
import portalocker
import os
from collections import Counter

def run(context):
    config = context['config']['anomaly']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'tier1_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier1 = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier2 = json.load(f)
            portalocker.unlock(f)
        
        anomalies = []
        for idx, slice_data in enumerate(transcript['slices']):
            text = slice_data['text']
            # Whisper hallucination
            if len(text) < config['hallucination_min_len']:
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx, 'reason': 'short_text'})
            char_counts = Counter(text)
            if max(char_counts.values()) / len(text) > config['repetition_thresh']:
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx, 'reason': 'repetitive'})
            
            # VADER anomaly
            compound = tier1[idx]['compound']
            compounds = [t['compound'] for t in tier1]
            mean_comp = np.mean(compounds)
            std_comp = np.std(compounds)
            if abs(compound - mean_comp) > config['outlier_std_mult'] * std_comp:
                anomalies.append({'type': 'vader_anomaly', 'slice': idx})
        
        # Inject to drift_vector
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            drift = json.load(f)
            f.seek(0)
            drift['anomalies'] = anomalies
            json.dump(drift, f)
            f.truncate()
            portalocker.unlock(f)
        
        # Update drift_log with entropy and slope
        labels = [t['label'] for t in tier2]
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        confidences = np.array([t['confidence'] for t in tier2])
        time = np.arange(len(confidences))
        slope = np.polyfit(time, confidences, 1)[0] if len(confidences) > 1 else 0.0
        
        with open(os.path.join(speaker_out, 'drift_log.json'), 'r+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            log = json.load(f)
            f.seek(0)
            log['emotion_entropy'] = entropy
            log['confidence_drift_slope'] = slope
            json.dump(log, f)
            f.truncate()
            portalocker.unlock(f)
    
    return {'updated_drift': True}