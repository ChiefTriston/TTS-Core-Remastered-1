# modules/anomaly/anomaly.py
"""
Anomaly flagging for hallucinations and VADER outliers.
Injects into drift_vector.json, updates drift_log.json.
"""

import json
import numpy as np
import portalocker
import os

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
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx})
            elif len(set(text)) / len(text) < 0.2:  # Low diversity
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx})
            
            # VADER anomaly
            compound = tier1[idx]['compound']
            compounds = [t['compound'] for t in tier1]
            mean = np.mean(compounds)
            std = np.std(compounds)
            if abs(compound - mean) > config['outlier_std_mult'] * std:
                anomalies.append({'type': 'vader_anomaly', 'slice': idx})
        
        # Inject
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            drift = json.load(f)
            f.seek(0)
            drift['anomalies'] = anomalies
            json.dump(drift, f)
            f.truncate()
            portalocker.unlock(f)
        
        # Entropy and slope
        labels = [t['label'] for t in tier2]
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        emotion_entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        confidences = [t['confidence'] for t in tier2]
        if len(confidences) > 1:
            slope = np.polyfit(range(len(confidences)), confidences, 1)[0]
        else:
            slope = 0.0
        
        with open(os.path.join(speaker_out, 'drift_log.json'), 'r+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            log = json.load(f)
            f.seek(0)
            log['emotion_entropy'] = emotion_entropy
            log['confidence_drift_slope'] = slope
            json.dump(log, f)
            f.truncate()
            portalocker.unlock(f)
    
    return {'updated_drift': True}