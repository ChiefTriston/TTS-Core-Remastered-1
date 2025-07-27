# modules/fingerprint/fingerprint.py
"""
Speaker fingerprinting with aggregated emotion bias, entropy, slope.
Outputs fingerprint.json.
"""

import json
from collections import Counter
import numpy as np
import portalocker
import os

def run(context):
    config = context['config']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier2 = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'drift_log.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            log = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)
        
        dominant_tags = Counter([t['label'] for t in tier2]).most_common()
        avg_conf = np.mean([t['confidence'] for t in tier2])
        entropy = log['entropy']
        slope = log['confidence_drift_slope']
        
        # Average drift magnitude
        deltas = np.array(drift['deltas'])
        pitch_deltas = deltas[:len(deltas)//2]
        energy_deltas = deltas[len(deltas)//2:]
        avg_drift_magnitude = np.mean(np.abs(np.diff(pitch_deltas))) + np.mean(np.abs(np.diff(energy_deltas)))
        
        # Pause-ratio profile (mean across globals, but since single, use it)
        pause_ratio = prosody['globals']['pause_ratio']  # Assume computed in prosody
        
        # Overall emotion bias (pos - neg count)
        pos_count = sum(1 for t in tier2 if t['label'] in ['joy', 'surprise'])
        neg_count = sum(1 for t in tier2 if t['label'] in ['sadness', 'anger', 'fear'])
        emotion_bias = pos_count - neg_count
        
        fingerprint = {
            'dominant_tags': dominant_tags,
            'avg_confidence': avg_conf,
            'entropy': entropy,
            'slope': slope,
            'avg_drift_magnitude': avg_drift_magnitude,
            'pause_ratio': pause_ratio,
            'emotion_bias': emotion_bias
        }
        
        json_path = os.path.join(speaker_out, 'fingerprint.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(fingerprint, f)
            portalocker.unlock(f)
    
    return {'fingerprint': json_path}