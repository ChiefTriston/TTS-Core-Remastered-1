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
        
        dominant_tags = Counter([t['label'] for t in tier2]).most_common()
        avg_conf = np.mean([t['confidence'] for t in tier2])
        entropy = log['entropy']
        slope = log['confidence_drift_slope']
        
        fingerprint = {
            'dominant_tags': dominant_tags,
            'avg_confidence': avg_conf,
            'entropy': entropy,
            'slope': slope
        }
        
        json_path = os.path.join(speaker_out, 'fingerprint.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(fingerprint, f)
            portalocker.unlock(f)
    
    return {'fingerprint': json_path}