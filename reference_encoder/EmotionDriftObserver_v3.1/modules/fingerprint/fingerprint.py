# modules/fingerprint/fingerprint.py
"""
Speaker fingerprinting with aggregated emotion bias, entropy, slope.
Outputs fingerprint.json.
"""

import json
import os
from collections import Counter
import numpy as np
import portalocker

def run(context):
    for sp in context['speaker_ids']:
        base = os.path.join(context['output_dir'],'emotion_tags',sp)
        with open(os.path.join(base,'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier2 = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(base,'drift_log.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)
        
        labels = [t['label'] for t in tier2]
        dom = Counter(labels).most_common()
        avg_c = np.mean([t['confidence'] for t in tier2])
        counts = Counter(labels)
        probs = [counts[_] / len(labels) for _ in counts]
        ent = -sum(p * np.log(p) for p in probs if p > 0)
        slope = drift.get('confidence_drift_slope',0)
        
        fp = {
          'dominant_tags': dom,
          'avg_confidence': float(avg_c),
          'entropy': float(ent),
          'avg_drift_slope': slope
        }
        with open(os.path.join(base,'fingerprint.json'),'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(fp,f)
            portalocker.unlock(f)
    return {'fingerprint':'fingerprint.json'}