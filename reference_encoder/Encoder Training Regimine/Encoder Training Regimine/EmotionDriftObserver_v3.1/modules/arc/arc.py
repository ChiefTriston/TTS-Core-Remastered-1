# modules/arc/arc.py
"""
Global arc classification with pivot detection.
Outputs arc_classification.json at job level.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import portalocker
import os

def run(context):
    config = context['config']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    all_tier2 = []
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier2 = json.load(f)
            portalocker.unlock(f)
        all_tier2.extend(tier2)
    
    conf_sequence = np.array([t['confidence'] for t in all_tier2]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=config['arc']['num_clusters']).fit(conf_sequence)
    pivots = np.where(np.diff(kmeans.labels_) != 0)[0]
    
    # Stub arc inference
    dominant_arc = 'hope→betrayal→resignation'
    
    classification = {'pivots': pivots.tolist(), 'dominant_arc': dominant_arc}
    
    json_path = os.path.join(output_dir, 'arc_classification.json')
    with open(json_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(classification, f)
        portalocker.unlock(f)
    
    return {'arc_classification': json_path}