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
    config = context['config']['arc']
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
    kmeans = KMeans(n_clusters=config['num_clusters']).fit(conf_sequence)
    pivots = np.where(np.diff(kmeans.labels_) != 0)[0]
    
    # Infer arc from label transitions
    label_sequence = [t['label'] for t in all_tier2]
    transitions = []
    for i in range(1, len(label_sequence)):
        if label_sequence[i] != label_sequence[i-1]:
            transitions.append(f"{label_sequence[i-1]}→{label_sequence[i]}")
    
    # Map to arcs
    arc_map = {
        'joy→sadness': 'hope→betrayal',
        'surprise→fear': 'excitement→disappointment',
        # Add more mappings
    }
    dominant_arc = ' '.join(transitions) if not transitions else arc_map.get(transitions[0], 'unknown')
    for trans in transitions[1:]:
        dominant_arc += '→' + arc_map.get(trans, 'unknown').split('→')[1]
    
    classification = {'pivots': pivots.tolist(), 'dominant_arc': dominant_arc}
    
    json_path = os.path.join(output_dir, 'arc_classification.json')
    with open(json_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(classification, f)
        portalocker.unlock(f)
    
    return {'arc_classification': json_path}    
    return {'arc_classification': json_path}