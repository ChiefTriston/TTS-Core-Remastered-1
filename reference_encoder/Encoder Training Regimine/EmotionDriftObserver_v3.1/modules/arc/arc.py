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
from collections import Counter

def run(context):
    config = context['config']['arc']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    all_tier2 = []
    all_times = []  # Collect (start, end) for each slice
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier2 = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        all_tier2.extend(tier2)
        all_times.extend([(s['start'], s['end']) for s in transcript['slices']])
    
    confidences = np.array([t['confidence'] for t in all_tier2])
    conf_sequence = confidences.reshape(-1, 1)
    
    # Edge-case: if variance low, neutral arc
    if np.std(confidences) < 0.1:
        classification = {'pivots': [], 'arc_segments': [{'segment': 'neutral', 'start': all_times[0][0] if all_times else 0, 'end': all_times[-1][1] if all_times else 0}]}
    else:
        kmeans = KMeans(n_clusters=config['num_clusters']).fit(conf_sequence)
        labels = kmeans.labels_
        pivots = np.where(np.diff(labels) != 0)[0] + 1  # Adjust for diff offset
        
        # Map to timestamps
        pivot_times = [all_times[p]['start'] for p in pivots] if pivots.size > 0 else []
        
        # Semantic arc inference
        label_sequence = [t['label'] for t in all_tier2]
        arc_segments = []
        start_idx = 0
        for p in pivots:
            segment_labels = label_sequence[start_idx:p]
            dominant = Counter(segment_labels).most_common(1)[0][0] if segment_labels else 'neutral'
            start_time = all_times[start_idx][0]
            end_time = all_times[p-1][1] if p > 0 else all_times[-1][1]
            arc_segments.append({'segment': dominant, 'start': start_time, 'end': end_time})
            start_idx = p
        # Last segment
        segment_labels = label_sequence[start_idx:]
        dominant = Counter(segment_labels).most_common(1)[0][0] if segment_labels else 'neutral'
        start_time = all_times[start_idx][0] if start_idx < len(all_times) else 0
        end_time = all_times[-1][1] if all_times else 0
        arc_segments.append({'segment': dominant, 'start': start_time, 'end': end_time})
        
        # Pattern matching for named arcs
        seq = [seg['segment'] for seg in arc_segments]
        patterns = {
            'hope→betrayal': ['pos', 'neg'],
            'excitement→despair': ['surprise', 'sadness'],
            # Add more
        }
        named_arc = next((name for name, pat in patterns.items() if seq == pat), 'custom')
        
        classification = {'pivots': pivot_times, 'arc_segments': arc_segments, 'named_arc': named_arc}
    
    json_path = os.path.join(output_dir, 'arc_classification.json')
    with open(json_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(classification, f)
        portalocker.unlock(f)
    
    return {'arc_classification': json_path}

import json, os
import numpy as np
from sklearn.cluster import KMeans

def run(context):
    all_conf = []
    all_times = []
    for sp in context['speaker_ids']:
        data = json.load(open(os.path.join(context['output_dir'],'emotion_tags',sp,'tier2_tags.json')))
        # assume each slice has 'time' & 'confidence'
        all_conf += [(seg['start'],seg['confidence']) for seg in data]
    times, confs = zip(*all_conf)
    arr = np.array(confs).reshape(-1,1)
    k = context['config']['arc']['num_clusters']
    km = KMeans(n_clusters=k).fit(arr)
    pivots = np.where(np.diff(km.labels_)!=0)[0].tolist()
    
    arc = {
      'pivots': pivots,
      'arc': " → ".join(
         ["hope","betrayal","resignation"]  # derive from tags or hardcode
      )
    }
    with open(os.path.join(context['output_dir'],'arc_classification.json'),'w') as f:
        json.dump(arc,f)
    return {'arc_classification':'arc_classification.json'}