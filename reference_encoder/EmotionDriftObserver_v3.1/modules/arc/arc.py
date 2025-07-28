# modules/arc/arc.py
"""
Global arc classification with pivot detection.
Outputs arc_classification.json at job level.
"""

import json
import os
import numpy as np
from sklearn.cluster import KMeans
import portalocker
from collections import Counter

def infer_named_arc(sequence):
    """
    Map a sequence of dominant emotion labels to a named narrative arc.
    Extend `patterns` as needed.
    """
    patterns = {
        ('hope', 'betrayal', 'resignation'): 'hope→betrayal→resignation',
        ('surprise', 'sadness'): 'excitement→despair',
        # add more patterns here
    }
    return patterns.get(tuple(sequence), 'custom')

def run(context):
    cfg         = context['config']['arc']
    out_base    = context['output_dir']
    speaker_ids = context['speaker_ids']

    all_labels = []
    all_confs  = []
    all_times  = []

    # Gather every slice's label, confidence, and timestamp across speakers
    for spk in speaker_ids:
        spk_dir = os.path.join(out_base, 'emotion_tags', spk)

        with open(os.path.join(spk_dir, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier2 = json.load(f)
            portalocker.unlock(f)

        with open(os.path.join(spk_dir, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)

        slices = transcript.get('slices', [])
        for idx, tag in enumerate(tier2):
            all_labels.append(tag['label'])
            all_confs.append(tag['confidence'])
            if idx < len(slices):
                all_times.append((slices[idx]['start'], slices[idx]['end']))
            else:
                all_times.append((0.0, 0.0))

    # Prepare confidence array for clustering
    conf_array = np.array(all_confs).reshape(-1, 1)

    # If no variability, treat entire session as neutral
    if len(all_confs) == 0 or np.std(all_confs) < 1e-6:
        pivots   = []
        arc_segs = [{
            'segment': 'neutral',
            'start': all_times[0][0] if all_times else 0.0,
            'end':   all_times[-1][1] if all_times else 0.0
        }]
    else:
        # Pivot detection via KMeans
        k      = cfg.get('num_clusters', 3)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(conf_array)
        labels = kmeans.labels_
        # find indices where cluster label changes
        changes   = np.where(np.diff(labels) != 0)[0] + 1
        pivots    = [ all_times[i][0] for i in changes ]

        # Build per‑segment arcs
        arc_segs = []
        start_idx = 0
        for end_idx in changes:
            seg_labels = all_labels[start_idx:end_idx]
            seg_times  = all_times[start_idx:end_idx]
            dominant   = Counter(seg_labels).most_common(1)[0][0] if seg_labels else 'neutral'
            seg_start  = seg_times[0][0] if seg_times else 0.0
            seg_end    = seg_times[-1][1] if seg_times else 0.0
            arc_segs.append({'segment': dominant, 'start': seg_start, 'end': seg_end})
            start_idx = end_idx
        # last segment
        seg_labels = all_labels[start_idx:]
        seg_times  = all_times[start_idx:]
        dominant   = Counter(seg_labels).most_common(1)[0][0] if seg_labels else 'neutral'
        seg_start  = seg_times[0][0] if seg_times else 0.0
        seg_end    = seg_times[-1][1] if seg_times else 0.0
        arc_segs.append({'segment': dominant, 'start': seg_start, 'end': seg_end})

    # Name the arc
    seq       = [seg['segment'] for seg in arc_segs]
    named_arc = infer_named_arc(seq)

    classification = {
        'pivots':         pivots,
        'arc_segments':   arc_segs,
        'named_arc':      named_arc
    }

    # Write to JSON
    out_path = os.path.join(out_base, 'arc_classification.json')
    with open(out_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(classification, f, indent=2)
        portalocker.unlock(f)

    return {'arc_classification': out_path}
