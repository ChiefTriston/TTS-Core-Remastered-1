# modules/alignment/alignment.py
"""
Alignment and composite scoring of slices.
Outputs alignment.json with ranked slices and scores.
"""

import json
import numpy as np
import portalocker
import os

def run(context):
    cfg = context['config']['alignment']
    for sp in context['speaker_ids']:
        base = os.path.join(context['output_dir'],'emotion_tags',sp)
        with open(os.path.join(base,'transcript.json')) as f:  txt = json.load(f)['slices']
        with open(os.path.join(base,'prosody_trend.json')) as f: pros = json.load(f)
        with open(os.path.join(base,'drift_vector.json')) as f: drift = json.load(f)

        scores = []
        time = pros['frame_series']['time']
        step = time[1] - time[0]
        max_len = cfg['max_slice_len']
        fade = cfg['fade_buffer']
        for seg in txt:
            start = max(0.0, seg['start'] - fade)
            end = seg['end'] + fade
            slice_dur = end - start
            silence = 1 - (slice_dur / max_len)

            i0 = int(start/step); i1 = int(end/step)
            prosody_score = abs(np.mean(pros['frame_series']['f0_z'][i0:i1]))

            pol = np.sign(np.mean(drift['deltas'][i0:i1]))

            vad = seg.get('avg_logprob', 0)  

            c = (
                cfg['weights']['silence']*silence
              + cfg['weights']['prosody']*prosody_score
              + cfg['weights']['polarity']*pol
              + cfg['weights']['vad']*vad
            )
            scores.append(c)

        order = list(np.argsort(scores)[::-1])
        out = {'ranked_slices':order,'scores':scores}
        with open(os.path.join(base,'alignment.json'),'w') as f:
            json.dump(out,f)
    return {'alignment':'alignment.json'}