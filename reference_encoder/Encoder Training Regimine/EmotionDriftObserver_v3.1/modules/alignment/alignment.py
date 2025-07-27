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
        time = np.array(pros['frame_series']['time'])
        step = time[1] - time[0] if len(time)>1 else 0.02
        for slice in txt:
            silence_score = 1 - ((slice['end'] - slice['start']) / cfg['max_slice_len'])
            start_idx = int(slice['start'] / step)
            end_idx = int(slice['end'] / step)
            prosody_score = np.mean( pros['frame_series']['f0_z'][start_idx:end_idx] )
            vader_scores_in_slice = [{"compound": val} for val in np.array(drift['deltas'])[start_idx:end_idx]]
            polarity_score = np.sign( np.mean([v['compound'] for v in vader_scores_in_slice]) )
            vad_score = slice['score']   # from transcription step
            composite = (
                cfg['weights']['silence']*silence_score +
                cfg['weights']['prosody']*prosody_score +
                cfg['weights']['polarity']*polarity_score +
                cfg['weights']['vad']*vad_score
            )
            scores.append(composite)

        order = list(np.argsort(scores)[::-1])
        out = {'ranked_slices':order,'scores':scores}
        with open(os.path.join(base,'alignment.json'),'w') as f:
            json.dump(out,f)
    return {'alignment':'alignment.json'}