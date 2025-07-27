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
    config = context['config']['alignment']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    weights = config['weights']
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)
        
        time = prosody['frame_series']['time']
        time_delta = time[1] - time[0] if len(time) > 1 else 0.02
        f0_z = np.array(prosody['frame_series']['f0_z'])
        deltas = np.array(drift['deltas'])
        
        scores = []
        faded_segments = []
        for idx, slice_data in enumerate(transcript['slices']):
            start_time = slice_data['start'] - config['fade_buffer']
            end_time = slice_data['end'] + config['fade_buffer']
            
            slice_len = end_time - start_time
            # True silence score
            words = slice_data.get('words', [])
            speech_len = sum(w['end'] - w['start'] for w in words if 'start' in w and 'end' in w)
            silence_score = 1 - speech_len / slice_len if slice_len > 0 else 0.0
            
            start_idx = max(0, min(len(f0_z) - 1, int(start_time / time_delta)))
            end_idx = max(start_idx + 1, min(len(f0_z), int(end_time / time_delta)))
            prosody_score = np.mean(f0_z[start_idx:end_idx]) if end_idx > start_idx else 0.0
            
            # Lexical polarity from drift deltas mean (as proxy before tier1)
            polarity_score = np.mean(deltas[start_idx:end_idx]) if end_idx > start_idx else 0.0
            
            vad_score = slice_data.get('score', 1.0)
            
            composite = weights['silence'] * silence_score + weights['prosody'] * prosody_score + \
                        weights['polarity'] * polarity_score + weights['vad'] * vad_score
            scores.append(composite)
            
            faded_segments.append({'start': start_time, 'end': end_time})
        
        ranked_indices = np.argsort(scores)[::-1].tolist()
        ranked_faded = [faded_segments[i] for i in ranked_indices]
        
        N = 5  # Top N fallback
        fallback_candidates = ranked_indices[:N]
        
        alignment = {'ranked_slices': ranked_indices, 'scores': [scores[i] for i in ranked_indices], 'faded_segments': ranked_faded, 'fallback_candidates': fallback_candidates}
        
        json_path = os.path.join(speaker_out, 'alignment.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(alignment, f)
            portalocker.unlock(f)
    
    return {'alignment': json_path}