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
        ranked_slices = []
        for idx, slice_data in enumerate(transcript['slices']):
            slice_len = slice_data['end'] - slice_data['start']
            silence_score = 1 - slice_len / config['max_slice_len']
            start_idx = max(0, min(len(f0_z) - 1, int(slice_data['start'] / time_delta)))
            end_idx = max(start_idx + 1, min(len(f0_z), int(slice_data['end'] / time_delta)))
            prosody_score = np.mean(f0_z[start_idx:end_idx]) if end_idx > start_idx else 0.0
            polarity_score = np.sign(np.mean(deltas[start_idx:end_idx]) if end_idx > start_idx else 0.0)
            vad_score = slice_data.get('score', 1.0)
            
            composite = weights['silence'] * silence_score + weights['prosody'] * prosody_score + \
                        weights['polarity'] * polarity_score + weights['vad'] * vad_score
            scores.append(composite)
            
            # Add fade buffers
            faded_start = slice_data['start'] - config['fade_buffer']
            faded_end = slice_data['end'] + config['fade_buffer']
            ranked_slices.append({'original': slice_data, 'faded_start': faded_start, 'faded_end': faded_end})
        
        ranked_indices = np.argsort(scores)[::-1].tolist()
        ranked_slices_sorted = [ranked_slices[i] for i in ranked_indices]
        
        alignment = {'ranked_slices': ranked_slices_sorted, 'scores': [scores[i] for i in ranked_indices]}
        
        json_path = os.path.join(speaker_out, 'alignment.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(alignment, f)
            portalocker.unlock(f)
    
    return {'alignment': json_path}