# modules/drift/drift.py (updated with config lookup)
"""
Drift computation based on prosody deltas.
Outputs drift_vector.json and drift_log.json.
"""

import json
import numpy as np
from scipy.signal import savgol_filter
import portalocker
import os

def run(context):
    config = context['config']['drift']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)
        
        f0_z = np.array(prosody['frame_series']['f0_z'])
        energy_z = np.array(prosody['frame_series']['energy_z'])
        
        delta_f0 = np.diff(f0_z)
        delta_energy = np.diff(energy_z)
        
        thresh_f0 = config['thresh_pitch'] * np.std(delta_f0)
        thresh_energy = config['thresh_energy'] * np.std(delta_energy)
        
        drift_points = np.where((np.abs(delta_f0) > thresh_f0) | (np.abs(delta_energy) > thresh_energy))[0]
        
        # Buffer and merge (stub: simple)
        # Whiplash filter: ignore small reversals
        # 2nd-order smoothing
        deltas = np.concatenate([delta_f0, delta_energy])
        smoothed_deltas = savgol_filter(deltas, window_length=config['smoothing_window'], polyorder=config['smoothing_order'])
        
        drift_vector = {'deltas': smoothed_deltas.tolist(), 'slices': drift_points.tolist()}
        drift_log = {'thresholds': {'f0': thresh_f0, 'energy': thresh_energy}, 'num_drifts': len(drift_points)}
        
        vector_path = os.path.join(speaker_out, 'drift_vector.json')
        log_path = os.path.join(speaker_out, 'drift_log.json')
        with open(vector_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(drift_vector, f)
            portalocker.unlock(f)
        with open(log_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(drift_log, f)
            portalocker.unlock(f)
    
    return {'drift_vector': vector_path, 'drift_log': log_path}