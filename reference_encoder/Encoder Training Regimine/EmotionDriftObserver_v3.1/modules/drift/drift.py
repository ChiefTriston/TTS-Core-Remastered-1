# modules/drift/drift.py (updated with buffer-zone merge, whiplash filter, refined smoothing)
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
        
        time = np.array(prosody['frame_series']['time'])
        f0_z = np.array(prosody['frame_series']['f0_z'])
        energy_z = np.array(prosody['frame_series']['energy_z'])
        
        delta_f0 = np.diff(f0_z)
        delta_energy = np.diff(energy_z)
        
        thresh_f0 = config['thresh_pitch'] * np.std(delta_f0)
        thresh_energy = config['thresh_energy'] * np.std(delta_energy)
        
        # Detect initial drift points
        drift_points_f0 = np.where(np.abs(delta_f0) > thresh_f0)[0]
        drift_points_energy = np.where(np.abs(delta_energy) > thresh_energy)[0]
        drift_points = np.unique(np.concatenate([drift_points_f0, drift_points_energy]))
        
        # Buffer-zone merge: merge points within buffer_zone seconds
        time_delta = time[1] - time[0] if len(time) > 1 else 1.0
        buffer_samples = int(config['buffer_zone'] / time_delta)
        merged_drifts = []
        if len(drift_points) > 0:
            current = drift_points[0]
            for point in drift_points[1:]:
                if point - current <= buffer_samples:
                    # Merge by taking average or keep first; here keep first for simplicity
                    pass
                else:
                    merged_drifts.append(current)
                    current = point
            merged_drifts.append(current)
        
        # Whiplash filter: ignore tiny polarity flips (rapid sign changes)
        deltas_f0 = delta_f0[merged_drifts] if len(merged_drifts) > 0 else []
        deltas_energy = delta_energy[merged_drifts] if len(merged_drifts) > 0 else []
        # Example: if consecutive deltas have opposite signs and small magnitude, remove
        filtered_drifts = []
        for i in range(len(merged_drifts)):
            if i > 0 and np.sign(deltas_f0[i]) != np.sign(deltas_f0[i-1]) and abs(deltas_f0[i]) < thresh_f0 * 0.5:
                continue
            if i > 0 and np.sign(deltas_energy[i]) != np.sign(deltas_energy[i-1]) and abs(deltas_energy[i]) < thresh_energy * 0.5:
                continue
            filtered_drifts.append(merged_drifts[i])
        
        # 2nd-order smoothing on deltas
        deltas = np.concatenate([delta_f0, delta_energy])
        smoothed_deltas = savgol_filter(deltas, window_length=config['smoothing_window'], polyorder=config['smoothing_order'], mode='nearest')
        
        drift_vector = {'deltas': smoothed_deltas.tolist(), 'slices': filtered_drifts}
        drift_log = {'thresholds': {'f0': thresh_f0, 'energy': thresh_energy}, 'num_drifts': len(filtered_drifts)}
        
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