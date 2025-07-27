# modules/drift/drift.py
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
        
        drift_points = np.where((np.abs(delta_f0) > thresh_f0) | (np.abs(delta_energy) > thresh_energy))[0]
        
        # Buffer-zone merging
        frame_delta = time[1] - time[0] if len(time) > 1 else 0.02
        buffer_samples = int(config['buffer_zone'] / frame_delta)
        merged_drifts = []
        if len(drift_points) > 0:
            current_start = drift_points[0]
            for point in drift_points[1:]:
                if point - current_start <= buffer_samples:
                    continue  # Merge by keeping first
                else:
                    merged_drifts.append(current_start)
                    current_start = point
            merged_drifts.append(current_start)
        
        # Polarity-run collapse: merge same-polarity consecutive
        deltas_combined = (delta_f0 + delta_energy) / 2  # Average for polarity
        polarity = np.sign(deltas_combined[merged_drifts])
        keep = np.concatenate([[True], np.diff(polarity) != 0])
        polarity_merged = np.array(merged_drifts)[keep]
        
        # Whiplash filter: ignore tiny reversals
        small_thresh = thresh_f0 * 0.5  # Arbitrary small
        filtered_drifts = [polarity_merged[0]]
        for i in range(1, len(polarity_merged)):
            if np.sign(deltas_combined[polarity_merged[i]]) != np.sign(deltas_combined[filtered_drifts[-1]]) and abs(deltas_combined[polarity_merged[i]]) < small_thresh:
                continue
            filtered_drifts.append(polarity_merged[i])
        
        # 2nd-order smoothing
        deltas = np.concatenate([delta_f0, delta_energy])
        smoothed_deltas = savgol_filter(deltas, window_length=config['smoothing_window'], polyorder=config['smoothing_order'])
        
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
    return {'drift_vector': vector_path, 'drift_log': log_path}