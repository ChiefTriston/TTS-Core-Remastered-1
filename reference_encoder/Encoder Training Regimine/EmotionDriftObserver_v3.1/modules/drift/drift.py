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
import pandas as pd
from drift_utils import generate_segment_plot_map

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
        
        # Dynamic threshold adaptivity (rolling std)
        window = 50  # Rolling window size
        rolling_std_f0 = pd.Series(delta_f0).rolling(window=window, min_periods=1).std().values
        rolling_std_energy = pd.Series(delta_energy).rolling(window=window, min_periods=1).std().values
        thresh_f0 = config['thresh_pitch'] * rolling_std_f0
        thresh_energy = config['thresh_energy'] * rolling_std_energy
        
        drift_points = np.where((np.abs(delta_f0) > thresh_f0) | (np.abs(delta_energy) > thresh_energy))[0]
        
        # Buffer-zone merging into contiguous slices
        frame_delta = time[1] - time[0] if len(time) > 1 else 0.02
        buffer_samples = int(config['buffer_zone'] / frame_delta)
        merged_drifts = []
        if len(drift_points) > 0:
            current_start = drift_points[0]
            for point in drift_points[1:]:
                if point - current_start <= buffer_samples:
                    continue  # Merge
                else:
                    merged_drifts.append(current_start)
                    current_start = point
            merged_drifts.append(current_start)
        
        # Polarity merging (group positive vs negative)
        deltas_combined = (delta_f0 + delta_energy) / 2
        polarity = np.sign(deltas_combined[merged_drifts])
        polarity_merged = []
        current_polarity = polarity[0]
        current_group = [merged_drifts[0]]
        for i in range(1, len(merged_drifts)):
            if polarity[i] == current_polarity:
                current_group.append(merged_drifts[i])
            else:
                polarity_merged.append(np.mean(current_group))  # Average for group representative
                current_polarity = polarity[i]
                current_group = [merged_drifts[i]]
        polarity_merged.append(np.mean(current_group))
        
        # Whiplash filter (drop tiny oscillations)
        small_thresh = min(thresh_f0.mean(), thresh_energy.mean()) * 0.5
        filtered_drifts = [polarity_merged[0]]
        for i in range(1, len(polarity_merged)):
            if abs(deltas_combined[int(polarity_merged[i])] - deltas_combined[int(filtered_drifts[-1])]) < small_thresh:
                continue  # Drop tiny
            filtered_drifts.append(polarity_merged[i])
        
        # 2nd-order smoothing
        deltas = np.concatenate([delta_f0, delta_energy])
        smoothed_deltas = savgol_filter(deltas, window_length=config['smoothing_window'], polyorder=config['smoothing_order'])
        
        # Slice boundaries in time
        slice_starts = [0] + filtered_drifts
        slice_ends = filtered_drifts + [len(time) - 1]
        slice_boundaries = [(time[int(start)], time[int(end)]) for start, end in zip(slice_starts, slice_ends)]
        
        drift_vector = {'deltas': smoothed_deltas.tolist(), 'slices': filtered_drifts, 'slice_boundaries': slice_boundaries}
        drift_log = {'thresholds': {'f0': thresh_f0.mean(), 'energy': thresh_energy.mean()}, 'num_drifts': len(filtered_drifts)}
        
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
        
        # Generate a plot‐map for each drift slice:
        # You need to keep around your full_scores list of dicts:
        full_scores = [{"time": t, "vader_compound": d} for t, d in zip(time[1:], smoothed_deltas)]
        for idx, slice_start in enumerate(drift_vector['slices']):
            slice_end = drift_vector['slices'][idx + 1] if idx+1 < len(drift_vector['slices']) else time[-1]
            img_path = generate_segment_plot_map(
                full_scores=full_scores,
                segment_start=slice_start,
                segment_end=slice_end,
                clip_id=f"{speaker_id}_{idx}",
                tier1_transition="",     # or compute if you have transition info
                drift_reason="",         # optional reason
                save_dir=plot_dir
            )
            # you could log img_path or embed in drift_log if you like
    
    return {'drift_vector': vector_path, 'drift_log': log_path}