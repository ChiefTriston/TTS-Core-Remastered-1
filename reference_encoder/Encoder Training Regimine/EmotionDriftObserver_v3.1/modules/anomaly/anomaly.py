# modules/anomaly/anomaly.py
"""
Anomaly flagging for hallucinations and VADER outliers.
Injects into drift_vector.json, updates drift_log.json.
"""

import json
import numpy as np
import portalocker
import os
from scipy.stats import entropy as shannon_entropy

def run(context):
    config = context['config']['anomaly']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'tier1_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier1 = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier2 = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)
        
        time = np.array(prosody['frame_series']['time'])
        time_delta = time[1] - time[0] if len(time) > 1 else 0.02
        energy_z = np.array(prosody['frame_series']['energy_z'])
        
        anomalies = []
        for idx, slice_data in enumerate(transcript['slices']):
            text = slice_data['text']
            start_time = slice_data['start']
            end_time = slice_data['end']
            start_idx = max(0, min(len(energy_z) - 1, int(start_time / time_delta)))
            end_idx = max(start_idx + 1, min(len(energy_z), int(end_time / time_delta)))
            
            # Whisper hallucination: short, repetitive, or long silence with words
            if len(text) < config['hallucination_min_len']:
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx, 'reason': 'short_text'})
            if len(text) > 0 and max(Counter(text).values()) / len(text) > config['repetition_thresh']:
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx, 'reason': 'repetitive'})
            slice_energy = energy_z[start_idx:end_idx]
            silence_frames = np.sum(slice_energy < -1.5) / len(slice_energy) if len(slice_energy) > 0 else 0.0  # Low energy thresh
            if len(text) > 0 and silence_frames > 0.7:
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx, 'reason': 'silent_with_words'})
            
            # VADER anomaly: sentiment swings in sliding window
            window_size = 3
            compounds = [t['compound'] for t in tier1]
            if idx >= window_size - 1:
                window = compounds[idx - window_size + 1:idx + 1]
                swing = max(window) - min(window)
                sigma = np.std(compounds)
                if swing > config['outlier_std_mult'] * sigma:
                    anomalies.append({'type': 'vader_anomaly', 'slice': idx, 'reason': 'swing'})
            elif abs(compounds[idx]) > np.mean(compounds) + config['outlier_std_mult'] * np.std(compounds):
                anomalies.append({'type': 'vader_anomaly', 'slice': idx, 'reason': 'outlier'})
        
        # Inject to drift_vector
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            drift = json.load(f)
            f.seek(0)
            drift['anomalies'] = anomalies
            json.dump(drift, f)
            f.truncate()
            portalocker.unlock(f)
        
        # Compute emotion_entropy (Shannon across tier2 distribution)
        labels = [t['label'] for t in tier2]
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels) if len(labels) > 0 else []
        emotion_entropy = shannon_entropy(probs) if len(probs) > 0 else 0.0
        
        # Confidence drift slope (linear regression on confidence vs. time)
        confidences = [t['confidence'] for t in tier2]
        times = [ (transcript['slices'][i]['start'] + transcript['slices'][i]['end']) / 2 for i in range(len(transcript['slices'])) ]  # Midpoint time
        if len(confidences) > 1:
            slope = np.polyfit(times, confidences, 1)[0]
        else:
            slope = 0.0
        
        with open(os.path.join(speaker_out, 'drift_log.json'), 'r+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            log = json.load(f)
            f.seek(0)
            log['emotion_entropy'] = emotion_entropy
            log['confidence_drift_slope'] = slope
            json.dump(log, f)
            f.truncate()
            portalocker.unlock(f)
    
    return {'updated_drift': True}