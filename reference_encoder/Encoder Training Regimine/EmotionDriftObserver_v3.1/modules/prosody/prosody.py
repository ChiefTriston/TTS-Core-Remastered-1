# modules/prosody/prosody.py
"""
Prosody extraction using prosody3 at 50Hz.
Outputs prosody_trend.json with frame series and trendlines.
"""

import sys
import json
import numpy as np
import torch
import torchaudio
import portalocker
import os
from scipy.signal import savgol_filter

sys.path.insert(0, r'C:\Users\trist\OneDrive\Documents\Remastered TTS Final Version\TTS Core Remastered\prosody3')

from prosody_predictor import ProsodyPredictorV15

def run(context):
    config = context['config']['prosody']
    global_config = context['config']['global']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    device = 'cuda' if global_config['use_gpu'] and torch.cuda.is_available() else 'cpu'
    sr = global_config['sample_rate']
    freq = config['extract_freq']
    hop_length = int(sr / freq)
    
    model = ProsodyPredictorV15(hop_length=hop_length, sample_rate=sr, **config).to(device)
    model.eval()
    
    for speaker_id in speaker_ids:
        wav_path = os.path.join(output_dir, 'emotion_tags', speaker_id, f'{speaker_id}.wav')
        waveform, _ = torchaudio.load(wav_path)
        mel = model.mel_spec(waveform.to(device))
        prosody = model(mel)
        
        f0 = prosody['f0'].squeeze(0).cpu().numpy()
        energy = prosody['energy'].squeeze(0).cpu().numpy()
        pitch_var = prosody['pitch_var'].squeeze(0).cpu().numpy()
        speech_rate = prosody['speech_rate'].item()
        pause_dur = prosody['pause_dur'].item()
        mfcc = prosody['mfcc'].squeeze(0).cpu().numpy()
        
        T = len(f0)
        time = np.arange(0, T * (hop_length / sr), hop_length / sr)
        duration = time[-1] if T > 0 else 0
        
        # Smoothing raw series
        window = 5  # Example window
        f0_smoothed = savgol_filter(f0, window_length=window, polyorder=2)
        energy_smoothed = savgol_filter(energy, window_length=window, polyorder=2)
        pitch_var_smoothed = savgol_filter(pitch_var, window_length=window, polyorder=2)
        
        # Z-norm smoothed
        f0_z = (f0_smoothed - np.mean(f0_smoothed)) / np.std(f0_smoothed) if np.std(f0_smoothed) > 0 else f0_smoothed
        energy_z = (energy_smoothed - np.mean(energy_smoothed)) / np.std(energy_smoothed) if np.std(energy_smoothed) > 0 else energy_smoothed
        pitch_var_z = (pitch_var_smoothed - np.mean(pitch_var_smoothed)) / np.std(pitch_var_smoothed) if np.std(pitch_var_smoothed) > 0 else pitch_var_smoothed
        
        # Trendlines on z-norm
        trend_f0 = np.polyfit(time, f0_z, 1).tolist()
        trend_energy = np.polyfit(time, energy_z, 1).tolist()
        trend_pitch_var = np.polyfit(time, pitch_var_z, 1).tolist()
        
        # Global summaries
        summaries = {
            'f0': {'min': np.min(f0_smoothed), 'max': np.max(f0_smoothed), 'mean': np.mean(f0_smoothed), 'std': np.std(f0_smoothed)},
            'energy': {'min': np.min(energy_smoothed), 'max': np.max(energy_smoothed), 'mean': np.mean(energy_smoothed), 'std': np.std(energy_smoothed)},
            'pitch_var': {'min': np.min(pitch_var_smoothed), 'max': np.max(pitch_var_smoothed), 'mean': np.mean(pitch_var_smoothed), 'std': np.std(pitch_var_smoothed)}
        }
        
        # Pause-ratio (example: pause_dur is average, ratio = total_pause / duration ~ pause_dur / avg_segment_len, but stub as pause_dur / duration)
        pause_ratio = pause_dur / duration if duration > 0 else 0
        
        prosody_data = {
            'frame_series': {
                'time': time.tolist(),
                'f0_z': f0_z.tolist(),
                'energy_z': energy_z.tolist(),
                'pitch_var_z': pitch_var_z.tolist()
            },
            'globals': {
                'speech_rate': speech_rate,
                'pause_dur': pause_dur,
                'pause_ratio': pause_ratio,
                'mfcc': mfcc.tolist()
            },
            'trendlines': {
                'f0': trend_f0,
                'energy': trend_energy,
                'pitch_var': trend_pitch_var
            },
            'summaries': summaries
        }
        
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        json_path = os.path.join(speaker_out, 'prosody_trend.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(prosody_data, f)
            portalocker.unlock(f)
    
    return {'prosody_trend': json_path}  # Last one, or collect all