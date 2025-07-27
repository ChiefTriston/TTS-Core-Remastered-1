# modules/prosody/prosody.py (unchanged)
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
        
        f0_z = (f0 - np.mean(f0)) / np.std(f0) if np.std(f0) > 0 else f0
        energy_z = (energy - np.mean(energy)) / np.std(energy) if np.std(energy) > 0 else energy
        pitch_var_z = (pitch_var - np.mean(pitch_var)) / np.std(pitch_var) if np.std(pitch_var) > 0 else pitch_var
        
        trend_f0 = np.polyfit(time, f0_z, 1).tolist()
        trend_energy = np.polyfit(time, energy_z, 1).tolist()
        trend_pitch_var = np.polyfit(time, pitch_var_z, 1).tolist()
        
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
                'mfcc': mfcc.tolist()
            },
            'trendlines': {
                'f0': trend_f0,
                'energy': trend_energy,
                'pitch_var': trend_pitch_var
            }
        }
        
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        json_path = os.path.join(speaker_out, 'prosody_trend.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(prosody_data, f)
            portalocker.unlock(f)
    
    return {'prosody_trend': json_path}  # Last one, or collect all