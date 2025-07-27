# modules/transcription/transcription.py
"""
Transcription of slices using Whisper + VAD cleanup.
Outputs transcript.json.
"""

import json
import whisperx
import torch
import portalocker
import os
import torchaudio

def run(context):
    config = context['config']['transcription']
    global_config = context['config']['global']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    device = 'cuda' if global_config['use_gpu'] and torch.cuda.is_available() else 'cpu'
    vad_thresh = config['vad_thresh']
    
    model = whisperx.load_model(config['model'], device)
    
    sr = global_config['sample_rate']
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        wav_path = os.path.join(speaker_out, f'{speaker_id}.wav')
        waveform, _ = torchaudio.load(wav_path)
        
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)
        
        with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)
        
        frame_delta = prosody['frame_series']['time'][1] - prosody['frame_series']['time'][0] if len(prosody['frame_series']['time']) > 1 else 0.02
        slices = sorted(drift['slices'])
        slice_starts = [0] + slices
        slice_ends = slices + [len(prosody['frame_series']['time'])]
        
        cleaned_segments = []
        for i in range(len(slice_starts)):
            start_frame = slice_starts[i]
            end_frame = slice_ends[i]
            start_time = start_frame * frame_delta
            end_time = end_frame * frame_delta
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            slice_wave = waveform[:, start_sample:end_sample]
            
            # Save temp slice wav
            temp_path = os.path.join(speaker_out, f'temp_slice_{i}.wav')
            torchaudio.save(temp_path, slice_wave, sr)
            
            audio = whisperx.load_audio(temp_path)
            result = model.transcribe(audio)
            
            for seg in result['segments']:
                if seg.get('score', 1.0) > vad_thresh:
                    seg['start'] += start_time
                    seg['end'] += start_time
                    cleaned_segments.append(seg)
            
            os.remove(temp_path)
        
        transcript = {'slices': cleaned_segments}
        
        json_path = os.path.join(speaker_out, 'transcript.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(transcript, f)
            portalocker.unlock(f)
    
    return {'transcript': json_path}