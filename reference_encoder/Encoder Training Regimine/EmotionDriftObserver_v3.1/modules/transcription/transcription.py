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
    silence_thresh = 0.5  # % silence to flag
    
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
            duration = end_time - start_time
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            slice_wave = waveform[:, start_sample:end_sample].to(device)
            
            # VAD-based silence trimming
            vad_model = whisperx.vad.load_vad_model(device)
            vad_options = {"threshold": vad_thresh}
            vad_segments = whisperx.vad.segment(slice_wave.numpy().squeeze(), vad_model, vad_options)
            
            speech_dur = sum(s['end'] - s['start'] for s in vad_segments)
            silence_percent = 1 - speech_dur / duration if duration > 0 else 1.0
            
            # Trim and transcribe speech segments
            text = ''
            for vad_seg in vad_segments:
                vad_start = int(vad_seg['start'] * sr)
                vad_end = int(vad_seg['end'] * sr)
                vad_wave = slice_wave[:, vad_start:vad_end]
                result = model.transcribe(vad_wave.numpy().squeeze())
                for seg in result['segments']:
                    if seg.get('score', 1.0) > vad_thresh:
                        text += seg['text'] + ' '
            
            seg = {'text': text.strip(), 'start': start_time, 'end': end_time, 'score': 1.0 - silence_percent, 'silence_percent': silence_percent}
            if silence_percent > silence_thresh:
                seg['hallucination_flag'] = True  # Or high_silence_flag
            
            cleaned_segments.append(seg)
        
        transcript = {'slices': cleaned_segments}
        
        json_path = os.path.join(speaker_out, 'transcript.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(transcript, f)
            portalocker.unlock(f)
    
    return {'transcript': json_path}