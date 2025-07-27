# modules/transcription/transcription.py (updated with config lookup)
"""
Transcription of slices using Whisper + VAD cleanup.
Outputs transcript.json.
"""

import json
import whisperx
import torch
import portalocker
import os

def run(context):
    config = context['config']['transcription']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    device = 'cuda' if context['config']['global']['use_gpu'] and torch.cuda.is_available() else 'cpu'
    vad_thresh = config['vad_thresh']
    
    model = whisperx.load_model(config['model'], device)
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        wav_path = os.path.join(speaker_out, f'{speaker_id}.wav')
        audio = whisperx.load_audio(wav_path)
        
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)
        
        # Assume slices from drift_points, but stub: transcribe whole, then slice
        result = model.transcribe(audio)
        
        # VAD cleanup: filter low conf
        cleaned_segments = [seg for seg in result['segments'] if seg.get('score', 1.0) > vad_thresh]
        
        transcript = {'slices': cleaned_segments}
        
        json_path = os.path.join(speaker_out, 'transcript.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(transcript, f)
            portalocker.unlock(f)
    
    return {'transcript': json_path}