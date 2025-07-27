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
import webrtcvad
import numpy as np

def run(context):
    gcfg = context['config']['global']
    cfg = context['config']['transcription']
    device = "cuda" if gcfg['use_gpu'] and torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(cfg['model'], device)

    sr = gcfg['sample_rate']
    vad = webrtcvad.Vad(mode=3)  # Aggressive VAD

    for sp in context['speaker_ids']:
        wav = os.path.join(context['output_dir'],'emotion_tags',sp,f"{sp}.wav")
        audio = whisperx.load_audio(wav)
        
        speaker_out = os.path.join(context['output_dir'],'emotion_tags',sp)
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r') as f:
            drift = json.load(f)
        boundaries = drift['boundaries']
        
        cleaned = []
        for bound in boundaries:
            start, end = bound
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            slice_audio = audio[start_sample:end_sample]
            result = model.transcribe(slice_audio)
            
            for seg in result['segments']:
                seg['start'] += start
                seg['end'] += start
                
                # VAD cleanup
                seg_start_sample = int(seg['start'] * sr)
                seg_end_sample = int(seg['end'] * sr)
                slice_wave = audio[seg_start_sample:seg_end_sample]
                frame_duration_ms = 30
                frame_size = int(sr * frame_duration_ms / 1000)
                num_frames = len(slice_wave) // frame_size
                voiced_frames = 0
                for i in range(num_frames):
                    frame = slice_wave[i*frame_size:(i+1)*frame_size]
                    frame_int16 = (frame * 32767).astype(np.int16)
                    if vad.is_speech(frame_int16.tobytes(), sr):
                        voiced_frames += 1
                vad_score = voiced_frames / num_frames if num_frames > 0 else 0.0
                if vad_score > cfg['vad_thresh'] or seg.get('avg_logprob', -10) > -1.0:  # Combined
                    cleaned.append({
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'],
                        'score': vad_score
                    })

        out = {'slices': cleaned}
        fp = os.path.join(context['output_dir'],'emotion_tags',sp,'transcript.json')
        with open(fp,'w') as f:
            json.dump(out,f)
    return {'transcript': 'transcript.json'}