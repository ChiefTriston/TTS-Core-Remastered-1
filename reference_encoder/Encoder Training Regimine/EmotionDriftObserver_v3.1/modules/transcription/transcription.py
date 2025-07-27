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
        result = model.transcribe(audio)

        # VAD cleanup using webrtcvad or token score
        cleaned = []
        for seg in result['segments']:
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            slice_wave = audio[start_sample:end_sample]
            frame_duration_ms = 30
            frame_size = int(sr * frame_duration_ms / 1000)
            num_frames = len(slice_wave) // frame_size
            voiced_frames = 0
            for i in range(num_frames):
                frame = slice_wave[i*frame_size:(i+1)*frame_size]
                if vad.is_speech(frame.tobytes(), sr):
                    voiced_frames += 1
            vad_score = voiced_frames / num_frames if num_frames > 0 else 0.0
            if vad_score > cfg['vad_thresh'] or seg.get('avg_logprob', -10) > -1.0:  # Combined
                cleaned.append(seg)

        out = {'slices': cleaned}
        fp = os.path.join(context['output_dir'],'emotion_tags',sp,'transcript.json')
        with open(fp,'w') as f:
            json.dump(out,f)
    return {'transcript': 'transcript.json'}