# modules/diarization/diarization.py
"""
Speaker diarization using WhisperX.
Outputs speaker_{id}.wav files and speaker_mapping.json.
"""

import json
import os
import whisperx
import torch
from pydub import AudioSegment
import portalocker
import time

def run(context):
    cfg = context['config']['diarization']
    gcfg = context['config']['global']
    device = "cuda" if gcfg['use_gpu'] and torch.cuda.is_available() else "cpu"

    # 1. load & transcribe+align
    model = whisperx.load_model(cfg['model'], device, compute_type=cfg['compute_type'])
    audio = whisperx.load_audio(context['input_wav'])
    result = model.transcribe(audio, batch_size=cfg['batch_size'])

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    # 2. diarize
    diarizer = whisperx.DiarizationPipeline(
        use_auth_token=gcfg['hf_token'], device=device
    )
    diarize_segments = diarizer(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 3. group & stitch
    speaker_segs = {}
    mapping = {}
    full = AudioSegment.from_wav(context['input_wav'])
    for seg in result['segments']:
        sp = seg['speaker']
        speaker_segs.setdefault(sp, []).append(seg)
        mapping.setdefault(sp, {'timestamps': [], 'confidences': []})
        mapping[sp]['timestamps'].append((seg['start'], seg['end']))
        mapping[sp]['confidences'].append(seg.get('avg_logprob', 0.0))

    out_base = os.path.join(context['output_dir'], 'emotion_tags')
    for sp, segs in speaker_segs.items():
        odir = os.path.join(out_base, sp)
        os.makedirs(odir, exist_ok=True)
        out_wav = os.path.join(odir, f"{sp}.wav")
        combined = AudioSegment.empty()
        for s in segs:
            combined += full[s['start']*1000 : s['end']*1000]
        combined.export(out_wav, format='wav')

    # 4. save mapping
    mapping_path = os.path.join(out_base, 'speaker_mapping.json')
    with open(mapping_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(mapping, f)
        portalocker.unlock(f)

    return {'speaker_mapping': 'speaker_mapping.json'}