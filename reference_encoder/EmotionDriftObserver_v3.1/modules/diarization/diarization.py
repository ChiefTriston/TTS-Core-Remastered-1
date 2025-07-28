# modules/diarization/diarization.py
"""
Speaker diarization using WhisperX (with Pyannote fallback).
Outputs per‑speaker WAVs under emotion_tags/{speaker}/ and a speaker_mapping.json.
"""

import json
import os
import time
import torch
import whisperx
import portalocker
from pydub import AudioSegment
import pyannote.audio

def merge_overlaps(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for curr in intervals[1:]:
        last = merged[-1]
        if curr[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], curr[1]))
        else:
            merged.append(curr)
    return merged

def run(context):
    cfg   = context['config']['diarization']
    gcfg  = context['config']['global']
    inp   = context['input_wav']
    out_b = os.path.join(context['output_dir'], 'emotion_tags')
    os.makedirs(out_b, exist_ok=True)
    device = "cuda" if gcfg['use_gpu'] and torch.cuda.is_available() else "cpu"

    # 1) Transcribe + align
    model       = whisperx.load_model(cfg['model'], device, compute_type=cfg['compute_type'])
    audio_array = whisperx.load_audio(inp)
    result      = model.transcribe(audio_array, batch_size=cfg['batch_size'])

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio_array, device
    )

    # 2) Diarize (WhisperX → fallback Pyannote)
    try:
        diarizer = whisperx.DiarizationPipeline(
            use_auth_token=gcfg.get('hf_token',''), device=device
        )
        diarize_segs = diarizer(audio_array)
    except Exception as e:
        print(f"[warn] WhisperX diarizer error: {e}; falling back to pyannote")
        pn_pipeline = pyannote.audio.Pipeline.from_pretrained(
            "pyannote/speaker-diarization", use_auth_token=gcfg.get('hf_token','')
        )
        diarize_segs = pn_pipeline({ "waveform": torch.tensor(audio_array).unsqueeze(0),
                                      "sample_rate": metadata["sample_rate"] })

    result = whisperx.assign_word_speakers(diarize_segs, result)

    # 3) Group segments per speaker & stitch into WAVs
    speaker_segs = {}
    mapping      = {}
    full_audio   = AudioSegment.from_wav(inp)

    for seg in result["segments"]:
        sp = seg["speaker"]
        speaker_segs.setdefault(sp, []).append(seg)
        mapping.setdefault(sp, {"timestamps": [], "confidences": []})
        mapping[sp]["timestamps"].append((seg["start"], seg["end"]))
        # use confidence if available, otherwise avg_logprob
        mapping[sp]["confidences"].append(
            seg.get("confidence", seg.get("avg_logprob", 0.0))
        )

    # a) merge overlapping time‑ranges in mapping
    for sp in mapping:
        mapping[sp]["timestamps"] = merge_overlaps(mapping[sp]["timestamps"])

    # b) for each speaker, concatenate all their segments
    for sp, segs in speaker_segs.items():
        speaker_dir = os.path.join(out_b, sp)
        os.makedirs(speaker_dir, exist_ok=True)
        out_wav = os.path.join(speaker_dir, f"{sp}.wav")

        combined = AudioSegment.empty()
        for s in segs:
            start_ms = int(s["start"] * 1000)
            end_ms   = int(s["end"]   * 1000)
            combined += full_audio[start_ms:end_ms]

        combined.export(out_wav, format="wav")

    # 4) write mapping JSON
    mapping_path = os.path.join(out_b, "speaker_mapping.json")
    with open(mapping_path, "w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(mapping, f, indent=2)
        portalocker.unlock(f)

    return { "speaker_mapping": mapping_path }
