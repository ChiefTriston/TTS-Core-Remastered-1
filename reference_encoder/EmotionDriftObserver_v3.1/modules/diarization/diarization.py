# modules/diarization/diarization.py
"""
Speaker diarization using WhisperX (with Pyannote fallback),
streaming in 10‑minute chunks to avoid OOM on huge files.
Suppresses version‑mismatch banner warnings.
Outputs per‑speaker WAVs under emotion_tags/{speaker}/ and a speaker_mapping.json.
"""

import json
import os
import math
import torch
import portalocker
from pydub import AudioSegment
import pyannote.audio
import torchaudio
import whisperx

import contextlib
import io

CHUNK_SECONDS = 600  # 10 minutes

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

def _quiet_load_model(model_name, device, compute_type):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return whisperx.load_model(model_name, device, compute_type=compute_type)

def _quiet_load_align_model(language_code, device):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return whisperx.load_align_model(language_code=language_code, device=device)

def _quiet_diarizer(audio_array, use_auth_token, device):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return whisperx.DiarizationPipeline(use_auth_token=use_auth_token, device=device)(audio_array)

def run(context):
    cfg   = context['config']['diarization']
    gcfg  = context['config']['global']
    inp   = context['input_wav']
    out_b = os.path.join(context['output_dir'], 'emotion_tags')
    os.makedirs(out_b, exist_ok=True)
    device = "cuda" if gcfg['use_gpu'] and torch.cuda.is_available() else "cpu"

    # get file metadata
    info = torchaudio.info(inp)
    sr   = info.sample_rate
    total_seconds = info.num_frames / sr
    num_chunks    = math.ceil(total_seconds / CHUNK_SECONDS)

    # accumulate segments across chunks
    all_segments = []
    align_model = None
    metadata   = None

    for chunk_idx in range(num_chunks):
        # compute chunk offset in seconds
        start_sec = chunk_idx * CHUNK_SECONDS
        length_sec = min(CHUNK_SECONDS, total_seconds - start_sec)
        frame_offset = int(start_sec * sr)
        num_frames   = int(length_sec * sr)
        waveform, _  = torchaudio.load(inp, frame_offset=frame_offset, num_frames=num_frames)
        audio_array  = waveform.mean(dim=0).numpy()  # to mono numpy

        # 1) transcribe + align
        model = _quiet_load_model(cfg['model'], device, cfg['compute_type'])
        result = model.transcribe(audio_array, batch_size=cfg['batch_size'])

        if align_model is None:
            align_model, metadata = _quiet_load_align_model(
                language_code=result["language"], device=device
            )
        result = whisperx.align(result["segments"], align_model, metadata, audio_array, device)

        # 2) diarize (WhisperX → fallback Pyannote)
        try:
            diarize_segs = _quiet_diarizer(audio_array, gcfg.get('hf_token',''), device)
        except Exception:
            # fallback to pyannote
            pn = pyannote.audio.Pipeline.from_pretrained(
                "pyannote/speaker-diarization", use_auth_token=gcfg.get('hf_token','')
            )
            diarize_segs = pn({"waveform": torch.from_numpy(audio_array).unsqueeze(0),
                               "sample_rate": metadata["sample_rate"]})

        result = whisperx.assign_word_speakers(diarize_segs, result)

        # shift chunk timestamps back to full-file time
        for seg in result["segments"]:
            seg["start"] += start_sec
            seg["end"]   += start_sec
        all_segments.extend(result["segments"])

    # now group & stitch
    speaker_segs = {}
    mapping      = {}
    full_audio   = AudioSegment.from_wav(inp)

    for seg in all_segments:
        sp = seg["speaker"]
        speaker_segs.setdefault(sp, []).append(seg)
        mapping.setdefault(sp, {"timestamps": [], "confidences": []})
        mapping[sp]["timestamps"].append((seg["start"], seg["end"]))
        mapping[sp]["confidences"].append(
            seg.get("confidence", seg.get("avg_logprob", 0.0))
        )

    # merge and export per speaker
    for sp in mapping:
        mapping[sp]["timestamps"] = merge_overlaps(mapping[sp]["timestamps"])

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

    # write mapping
    mapping_path = os.path.join(out_b, "speaker_mapping.json")
    with open(mapping_path, "w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(mapping, f, indent=2)
        portalocker.unlock(f)

    return {"speaker_mapping": mapping_path}

