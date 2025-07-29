"""
Speaker diarization using WhisperX (with Pyannote fallback), streaming in 10‑minute chunks to avoid OOM on huge files.
Suppresses version‑mismatch banner warnings.
Outputs per‑speaker WAVs under emotion_tags/{speaker}/ and a speaker_mapping.json.
"""
import json
import os
import math
import torch
import portalocker
from pydub import AudioSegment
import whisperx
from whisperx.diarization import DiarizationPipeline
import pyannote.audio
import torchaudio
import contextlib
import io

# Load chunk size from config or default
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


def _quiet_diarizer(pipeline, audio_array):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return pipeline(audio_array)


def run(context):
    cfg = context['config']['diarization']
    gcfg = context['config']['global']
    inp = context['input_wav']
    out_b = os.path.join(context['output_dir'], 'emotion_tags')
    os.makedirs(out_b, exist_ok=True)
    device = "cuda" if gcfg.get('use_gpu', False) and torch.cuda.is_available() else "cpu"

    # Load models from context
    whisper_model = context['models']['whisper']
    align_model, metadata = context['models']['align']

    # Initialize WhisperX DiarizationPipeline
    diarizer_pipeline = DiarizationPipeline(
        use_auth_token=gcfg.get('hf_token', ''),
        device=device
    )

    # get file metadata
    info = torchaudio.info(inp)
    sr = info.sample_rate
    total_seconds = info.num_frames / sr
    num_chunks = math.ceil(total_seconds / CHUNK_SECONDS)

    all_segments = []
    full_audio = AudioSegment.from_wav(inp)

    for chunk_idx in range(num_chunks):
        start_sec = chunk_idx * CHUNK_SECONDS
        length_sec = min(CHUNK_SECONDS, total_seconds - start_sec)
        frame_offset = int(start_sec * sr)
        num_frames = int(length_sec * sr)

        waveform, _ = torchaudio.load(inp, frame_offset=frame_offset, num_frames=num_frames)
        audio_array = waveform.mean(dim=0).numpy()

        # Transcribe and align once per chunk under no_grad
        with torch.no_grad():
            result = whisper_model.transcribe(audio_array, batch_size=cfg.get('batch_size', 16))
            result = whisperx.align(result['segments'], align_model, metadata, audio_array, device)

        # Diarize using WhisperX pipeline
        try:
            diarize_segs = _quiet_diarizer(diarizer_pipeline, audio_array)
        except Exception:
            pn_pipeline = pyannote.audio.Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=gcfg.get('hf_token', '')
            )
            diarize_segs = pn_pipeline({
                "waveform": torch.from_numpy(audio_array).unsqueeze(0),
                "sample_rate": metadata['sample_rate']
            })

        result = whisperx.assign_word_speakers(diarize_segs, result)

        # shift chunk timestamps
        for seg in result['segments']:
            seg['start'] += start_sec
            seg['end'] += start_sec
            all_segments.append(seg)

    # group & stitch
    speaker_segs = {}
    mapping = {}
    for seg in all_segments:
        sp = seg['speaker']
        speaker_segs.setdefault(sp, []).append(seg)
        mapping.setdefault(sp, {'timestamps': [], 'confidences': []})
        mapping[sp]['timestamps'].append((seg['start'], seg['end']))
        mapping[sp]['confidences'].append(seg.get('confidence', seg.get('avg_logprob', 0.0)))

    for sp in mapping:
        mapping[sp]['timestamps'] = merge_overlaps(mapping[sp]['timestamps'])

    for sp, segs in speaker_segs.items():
        speaker_dir = os.path.join(out_b, sp)
        os.makedirs(speaker_dir, exist_ok=True)
        out_wav = os.path.join(speaker_dir, f"{sp}.wav")
        combined = AudioSegment.empty()
        for s in segs:
            start_ms = int(s['start'] * 1000)
            end_ms = int(s['end'] * 1000)
            combined += full_audio[start_ms:end_ms]
        combined.export(out_wav, format="wav")

    mapping_path = os.path.join(out_b, 'speaker_mapping.json')
    tmp_map = mapping_path + ".tmp"
    with open(tmp_map, "w") as f:
        json.dump(mapping, f, indent=2)
    os.replace(tmp_map, mapping_path)

    return {"speaker_mapping": mapping_path}
