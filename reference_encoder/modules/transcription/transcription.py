# transcription.py (Full Refined Patch: Load Model Once, Pass to Func)
"""
Transcription of slices using WhisperX + VAD cleanup.
Outputs transcript.json per speaker.
"""

import json
import whisperx
import portalocker
import os
import webrtcvad
import numpy as np
import torch

def run(context):
    gcfg = context['config']['global']
    cfg  = context['config']['transcription']

    # Determine compute device
    device = "cuda" if gcfg.get('use_gpu', False) and torch.cuda.is_available() else "cpu"
    # Set compute_type based on device
    compute_type = 'float32' if device == 'cpu' else 'float16'  # Use float32 for CPU, float16 for GPU
    if device == 'cpu' and cfg.get('compute_type') == 'float16':
        print("[Transcription] Warning: float16 not supported on CPU; using float32 instead")
    print(f"[Transcription] Using device: {device}")

    # Load WhisperX model once
    model = whisperx.load_model(cfg['model'], device, compute_type=compute_type)
    sr    = gcfg['sample_rate']
    vad_thresh = cfg.get('vad_thresh', 0.5)
    logprob_thresh = cfg.get('logprob_thresh', -1.0)

    results = {}
    for sp in context['speaker_ids']:
        speaker_dir = os.path.join(context['output_dir'], 'emotion_tags', sp)
        wav_path    = os.path.join(speaker_dir, f"{sp}.wav")
        if os.path.getsize(wav_path) > 1e9:  # Chunk if large
            info = torchaudio.info(wav_path)
            total_frames = info.num_frames
            chunk_frames = 600 * sr  # 10min
            offset = 0
            all_slices = []
            while offset < total_frames:
                num_frames = min(chunk_frames, total_frames - offset)
                waveform, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=num_frames)
                time_offset = offset / sr
                trans_chunk = transcribe_chunk(waveform, model, sr, vad_thresh, logprob_thresh, device)
                for s in trans_chunk['slices']:
                    s['start'] += time_offset
                    s['end'] += time_offset
                all_slices.extend(trans_chunk['slices'])
                offset += num_frames
        else:
            # Normal full load
            audio = whisperx.load_audio(wav_path)
            # load drift boundaries
            with open(os.path.join(speaker_dir, 'drift_vector.json'), 'r') as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                drift = json.load(f)
                portalocker.unlock(f)
            boundaries = drift.get('boundaries', [])

            all_slices = []
            for start, end in boundaries:
                start_s     = int(start * sr)
                end_s       = int(end   * sr)
                slice_audio = audio[start_s:end_s]

                # Transcribe slice
                result = model.transcribe(slice_audio)

                for seg in result['segments']:
                    seg_start = seg['start'] + start
                    seg_end   = seg['end']   + start

                    # VAD cleanup
                    frame_ms     = cfg.get('vad_frame_ms', 30)
                    frame_size   = int(sr * frame_ms / 1000)
                    total_frames = int((seg_end - seg_start) * sr / frame_size)
                    voiced       = 0
                    for i in range(total_frames):
                        frame = slice_audio[i*frame_size:(i+1)*frame_size]
                        frame_int16 = (frame * 32767).astype(np.int16)
                        if vad.is_speech(frame_int16.tobytes(), sr):
                            voiced += 1
                    vad_score = voiced / total_frames if total_frames > 0 else 0.0

                    # Accept segment if enough voice activity or high confidence
                    if vad_score >= vad_thresh or seg.get('avg_logprob', -10) > logprob_thresh:
                        all_slices.append({
                            'start': seg_start,
                            'end':   seg_end,
                            'text':  seg['text'],
                            'score': vad_score
                        })

        # Write out transcript
        transcript = {'slices': all_slices}
        out_path   = os.path.join(speaker_dir, 'transcript.json')
        with open(out_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(transcript, f, indent=2)
            portalocker.unlock(f)

        results[sp] = out_path

    return {'transcript': results}

def transcribe_chunk(waveform, model, sr=22050, vad_thresh=0.5, logprob_thresh=-1.0, device='cpu'):
    """Full transcribe on chunk waveform tensor."""
    audio_np = waveform.numpy().squeeze()
    result = model.transcribe(audio_np)
    vad = webrtcvad.Vad(3)
    frame_ms = 30
    frame_size = int(sr * frame_ms / 1000)
    cleaned = []
    for seg in result['segments']:
        seg_start = seg['start']
        seg_end = seg['end']
        slice_audio = audio_np[int(seg_start * sr):int(seg_end * sr)]
        total_frames = int(len(slice_audio) / frame_size)
        voiced = 0
        for i in range(total_frames):
            frame = slice_audio[i*frame_size:(i+1)*frame_size]
            frame_int16 = (frame * 32767).astype(np.int16)
            if vad.is_speech(frame_int16.tobytes(), sr):
                voiced += 1
        vad_score = voiced / total_frames if total_frames > 0 else 0.0
        if vad_score >= vad_thresh or seg.get('avg_logprob', -10) > logprob_thresh:
            cleaned.append({
                'start': seg_start,
                'end': seg_end,
                'text': seg['text'],
                'score': vad_score
            })
    return {'slices': cleaned}
