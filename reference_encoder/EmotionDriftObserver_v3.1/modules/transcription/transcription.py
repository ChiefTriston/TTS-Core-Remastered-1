# modules/transcription/transcription.py
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

def run(context):
    gcfg = context['config']['global']
    cfg  = context['config']['transcription']
    device = "cuda" if gcfg['use_gpu'] and whisperx.is_available() else "cpu"
    model  = whisperx.load_model(cfg['model'], device)
    sr     = gcfg['sample_rate']
    vad    = webrtcvad.Vad(3)  # aggressive mode

    results = {}
    for sp in context['speaker_ids']:
        speaker_dir = os.path.join(context['output_dir'], 'emotion_tags', sp)
        wav_path    = os.path.join(speaker_dir, f"{sp}.wav")
        audio       = whisperx.load_audio(wav_path)

        # load drift boundaries
        with open(os.path.join(speaker_dir, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)
        boundaries = drift.get('boundaries', [])

        cleaned = []
        for start, end in boundaries:
            start_s = int(start * sr)
            end_s   = int(end   * sr)
            slice_audio = audio[start_s:end_s]
            result = model.transcribe(slice_audio)

            for seg in result['segments']:
                seg_start = seg['start'] + start
                seg_end   = seg['end']   + start

                # VAD cleanup
                frame_ms    = 30
                frame_size  = int(sr * frame_ms / 1000)
                total_frames = int((seg_end - seg_start) * sr / frame_size)
                voiced = 0
                for i in range(total_frames):
                    frame = slice_audio[i*frame_size:(i+1)*frame_size]
                    frame_int16 = (frame * 32767).astype(np.int16)
                    if vad.is_speech(frame_int16.tobytes(), sr):
                        voiced += 1
                vad_score = voiced / total_frames if total_frames > 0 else 0.0

                if vad_score > cfg['vad_thresh'] or seg.get('avg_logprob', -10) > -1.0:
                    cleaned.append({
                        'start': seg_start,
                        'end':   seg_end,
                        'text':  seg['text'],
                        'score': vad_score
                    })

        transcript = {'slices': cleaned}
        out_path   = os.path.join(speaker_dir, 'transcript.json')
        with open(out_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(transcript, f, indent=2)
            portalocker.unlock(f)

        results[sp] = out_path

    return {'transcript': results}

