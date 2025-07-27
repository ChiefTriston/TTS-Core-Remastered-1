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
    config = context['config']['diarization']
    global_config = context['config']['global']
    job_id = context['job_id']
    input_wav = context['input_wav']
    output_dir = os.path.join(context['output_dir'], 'emotion_tags')
    device = "cuda" if global_config['use_gpu'] and torch.cuda.is_available() else "cpu"
    hf_token = global_config['hf_token']
    
    batch_size = config['batch_size']
    compute_type = config['compute_type']
    model_name = config['model']
    
    retries = 3
    for attempt in range(retries):
        try:
            audio = whisperx.load_audio(input_wav)
            model = whisperx.load_model(model_name, device, compute_type=compute_type)
            result = model.transcribe(audio, batch_size=batch_size)
            
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5)  # Retry delay
    
    # Group segments by speaker
    speaker_segments = {}
    speaker_mapping = {}
    for seg in result['segments']:
        speaker = seg.get('speaker', 'unknown')
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
            speaker_mapping[speaker] = {'timestamps': [], 'confidences': [], 'word_alignment': []}
        speaker_segments[speaker].append(seg)
        speaker_mapping[speaker]['timestamps'].append((seg['start'], seg['end']))
        speaker_mapping[speaker]['confidences'].append(seg.get('score', 0.9))  # From whisper or stub
        speaker_mapping[speaker]['word_alignment'].extend(seg.get('words', []))  # Word-level with speaker
    
    # Stitching rules
    gap_thresh = config.get('gap_thresh', 0.5)
    min_len = config.get('min_len', 1.0)
    max_len = config.get('max_len', 30.0)
    
    for speaker in speaker_segments:
        segments = sorted(speaker_segments[speaker], key=lambda s: s['start'])
        stitched = []
        current = segments[0]
        for seg in segments[1:]:
            if seg['start'] - current['end'] < gap_thresh:
                current['end'] = max(current['end'], seg['end'])  # Merge
                current['text'] += ' ' + seg['text']
                current['words'] += seg['words']
            else:
                if current['end'] - current['start'] < min_len:
                    # Drop short
                    continue
                if current['end'] - current['start'] > max_len:
                    # Split long stub
                    pass
                stitched.append(current)
                current = seg
        stitched.append(current)
        speaker_segments[speaker] = stitched
        
        # Update mapping
        speaker_mapping[speaker]['timestamps'] = [(s['start'], s['end']) for s in stitched]
        speaker_mapping[speaker]['confidences'] = [0.9] * len(stitched)  # Stub
    
    # Save speaker wavs
    audio_seg = AudioSegment.from_wav(input_wav)
    for speaker in speaker_segments.keys():
        speaker_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        wav_path = os.path.join(speaker_dir, f'{speaker}.wav')
        combined = AudioSegment.empty()
        for seg in speaker_segments[speaker]:
            start_ms = seg['start'] * 1000
            end_ms = seg['end'] * 1000
            combined += audio_seg[start_ms:end_ms]
        with open(wav_path, 'wb') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            combined.export(f, format='wav')
            portalocker.unlock(f)
    
    mapping_path = os.path.join(output_dir, 'speaker_mapping.json')
    with open(mapping_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(speaker_mapping, f)
        portalocker.unlock(f)
    
    return {'speaker_mapping': mapping_path}