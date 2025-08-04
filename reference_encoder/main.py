# main.py (Refined Patch: FFmpeg Splitting, GPU Memory Mgmt, Parallel for More Steps)
import argparse
import yaml
import os
import uuid
import threading
import subprocess
import sys
import shutil
import torch
import numpy as np
import torchaudio
import psutil  # For RAM monitoring; pip install psutil if needed
from concurrent.futures import ThreadPoolExecutor, as_completed
import whisperx
import webrtcvad
import portalocker
import json
import logging  # For structured logging

from modules.prosody.prosody import has_prosody3, ProsodyPredictorV15

from hyper_diarizer.cli import DiarizerController
from modules.trigger.trigger import job_queue, run_trigger_watcher
from modules.prosody.prosody import run as prosody_run
from modules.drift.drift import run as drift_run
from modules.transcription.transcription import run as transcription_run
from modules.alignment.alignment import run as alignment_run
from modules.tier1.tier1 import run as tier1_run
from modules.tier2.tier2 import run as tier2_run
from modules.anomaly.anomaly import run as anomaly_run
from modules.fingerprint.fingerprint import run as fingerprint_run
from modules.arc.arc import run as arc_run
from modules.plot_map.plot_map import run as plot_map_run
from modules.observer.observer import run as observer_run
from modules.git_sync.git_sync import run as git_sync_run
from modules.utils.dynamic_learning import (
    load_tagged_data,
    update_validation_set,
    check_accuracy_drop
)

# Setup structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# In-memory status tracker
descriptions = {}

def log_gpu_status():
    """
    Log NVIDIA GPU status using nvidia-smi.
    """
    logging.info("\n=== GPU status (nvidia-smi) ===")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except FileNotFoundError:
        logging.warning("  nvidia-smi not found: no NVIDIA GPU or driver missing.")
    except subprocess.CalledProcessError as e:
        logging.warning(f"  nvidia-smi failed (code {e.returncode})")
    logging.info("=" * 32 + "\n")

def log_ram_status():
    """Log current RAM usage."""
    ram = psutil.virtual_memory()
    logging.info(f"\n=== RAM Status ===\nUsed: {ram.used / 1e9:.2f}GB / Total: {ram.total / 1e9:.2f}GB ({ram.percent}% used)\n===============\n")

def pipeline(context):
    # 1. Speaker splitting via HyperDiazer
    tmp_dir = os.path.join(context['output_dir'], 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    hz_args = argparse.Namespace(
        input=context['input_wav'],
        output_dir=tmp_dir,
        visualize=False,
        batch=False,
        eval=None,
        hmm_off=False,
        voiceprint_thresh=context['config'].get('hyperdiazer', {}).get('voiceprint_thresh', 0.6),
        streaming=False
    )
    controller = DiarizerController(hz_args)
    hyper_out = controller.diarize_single(context['input_wav'], tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)  # Cleanup tmp_dir after diarization

    # Collect speaker IDs and copy per-speaker WAVs
    context['speaker_ids'] = []
    emotion_dir = os.path.join(context['output_dir'], 'emotion_tags')
    for spk, wav_path in hyper_out.get('speaker_paths', {}).items():
        context['speaker_ids'].append(spk)
        dest_dir = os.path.join(emotion_dir, spk)
        os.makedirs(dest_dir, exist_ok=True)
        dest_wav = os.path.join(dest_dir, f"{spk}.wav")
        shutil.copy(wav_path, dest_wav)

    # Load models once for GPU
    device = 'cuda' if context['config']['global'].get('use_gpu', False) and torch.cuda.is_available() else 'cpu'
    # Load WhisperX for transcription
    whisper_model = whisperx.load_model(context['config']['transcription']['model'], device, context['config']['transcription'].get('compute_type', 'float16' if device == 'cuda' else 'float32'))
    # Load Prosody model if available
    if has_prosody3:
        prosody_model = ProsodyPredictorV15(**context['config']['prosody']).to(device)
        prosody_model.eval()
    else:
        prosody_model = None
    context['whisper_model'] = whisper_model
    context['prosody_model'] = prosody_model
    context['device'] = device

    # Set hop_length consistently
    sr = context['config']['global']['sample_rate']
    extract_freq = context['config']['prosody'].get('extract_freq', 50)
    context['hop_length'] = sr // extract_freq  # Integer division

    # Patch: Check for large speaker WAVs and process in chunks if needed
    chunk_sec = context['config']['global'].get('chunk_sec', 120)  # 2 min as suggested
    chunk_overlap_sec = context['config']['global'].get('chunk_overlap_sec', 0.5)
    use_ffmpeg_split = context['config']['global'].get('use_ffmpeg_split', True)
    for spk in context['speaker_ids']:
        spk_dir = os.path.join(emotion_dir, spk)
        wav_path = os.path.join(spk_dir, f"{spk}.wav")
        if os.path.getsize(wav_path) > 1e9:  # >1GB
            logging.info(f"[Pipeline] Chunking large WAV for {spk}")
            chunk_process_speaker(wav_path, spk_dir, sr, chunk_sec, chunk_overlap_sec, use_ffmpeg_split, context, device)
        else:
            # Normal non-chunked processing
            prosody_run(context)
            transcription_run(context)
        log_ram_status()
        log_gpu_status()

    # Remaining steps use small JSONs/arrays, no chunking needed
    drift_run(context)
    alignment_run(context)
    tier1_run(context)
    tier2_run(context)
    anomaly_run(context)
    fingerprint_run(context)
    arc_run(context)
    plot_map_run(context)
    observer_run(context)

    # 5. Dynamic learning updates
    data_root = os.path.join(context['output_dir'], 'emotion_tags')
    validation_path = os.path.join(context['output_dir'], 'validation_set.json')
    load_tagged_data(data_root)
    update_validation_set(data_root, validation_path,
                          sample_frac=context['config'].get('dynamic_learning', {}).get('sample_frac', 0.05),
                          max_samples=context['config'].get('dynamic_learning', {}).get('max_samples', 500))
    old_acc = context.get('old_accuracy', 1.0)
    new_acc = context.get('new_accuracy', 1.0)
    check_accuracy_drop(old_acc, new_acc)

    # 6. Git synchronization
    git_sync_run(context)

def chunk_process_speaker(wav_path, spk_dir, sr=22050, chunk_sec=120, overlap_sec=0.5, use_ffmpeg=True, context=None, device='cpu'):
    """Chunk large speaker WAV, process prosody/transcription per chunk in parallel, merge with offsets."""
    if context is None:
        context = {}
    chunk_dir = os.path.join(spk_dir, 'chunks')
    os.makedirs(chunk_dir, exist_ok=True)
    overlap_frames = int(overlap_sec * sr)
    chunk_paths = []
    if use_ffmpeg:
        # Split to disk with FFmpeg (faster, low mem)
        duration = torchaudio.info(wav_path).num_frames / sr
        start = 0
        chunk_idx = 0
        while start < duration:
            chunk_dur = min(chunk_sec + overlap_sec if start + chunk_sec < duration else duration - start, chunk_sec + overlap_sec)
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx}.wav")
            try:
                subprocess.run([
                    'ffmpeg', '-i', wav_path, '-ss', str(start), '-t', str(chunk_dur),
                    '-c:a', 'pcm_s16le', chunk_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                print(f"[Pipeline] Warning: chunk {chunk_idx} failed to split, skipping.")
                continue
            chunk_paths.append((chunk_path, start))
            start += chunk_sec  # Step by chunk_sec, overlap handled in dur
            chunk_idx += 1
    else:
        # In-memory chunking (fallback)
        info = torchaudio.info(wav_path)
        total_frames = info.num_frames
        chunk_frames = chunk_sec * sr
        offset = 0
        chunk_idx = 0
        while offset < total_frames:
            num_frames = min(chunk_frames + overlap_frames if offset + chunk_frames < total_frames else total_frames - offset, chunk_frames + overlap_frames)
            waveform, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=num_frames)
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx}.wav")
            torchaudio.save(chunk_path, waveform, sr)
            chunk_paths.append((chunk_path, offset / sr))
            offset += chunk_frames  # Step by chunk_frames
            chunk_idx += 1

    if not chunk_paths:
        print(f"[Pipeline] No valid chunks for {spk}, skipping prosody/transcription.")
        # Alert or clean up partial files
        if os.path.exists(os.path.join(spk_dir, 'prosody_trend.json')):
            os.remove(os.path.join(spk_dir, 'prosody_trend.json'))
        if os.path.exists(os.path.join(spk_dir, 'transcript.json')):
            os.remove(os.path.join(spk_dir, 'transcript.json'))
        return

    # Parallel process chunks with ThreadPool (data+model on GPU)
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust based on GPU
        future_to_offset = {executor.submit(process_chunk, chunk_path, time_offset, sr, context['prosody_model'], context['whisper_model'], device, context['hop_length'], context['config']['transcription']['vad_thresh'], context['config']['transcription']['logprob_thresh']): time_offset for chunk_path, time_offset in chunk_paths}
        all_time = np.array([])
        all_f0_z = np.array([])
        all_energy_z = np.array([])
        all_pitch_var_z = np.array([])
        all_slices = []
        sr_rate_sum = 0.0
        pause_dur_sum = 0.0
        mfcc_list = []
        chunk_failures = 0
        for future in as_completed(future_to_offset):
            time_offset = future_to_offset[future]
            try:
                prosody_chunk, trans_chunk = future.result()
            except Exception as e:
                print(f"[Pipeline] Chunk failed: {e}")
                chunk_failures += 1
                continue
            # Merge prosody
            all_time = np.concatenate([all_time, prosody_chunk['time']])
            all_f0_z = np.concatenate([all_f0_z, prosody_chunk['f0_z']])
            all_energy_z = np.concatenate([all_energy_z, prosody_chunk['energy_z']])
            all_pitch_var_z = np.concatenate([all_pitch_var_z, prosody_chunk['pitch_var_z']])
            sr_rate_sum += prosody_chunk['speech_rate']
            pause_dur_sum += prosody_chunk['pause_dur']
            mfcc_list.extend(prosody_chunk['mfcc'])

            # Merge transcription (no re-offset here)
            all_slices.extend(trans_chunk['slices'])

    # Sort merged by time (in case threads out-of-order)
    sort_idx = np.argsort(all_time)
    all_time = all_time[sort_idx]
    all_f0_z = all_f0_z[sort_idx]
    all_energy_z = all_energy_z[sort_idx]
    all_pitch_var_z = all_pitch_var_z[sort_idx]
    all_slices.sort(key=lambda x: x['start'])

    # Global averages
    num_chunks = len(chunk_paths)
    sr_rate = sr_rate_sum / num_chunks if num_chunks > 0 else 0.0
    pause_dur = pause_dur_sum / num_chunks if num_chunks > 0 else 0.0
    mfcc = np.array(mfcc_list)

    if len(all_time) == 0:
        print(f"[Pipeline] No prosody frames for {spk}, skipping write.")
        # Alert or clean up partial files
        if os.path.exists(os.path.join(spk_dir, 'prosody_trend.json')):
            os.remove(os.path.join(spk_dir, 'prosody_trend.json'))
        if os.path.exists(os.path.join(spk_dir, 'transcript.json')):
            os.remove(os.path.join(spk_dir, 'transcript.json'))
        return

    # Write merged prosody (rest as before)
    merged_prosody = {
        'frame_series': {
            'time': all_time.tolist(),
            'f0_z': all_f0_z.tolist(),
            'energy_z': all_energy_z.tolist(),
            'pitch_var_z': all_pitch_var_z.tolist()
        },
        'globals': {
            'speech_rate': sr_rate,
            'pause_dur': pause_dur,
            'pause_ratio': pause_dur / all_time[-1] if len(all_time) > 0 else 0.0,
            'mfcc': mfcc.tolist()
        },
        # trendlines/summaries on merged
    }
    prosody_path = os.path.join(spk_dir, 'prosody_trend.json')
    with portalocker.Lock(prosody_path, 'w', timeout=5) as f:
        json.dump(merged_prosody, f, indent=2)

    # Write merged transcript
    transcript_path = os.path.join(spk_dir, 'transcript.json')
    with portalocker.Lock(transcript_path, 'w', timeout=5) as f:
        json.dump({'slices': all_slices}, f, indent=2)

    # Clean up chunks
    shutil.rmtree(chunk_dir, ignore_errors=True)

    # Check for chunk failures and set status
    if chunk_failures > 0:
        descriptions[context['job_id']] = 'partial-failure'
    else:
        descriptions[context['job_id']] = 'done'

def process_chunk(chunk_path, time_offset, sr, prosody_model, whisper_model, device, hop_length, vad_thresh=0.5, logprob_thresh=-1.0):
    """Process prosody and transcription on chunk, return (prosody_dict, trans_dict)."""
    # Prosody
    try:
        prosody_chunk = prosody_extract(chunk_path, sr=sr, prosody_model=prosody_model, device=device, hop_length=hop_length)
        prosody_chunk['time'] += time_offset
    except Exception as e:
        print(f"[Pipeline] Prosody chunk failed: {e}")
        prosody_chunk = {'time': np.array([]), 'f0_z': np.array([]), 'energy_z': np.array([]), 'pitch_var_z': np.array([]), 'speech_rate': 0.0, 'pause_dur': 0.0, 'mfcc': []}

    # Transcription
    try:
        trans_chunk = transcribe_chunk(chunk_path, whisper_model, sr=sr, vad_thresh=vad_thresh, logprob_thresh=logprob_thresh, device=device)
    except Exception as e:
        print(f"[Pipeline] Transcription chunk failed: {e}")
        trans_chunk = {'slices': []}

    # Clear GPU
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return prosody_chunk, trans_chunk

# Full prosody_extract implementation
def prosody_extract(chunk_path, sr=22050, prosody_model=None, device='cpu', hop_length=None):
    """Extract prosody features from WAV path. Returns dict with arrays."""
    if hop_length is None:
        hop_length = 441  # Fallback if not passed
    if has_prosody3 and prosody_model is not None:
        waveform, _ = torchaudio.load(chunk_path)
        waveform = waveform.to(device)
        mel = prosody_model.mel_spec(waveform)
        prosody = prosody_model(mel)
        f0 = prosody['f0'].squeeze(0).cpu().numpy()
        eng = prosody['energy'].squeeze(0).cpu().numpy()
        pv = prosody['pitch_var'].squeeze(0).cpu().numpy()
        sr_rate = prosody['speech_rate'].item()
        pause_dur = prosody['pause_dur'].item()
        mfcc = prosody.get('mfcc', torch.zeros(0)).squeeze(0).cpu().numpy()
        del waveform, mel, prosody  # Free tensors
    else:
        y, _ = librosa.load(chunk_path, sr=sr)
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=hop_length/sr)
        f0 = pitch.selected_array['frequency']
        energy = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length).flatten()
        pv = np.var(f0[np.isfinite(f0)])
        sr_rate = 0.0
        pause_dur = 0.0
        mfcc = []
    time = np.arange(0, len(f0) * (hop_length / sr), hop_length / sr)
    f0_z = (f0 - f0.mean()) / f0.std() if f0.std() > 0 else f0
    energy_z = (energy - energy.mean()) / energy.std() if energy.std() > 0 else energy
    pv_z = (pv - np.mean(pv)) / np.std(pv) if np.std(pv) > 0 else pv  # Note: pv is scalar per chunk, adjust if needed
    return {'time': time, 'f0_z': f0_z, 'energy_z': energy_z, 'pitch_var_z': pv_z, 'speech_rate': sr_rate, 'pause_dur': pause_dur, 'mfcc': mfcc}

# Full transcribe_chunk implementation
def transcribe_chunk(chunk_path, whisper_model, sr=22050, vad_thresh=0.5, logprob_thresh=-1.0, device='cpu'):
    """Full transcribe on chunk WAV path."""
    audio = whisperx.load_audio(chunk_path)
    result = whisper_model.transcribe(audio)
    vad = webrtcvad.Vad(3)
    frame_ms = 30
    frame_size = int(sr * frame_ms / 1000)
    cleaned = []
    for seg in result['segments']:
        seg_start = seg['start']
        seg_end = seg['end']
        slice_audio = audio[int(seg_start * sr):int(seg_end * sr)]
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
    del audio, result  # Free memory
    return {'slices': cleaned}

def enqueue_job(config, job_id, input_wav):
    output_base = config['global']['output_base']
    output_dir = os.path.join(output_base, job_id)
    os.makedirs(os.path.join(output_dir, 'emotion_tags'), exist_ok=True)

    context = {
        'job_id': job_id,
        'input_wav': input_wav,
        'output_dir': output_dir,
        'speaker_ids': [],
        'config': config
    }

    # Load models
    lang = config.get('diarization', {}).get('language', 'en')
    device = 'cuda' if config['global'].get('use_gpu', False) and torch.cuda.is_available() else 'cpu'

    pipeline(context)


def worker():
    while True:
        config, job_id, input_wav = job_queue.get()
        descriptions[job_id] = 'processing'
        try:
            enqueue_job(config, job_id, input_wav)
            descriptions[job_id] = 'done'
        except Exception as e:
            descriptions[job_id] = 'failed'
            print(f"Job {job_id} failed: {e}")
        finally:
            job_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description='Emotion Drift & Observer v3.1')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--job', help='Path to input WAV file')
    args = parser.parse_args()

    log_gpu_status()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.job:
        job_id = str(uuid.uuid4())
        descriptions[job_id] = 'queued'
        enqueue_job(config, job_id, args.job)
    elif args.watch:
        run_trigger_watcher(config['global'])
        threading.Thread(target=worker, daemon=True).start()
        threading.Event().wait()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
