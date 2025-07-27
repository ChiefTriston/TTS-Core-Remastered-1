# main.py
"""
Entry point for the Emotion Drift & Observer pipeline v3.1.
Parses config.yaml, starts folder watcher, or runs a single job via CLI.
"""

import argparse
import yaml
import os
from modules.trigger.trigger import run_trigger_watcher
from modules.diarization.diarization import run as diarization_run
from modules.prosody.prosody import run as prosody_run
from modules.drift.drift import run as drift_run
from modules.transcription.transcription import run as transcription_run
from modules.alignment.alignment import run as alignment_run
from modules.tier1.tier1 import run as tier1_run
from modules.tier2.tier2 import run as tier2_run
from modules.anomaly.anomaly import run as anomaly_run
from modules.fingerprint.fingerprint import run as fingerprint_run
from modules.arc.arc import run as arc_run
from modules.observer.observer import run as observer_run
from modules.git_sync.git_sync import run as git_sync_run
import uuid
import shutil

def pipeline(context):
    diarization_run(context)
    context['speaker_ids'] = [d for d in os.listdir(os.path.join(context['output_dir'], 'emotion_tags')) if os.path.isdir(os.path.join(context['output_dir'], 'emotion_tags', d))]
    
    prosody_run(context)
    drift_run(context)
    transcription_run(context)
    alignment_run(context)
    tier1_run(context)
    tier2_run(context)
    anomaly_run(context)
    fingerprint_run(context)
    arc_run(context)
    observer_run(context)
    git_sync_run(context)

def enqueue_job(job_id, input_wav):
    output_dir = os.path.join(config['global']['output_base'], job_id)
    os.makedirs(os.path.join(output_dir, 'emotion_tags'), exist_ok=True)
    context = {
        'job_id': job_id,
        'input_wav': input_wav,
        'output_dir': output_dir,
        'speaker_ids': [],
        'config': config
    }
    pipeline(context)

def main():
    parser = argparse.ArgumentParser(description="Emotion Drift & Observer v3.1")
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--watch', action='store_true', help='Start folder watcher')
    parser.add_argument('--job', help='Run single job on this audio file (creates .ready)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.job:
        # Simulate trigger for single job
        job_id = str(uuid.uuid4())
        ready_path = args.job + '.ready'
        with open(ready_path, 'w') as f:
            f.write('ready')
        enqueue_job(job_id, args.job)
        os.remove(ready_path)
    elif args.watch:
        run_trigger_watcher(
            config['global'], 
            enqueue_job
        )
    else:
        parser.print_help()

if __name__ == '__main__':
    main()