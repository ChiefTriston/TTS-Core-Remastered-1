# main.py
"""
Entry point for the Emotion Drift & Observer pipeline v3.1.
Parses config.yaml, starts folder watcher, or runs a single job via CLI.
"""

import argparse
import yaml
import os
import uuid
import shutil

from modules.trigger.trigger    import run_trigger_watcher
from modules.diarization.diarization import run as diarization_run
from modules.prosody.prosody    import run as prosody_run
from modules.drift.drift        import run as drift_run
from modules.transcription.transcription import run as transcription_run
from modules.alignment.alignment  import run as alignment_run
from modules.tier1.tier1        import run as tier1_run
from modules.tier2.tier2        import run as tier2_run
from modules.anomaly.anomaly    import run as anomaly_run
from modules.fingerprint.fingerprint import run as fingerprint_run
from modules.arc.arc            import run as arc_run
from modules.observer.observer  import run as observer_run
from modules.git_sync.git_sync  import run as git_sync_run
from modules.utils.dynamic_learning import (
    load_tagged_data,
    update_validation_set,
    update_emotion_rules
)

def pipeline(context):
    # 1. split speakers
    diarization_run(context)
    context['speaker_ids'] = [
        d for d in os.listdir(os.path.join(context['output_dir'], 'emotion_tags'))
        if os.path.isdir(os.path.join(context['output_dir'], 'emotion_tags', d))
    ]

    # 2. prosody → 3. drift → 4. transcribe → 5. align → 6. tier1 → 7. tier2 → 8. anomaly
    prosody_run(context)
    drift_run(context)
    transcription_run(context)
    alignment_run(context)
    tier1_run(context)
    tier2_run(context)
    anomaly_run(context)

    # 9. fingerprint → 10. arc → 11. observer → dynamic‐learning → 12. git sync
    fingerprint_run(context)
    arc_run(context)
    observer_run(context)

    # dynamic learning
    rule_updates, validation = load_tagged_data()
    validation = update_validation_set()
    update_emotion_rules(rule_updates, validation)

    git_sync_run(context)


def enqueue_job(config, job_id, input_wav):
    # Prepare an output folder for this job
    output_base = config['global']['output_base']
    output_dir = os.path.join(output_base, job_id)
    os.makedirs(os.path.join(output_dir, 'emotion_tags'), exist_ok=True)

    # Build the pipeline context
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
    parser.add_argument('--job', help='Run single job on this audio file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.job:
        # Run one‑off job immediately
        job_id = str(uuid.uuid4())
        enqueue_job(config, job_id, args.job)

    elif args.watch:
        # Start the folder watcher (uses .ready triggers)
        run_trigger_watcher(config['global'], enqueue_job)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
