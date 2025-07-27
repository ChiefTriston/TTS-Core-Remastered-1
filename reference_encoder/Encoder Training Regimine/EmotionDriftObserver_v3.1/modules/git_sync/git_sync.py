# modules/git_sync/git_sync.py
"""
Final manifest composition and GitHub sync using GitPython.
Outputs job_manifest.json and last_git_commit.json.
Pushes to GitHub.
"""

import json
import git
import os
import portalocker
import shutil
from datetime import datetime
import numpy as np
import time

def run(context):
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    config = context['config']
    
    # Aggregate stats
    total_slices = 0
    flagged_segments = 0
    observer_feedback_flag = False
    slopes = []
    entropies = []
    arc_label = ''
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        total_slices += len(transcript['slices'])
        
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)
        flagged_segments += len(drift.get('anomalies', []))
        
        with open(os.path.join(speaker_out, 'drift_log.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            log = json.load(f)
            portalocker.unlock(f)
        slopes.append(log['confidence_drift_slope'])
        entropies.append(log['emotion_entropy'])
    
    slope = np.mean(slopes)
    entropy = np.mean(entropies)
    
    with open(os.path.join(output_dir, 'learned_rules.json'), 'r') as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        rules = json.load(f)
        portalocker.unlock(f)
    observer_feedback_flag = len(rules.get('corrections', [])) > 0
    
    with open(os.path.join(output_dir, 'arc_classification.json'), 'r') as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        arc_data = json.load(f)
        portalocker.unlock(f)
    arc_label = arc_data['dominant_arc']
    
    manifest = {
        'job_id': context['job_id'],
        'status': 'complete',
        'total_slices': total_slices,
        'flagged_segments': flagged_segments,
        'observer_feedback': observer_feedback_flag,
        'slope': slope,
        'entropy': entropy,
        'arc_label': arc_label
    }
    
    manifest_path = os.path.join(output_dir, 'job_manifest.json')
    with open(manifest_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(manifest, f)
        portalocker.unlock(f)
    
    repo_path = config['global']['github_repo_path']
    repo = git.Repo(repo_path)
    
    branch = config['git_sync'].get('branch', 'main')
    remote_name = config['git_sync'].get('remote', 'origin')
    repo.git.checkout(branch)
    
    last_commit = repo.head.commit
    commit_info = {
        'hash': last_commit.hexsha,
        'timestamp': datetime.fromtimestamp(last_commit.authored_date).isoformat(),
        'pushed_by': repo.git.config('user.name')
    }
    commit_path = os.path.join(output_dir, 'last_git_commit.json')
    with open(commit_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(commit_info, f)
        portalocker.unlock(f)
    
    # Copy output to repo/emotion_tags/{job_id}
    target_dir = os.path.join(repo_path, config['global']['github_target_dir'], 'emotion_tags', context['job_id'])
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(output_dir, target_dir, dirs_exist_ok=True)
    
    repo.git.add(A=True)
    repo.index.commit(f"Emotion tags for job {context['job_id']}")
    
    origin = repo.remote(name=remote_name)
    retries = 3
    for attempt in range(retries):
        try:
            origin.push()
            break
        except git.exc.GitCommandError as e:
            print(f"Push failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            if attempt == retries - 1:
                # Rollback commit
                repo.git.reset('--hard', 'HEAD~1')
                raise
    
    return {'job_manifest': manifest_path, 'last_git_commit': commit_path}