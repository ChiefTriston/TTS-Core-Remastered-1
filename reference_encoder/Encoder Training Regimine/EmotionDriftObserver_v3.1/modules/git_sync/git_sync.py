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

def run(context):
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    config = context['config']
    
    # Aggregate for manifest
    total_slices = 0
    flagged_segments = 0
    observer_feedback = 0
    confidence_drift_slope = 0.0
    emotion_entropy = 0.0
    
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
        confidence_drift_slope += log['confidence_drift_slope']
        emotion_entropy += log['emotion_entropy']
    
    confidence_drift_slope /= len(speaker_ids)
    emotion_entropy /= len(speaker_ids)
    
    with open(os.path.join(output_dir, 'learned_rules.json'), 'r') as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        rules = json.load(f)
        portalocker.unlock(f)
    observer_feedback = len(rules.get('corrections', []))
    
    manifest = {
        'job_id': context['job_id'],
        'status': 'completed',
        'total_slices': total_slices,
        'flagged_segments': flagged_segments,
        'observer_feedback': observer_feedback,
        'confidence_drift_slope': confidence_drift_slope,
        'emotion_entropy': emotion_entropy,
        'arc': {}  # Stub from arc json
    }
    
    manifest_path = os.path.join(output_dir, 'job_manifest.json')
    with open(manifest_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(manifest, f)
        portalocker.unlock(f)
    
    repo_path = config['global']['github_repo_path']
    repo = git.Repo(repo_path)
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
    origin = repo.remote(name='origin')
    origin.push()
    
    return {'job_manifest': manifest_path, 'last_git_commit': commit_path}    
    return {'job_manifest': manifest_path, 'last_git_commit': commit_path}