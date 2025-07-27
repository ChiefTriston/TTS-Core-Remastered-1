# modules/git_sync/git_sync.py (updated with import shutil)
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

def run(context):
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    config = context['config']
    
    manifest = {}
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'fingerprint.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            fp = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'drift_log.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            log = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(output_dir, 'arc_classification.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            arc = json.load(f)
            portalocker.unlock(f)
        
        manifest[speaker_id] = {'fingerprint': fp, 'drift_log': log, 'arc': arc['dominant_arc']}
    
    manifest_path = os.path.join(output_dir, 'job_manifest.json')
    with open(manifest_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(manifest, f)
        portalocker.unlock(f)
    
    repo_path = config['global']['github_repo_path']
    repo = git.Repo(repo_path)
    last_commit = repo.head.commit.hexsha
    commit_path = os.path.join(output_dir, 'last_git_commit.json')
    with open(commit_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump({'commit': last_commit}, f)
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