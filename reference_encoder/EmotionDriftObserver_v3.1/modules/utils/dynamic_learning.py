# dynamic_learning.py

import json, os, random, portalocker

from modules.utils.emotion_utils import emotion_rules
from sklearn.metrics.pairwise import cosine_similarity

VALIDATION_SET_FILE = os.path.join("…", "validation_set.json")
LEARNED_CONF_FILE    = os.path.join("…", "learned_confidences.json")
REVIEW_QUEUE         = os.path.join("…", "review_queue")
TIER1_ROOT           = os.path.join("…", "Tier 1 Emotions")

def load_validation_set():
    try:
        with open(VALIDATION_SET_FILE,"r") as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            data = json.load(f)
            portalocker.unlock(f)
            return data
    except FileNotFoundError:
        return []

def save_validation_set(vset):
    tmp = VALIDATION_SET_FILE + ".tmp"
    with open(tmp,"w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(vset,f,indent=2)
        portalocker.unlock(f)
    os.replace(tmp, VALIDATION_SET_FILE)

def load_learned_confidences():
    try:
        with open(LEARNED_CONF_FILE,"r") as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            data = json.load(f)
            portalocker.unlock(f)
            return data
    except FileNotFoundError:
        return {}

def save_learned_confidences(conf):
    tmp = LEARNED_CONF_FILE + ".tmp"
    with open(tmp,"w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(conf,f,indent=2)
        portalocker.unlock(f)
    os.replace(tmp, LEARNED_CONF_FILE)

def load_tagged_data():
    """Scan Tier‑1 accepts + rejects to accumulate accept/reject counts per emotion."""
    rule_updates = {emo:{"count":0,"conf_adjust":0.0,"reject_count":0,"total_examples":0}
                    for emo in emotion_rules}
    validation = load_validation_set()
    # …copy the logic that walks your Tier1 folders and the reject folder…
    return rule_updates, validation

def update_validation_set():
    """Add 5% sample of new Tier‑1 JSONs to validation set (cap 500)."""
    validation = load_validation_set()
    # …copy stratified-sampling logic…
    save_validation_set(validation)
    return validation

def update_emotion_rules(rule_updates, validation):
    """EMA‐smooth each confidence; alert on >5% accuracy drop; persist."""
    learned   = load_learned_confidences()
    drift_log = load_drift_log()  # the same drift_log loader from pipeline.py
    α = 0.9
    # …copy the update+alert logic…
    save_learned_confidences(learned)
    # rewrite in‐memory emotion_rules:
    for emo,conf in learned.items():
        fn,_ = emotion_rules[emo]
        emotion_rules[emo] = (fn,conf)
