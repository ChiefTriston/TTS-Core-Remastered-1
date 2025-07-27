# modules/observer/observer.py
"""
Streamlit UI for manual review and feedback.
Updates learned_rules.json at job level.
"""

import streamlit as st
import json
import os
import portalocker

all_emotions = [
    'Anger', 'Anxiety', 'Contempt', 'Despair', 'Disgust', 'Fear', 'Frustration', 'Guilt', 
    'Irritation', 'Jealousy', 'Loneliness', 'Negative Surprise', 'Sadness',
    'Boredom', 'Calm', 'Concentration', 'Flat narration', 'Hesitant', 
    'Matter-of-fact Informational tone', 'Neutral', 'Tired',
    'Amusement', 'Enthusiasm', 'Gratitude', 'Happiness', 'Hope', 'Inspiration', 
    'Love', 'Pleasant', 'Relief', 'Surprise'
]

def update_emotion_rules(feedback, rules_path):
    with open(rules_path, 'r+') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        rules = json.load(f)
        if 'corrections' not in rules:
            rules['corrections'] = []
        rules['corrections'].append(feedback)
        f.seek(0)
        json.dump(rules, f)
        f.truncate()
        portalocker.unlock(f)

def run(context):
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    rules_path = os.path.join(output_dir, 'learned_rules.json')
    if not os.path.exists(rules_path):
        with open(rules_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump({}, f)
            portalocker.unlock(f)
    
    st.title("Observer: Manual Review")
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tags = json.load(f)
            portalocker.unlock(f)
        
        st.subheader(f"Speaker {speaker_id}")
        corrections = []
        for idx, tag in enumerate(tags):
            corrected = st.selectbox(f"Slice {idx}: Current {tag['label']}", all_emotions, index=all_emotions.index(tag['label']) if tag['label'] in all_emotions else 0)
            if corrected != tag['label']:
                corrections.append({'slice': idx, 'correction': corrected, 'original': tag['label'], 'rule_id': tag['rule_id']})
        
        if st.button(f"Commit Feedback for {speaker_id}"):
            for corr in corrections:
                update_emotion_rules(corr, rules_path)
            st.success("Feedback committed!")
    
    return {'learned_rules': rules_path}