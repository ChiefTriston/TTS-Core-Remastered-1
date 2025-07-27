# modules/observer/observer.py
"""
Streamlit UI for manual review and feedback.
Updates learned_rules.json at job level.
"""

import streamlit as st
import json
import os
import portalocker
import matplotlib.pyplot as plt
import numpy as np

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
    
    st.title("Observer: Manual Review Dashboard")
    
    # Multi-view tabs
    tab1, tab2 = st.tabs(["Overview", "Per-Speaker Review"])
    
    with tab1:
        st.header("Global Overview")
        # Stub global plots, e.g. aggregate prosody
        for speaker_id in speaker_ids:
            speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
            with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                prosody = json.load(f)
                portalocker.unlock(f)
            time = np.array(prosody['frame_series']['time'])
            f0_z = np.array(prosody['frame_series']['f0_z'])
            fig, ax = plt.subplots()
            ax.plot(time, f0_z, label='F0 Z')
            st.pyplot(fig)
    
    with tab2:
        # Per-speaker tabs
        speaker_tab = st.selectbox("Select Speaker", speaker_ids)
        
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_tab)
        with open(os.path.join(speaker_out, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tags = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)
        
        # Prosody trends plot
        st.subheader("Prosody Trends")
        time = np.array(prosody['frame_series']['time'])
        f0_z = np.array(prosody['frame_series']['f0_z'])
        energy_z = np.array(prosody['frame_series']['energy_z'])
        fig, ax = plt.subplots()
        ax.plot(time, f0_z, label='F0 Z')
        ax.plot(time, energy_z, label='Energy Z')
        ax.legend()
        st.pyplot(fig)
        
        # Drift vector plot
        st.subheader("Drift Vector")
        deltas = np.array(drift['deltas'])
        fig, ax = plt.subplots()
        ax.plot(deltas, label='Smoothed Deltas')
        ax.legend()
        st.pyplot(fig)
        
        # Pagination for slices
        num_slices = len(tags)
        page_size = 10
        page = st.number_input("Page", min_value=1, max_value=(num_slices // page_size) + 1, value=1)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, num_slices)
        
        corrections = []
        for idx in range(start_idx, end_idx):
            tag = tags[idx]
            slice_text = transcript['slices'][idx]['text']
            st.write(f"Slice {idx}: Transcript Snippet: {slice_text[:100]}...")
            corrected = st.selectbox(f"Current {tag['label']}", all_emotions, index=all_emotions.index(tag['label']) if tag['label'] in all_emotions else 0, key=f"select_{idx}")
            notes = st.text_input("Notes", key=f"notes_{idx}")
            severity = st.slider("Severity (1-5)", 1, 5, 1, key=f"severity_{idx}")
            if corrected != tag['label'] or notes or severity > 1:
                corrections.append({'slice': idx, 'correction': corrected, 'original': tag['label'], 'rule_id': tag['rule_id'], 'notes': notes, 'severity': severity})
        
        # Real-time rule update
        st.subheader("Suggested Rules")
        suggested_rules = []  # Stub: generate based on corrections, e.g. frequent negations
        for corr in corrections:
            if corr['rule_id'] == 'negation_invert':
                suggested_rules.append(f"Increase negation_weight by 0.1 for {corr['original']} to {corr['correction']}")
        
        for rule in suggested_rules:
            if st.checkbox(rule):
                # Update rules.json with new weight or rule
                with open(rules_path, 'r+') as f:
                    portalocker.lock(f, portalocker.LOCK_EX)
                    rules = json.load(f)
                    rules['negation_weight'] = rules.get('negation_weight', 1.0) + 0.1
                    f.seek(0)
                    json.dump(rules, f)
                    f.truncate()
                    portalocker.unlock(f)
        
        if st.button(f"Commit Feedback for {speaker_tab}"):
            for corr in corrections:
                update_emotion_rules(corr, rules_path)
            st.success("Feedback committed!")
    
    return {'learned_rules': rules_path}