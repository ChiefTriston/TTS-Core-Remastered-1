# modules/observer/observer.py
"""
Streamlit UI for manual review and feedback.
Updates learned_rules.json at job level.
"""
import streamlit as st
import json
import os
import portalocker
import numpy as np
import plotly.graph_objects as go
from modules.utils.plot_utils import plotly_trends, save_drift_plot

all_emotions = [
    'Anger', 'Anxiety', 'Contempt', 'Despair', 'Disgust', 'Fear', 'Frustration',
    'Guilt', 'Irritation', 'Jealousy', 'Loneliness', 'Negative Surprise',
    'Sadness', 'Boredom', 'Calm', 'Concentration', 'Flat narration', 'Hesitant',
    'Matter-of-fact Informational tone', 'Neutral', 'Tired', 'Amusement',
    'Enthusiasm', 'Gratitude', 'Happiness', 'Hope', 'Inspiration', 'Love',
    'Pleasant', 'Relief', 'Surprise'
]


def update_emotion_rules(feedback, rules_path):
    with open(rules_path, 'r+') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        rules = json.load(f)
        corrections = rules.setdefault('corrections', [])
        corrections.append(feedback)
        f.seek(0)
        json.dump(rules, f, indent=2)
        f.truncate()
        portalocker.unlock(f)


def run(context):
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']

    # Ensure learned_rules.json exists
    rules_path = os.path.join(output_dir, 'learned_rules.json')
    if not os.path.exists(rules_path):
        with open(rules_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump({}, f)
            portalocker.unlock(f)

    # Sidebar job summary
    st.sidebar.header("Job Summary")
    total_slices = 0
    flagged_anomalies = 0
    for spk in speaker_ids:
        spk_dir = os.path.join(output_dir, 'emotion_tags', spk)
        try:
            with open(os.path.join(spk_dir, 'transcript.json'), 'r') as tf:
                tdata = json.load(tf)
            total_slices += len(tdata.get('slices', []))
        except FileNotFoundError:
            pass
        try:
            with open(os.path.join(spk_dir, 'drift_vector.json'), 'r') as df:
                ddata = json.load(df)
            flagged_anomalies += len(ddata.get('anomalies', []))
        except FileNotFoundError:
            pass
    # Arc classification
    arc_path = os.path.join(output_dir, 'arc_classification.json')
    try:
        with open(arc_path, 'r') as af:
            arc = json.load(af)
        completed_arcs = arc.get('named_arc', 'N/A')
    except FileNotFoundError:
        completed_arcs = 'N/A'

    st.sidebar.write(f"Total Slices: {total_slices}")
    st.sidebar.write(f"Flagged Anomalies: {flagged_anomalies}")
    st.sidebar.write(f"Completed Arcs: {completed_arcs}")

    st.title("Observer: Manual Review Dashboard")

    tab1, tab2 = st.tabs(["Global Overview", "Per-Speaker Review"])

    with tab1:
        st.header("Global Prosody Overview")
        for spk in speaker_ids:
            spk_dir = os.path.join(output_dir, 'emotion_tags', spk)
            try:
                with open(os.path.join(spk_dir, 'prosody_trend.json'), 'r') as pf:
                    prosody = json.load(pf)
                time = np.array(prosody['frame_series']['time'])
                f0_z = np.array(prosody['frame_series']['f0_z'])
                energy_z = np.array(prosody['frame_series']['energy_z'])
                st.subheader(f"{spk} Prosody")
                st.plotly_chart(plotly_trends(time, f0_z, energy_z), use_container_width=True)
            except FileNotFoundError:
                continue

    with tab2:
        st.header("Per-Speaker Review")
        speaker_tab = st.selectbox("Select Speaker", speaker_ids)
        spk_dir = os.path.join(output_dir, 'emotion_tags', speaker_tab)

        # Load data
        try:
            with open(os.path.join(spk_dir, 'tier2_tags.json'), 'r') as tf:
                tags = json.load(tf)
        except FileNotFoundError:
            tags = []
        try:
            with open(os.path.join(spk_dir, 'transcript.json'), 'r') as tf:
                transcript = json.load(tf)
        except FileNotFoundError:
            transcript = {'slices': []}
        try:
            with open(os.path.join(spk_dir, 'drift_vector.json'), 'r') as df:
                drift = json.load(df)
        except FileNotFoundError:
            drift = {'deltas': [], 'anomalies': []}
        try:
            with open(os.path.join(spk_dir, 'prosody_trend.json'), 'r') as pf:
                prosody = json.load(pf)
        except FileNotFoundError:
            prosody = {'frame_series': {'time': [], 'f0_z': [], 'energy_z': []}}

        # Prosody trends
        st.subheader("Prosody Trends")
        time = np.array(prosody['frame_series']['time'])
        f0_z = np.array(prosody['frame_series']['f0_z'])
        energy_z = np.array(prosody['frame_series']['energy_z'])
        st.plotly_chart(plotly_trends(time, f0_z, energy_z), use_container_width=True)

        # Drift vector
        st.subheader("Drift Vector")
        deltas = drift.get('deltas', [])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(deltas))), y=deltas, mode='lines', name='Delta'))
        fig.update_layout(
            title="Drift Vector",
            xaxis_title="Slice Index",
            yaxis_title="Delta"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pagination and feedback
        num_slices = len(tags)
        page_size = 10
        total_pages = (num_slices - 1) // page_size + 1
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, num_slices)

        corrections = []
        for idx in range(start_idx, end_idx):
            tag = tags[idx]
            slice_text = transcript['slices'][idx].get('text', '')
            st.write(f"Slice {idx}: {slice_text[:100]}...")
            corrected = st.selectbox(
                f"Label (original: {tag.get('label')})", all_emotions,
                index=all_emotions.index(tag.get('label')) if tag.get('label') in all_emotions else 0,
                key=f"select_{idx}"
            )
            notes = st.text_input("Notes", key=f"notes_{idx}")
            severity = st.slider("Severity", 1, 5, 1, key=f"severity_{idx}")
            if corrected != tag.get('label') or notes or severity > 1:
                corrections.append({
                    'slice': idx,
                    'correction': corrected,
                    'original': tag.get('label'),
                    'rule_id': tag.get('rule_id'),
                    'notes': notes,
                    'severity': severity
                })

        st.subheader("Suggested Rules")
        suggested_rules = [
            f"Adjust weight for {c['original']}→{c['correction']}" for c in corrections
        ]
        for rule in suggested_rules:
            if st.checkbox(rule):
                st.write(f"Selected to apply: {rule}")

        if st.button(f"Commit Feedback for {speaker_tab}"):
            for corr in corrections:
                update_emotion_rules(corr, rules_path)
            st.success("Feedback committed!")

    return {'learned_rules': rules_path}
