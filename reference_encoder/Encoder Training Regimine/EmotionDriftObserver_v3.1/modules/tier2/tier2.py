# modules/tier2/tier2.py
"""
Tier 2 emotion refinement using heuristics, spaCy negation.
Outputs tier2_tags.json.
"""

import json
import spacy
import stanza
import portalocker
import os
import numpy as np

nlp_spacy = spacy.load("en_core_web_sm")
nlp_stanza = stanza.Pipeline('en')

# Heuristic rule engine
def apply_rules(text, base_tag, conf, prosody_score):
    rule_id = 'base'
    # Prosody + sentiment interplay
    if base_tag == 'pos' and prosody_score > 1.0:  # High prosody -> excited/happy
        label = 'excited'
        rule_id = 'high_prosody_pos'
    elif base_tag == 'neg' and prosody_score > 1.0:  # High -> angry
        label = 'angry'
        rule_id = 'high_prosody_neg'
    elif base_tag == 'pos' and prosody_score < -1.0:  # Low -> calm pleasant
        label = 'pleasant'
        rule_id = 'low_prosody_pos'
    elif base_tag == 'neg' and prosody_score < -1.0:  # Low -> sad
        label = 'sad'
        rule_id = 'low_prosody_neg'
    else:
        label = base_tag  # Default
    
    # Keyword rules for specific emotions
    happy_keywords = ['joy', 'happy', 'delighted']
    sad_keywords = ['sad', 'depressed', 'miserable']
    angry_keywords = ['angry', 'furious', 'mad']
    surprise_keywords = ['surprise', 'shocked', 'amazed']
    
    if any(k in text.lower() for k in happy_keywords):
        label = 'happy'
        rule_id = 'keyword_happy'
        conf += 0.1  # Boost conf
    elif any(k in text.lower() for k in sad_keywords):
        label = 'sad'
        rule_id = 'keyword_sad'
        conf += 0.1
    elif any(k in text.lower() for k in angry_keywords):
        label = 'angry'
        rule_id = 'keyword_angry'
        conf += 0.1
    elif any(k in text.lower() for k in surprise_keywords):
        label = 'surprise'
        rule_id = 'keyword_surprise'
        conf += 0.1
    
    return label, conf, rule_id

def run(context):
    cfg      = context['config']['tier2']
    t2_auto  = cfg['auto_accept_conf']
    t2_min   = cfg['min_conf']
    neg_w    = cfg['negation_weight']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'tier1_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier1 = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)
        
        time = np.array(prosody['frame_series']['time'])
        f0_z = np.array(prosody['frame_series']['f0_z'])
        energy_z = np.array(prosody['frame_series']['energy_z'])
        prosody_combined = (f0_z + energy_z) / 2  # Simple combine for score
        
        tier2_tags = []
        for idx, slice_data in enumerate(transcript['slices']):
            text = slice_data['text']
            base_tag = tier1[idx]['tag']
            conf = abs(tier1[idx]['compound'])
            
            doc_spacy = nlp_spacy(text)
            doc_stanza = nlp_stanza(text)
            
            # Negation inversion
            if any(token.dep_ == 'neg' for token in doc_spacy):
                rule_id = 'negation_invert'
                if base_tag == 'pos':
                    base_tag = 'neg'
                elif base_tag == 'neg':
                    base_tag = 'pos'
                conf *= neg_w
            else:
                rule_id = 'base'
            
            # Contradiction detection
            for sent in doc_stanza.sentences:
                words = [word.text.lower() for word in sent.words]
                if 'should' in words and 'happy' in words:
                    base_tag = 'despair'
                    rule_id = 'contradiction_should_happy'
                    conf *= 0.8  # Reduce conf for contradiction
            
            # Slice prosody score
            start_time = slice_data['start']
            end_time = slice_data['end']
            start_idx = np.searchsorted(time, start_time)
            end_idx = np.searchsorted(time, end_time)
            slice_prosody = np.mean(prosody_combined[start_idx:end_idx]) if end_idx > start_idx else 0.0
            
            # Apply rules
            label, conf, new_rule_id = apply_rules(text, base_tag, conf, slice_prosody)
            if new_rule_id != 'base':
                rule_id = new_rule_id
            
            # decide
            if conf >= t2_auto:
                status = 'auto-accepted'
            elif conf >= t2_min:
                status = 'needs-review'
            else:
                status = 'auto-reject'
            
            tier2_tags.append({'label': label, 'confidence': conf, 'rule_id': rule_id, 'status': status})
        
        json_path = os.path.join(speaker_out, 'tier2_tags.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tier2_tags, f)
            portalocker.unlock(f)
    
    return {'tier2_tags': json_path}