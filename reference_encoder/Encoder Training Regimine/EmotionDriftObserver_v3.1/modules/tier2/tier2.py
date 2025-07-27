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

nlp_spacy = spacy.load("en_core_web_sm")
nlp_stanza = stanza.Pipeline('en')

def run(context):
    config = context['config']['tier2']
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
        
        tier2_tags = []
        for idx, slice_data in enumerate(transcript['slices']):
            text = slice_data['text']
            base_tag = tier1[idx]['tag']
            conf = tier1[idx]['compound']
            
            doc_spacy = nlp_spacy(text)
            doc_stanza = nlp_stanza(text)
            rule_id = 'base'
            
            # Negation inversion
            if any(token.dep_ == 'neg' for token in doc_spacy):
                rule_id = 'negation_invert'
                base_tag = 'neg' if base_tag == 'pos' else 'pos' if base_tag == 'neg' else base_tag
                conf *= config['negation_weight']
            
            # Contradiction rules (e.g. "should be happy")
            for sent in doc_stanza.sentences:
                words = [word.text.lower() for word in sent.words]
                if 'should' in words and 'happy' in words:
                    base_tag = 'despair'
                    rule_id = 'contradiction_should_happy'
            
            # Keyword-based overrides
            keywords = {'anger': ['mad', 'furious'], 'fear': ['scared', 'terrified']}
            for emotion, keys in keywords.items():
                if any(k in text.lower() for k in keys):
                    base_tag = emotion
                    rule_id = f'keyword_{emotion}'
                    break
            
            # Prosody-based sub-labels (stub: high energy -> anger if neg)
            # Assume prosody data available
            
            tier2_tags.append({'label': base_tag, 'confidence': abs(conf), 'rule_id': rule_id})
        
        json_path = os.path.join(speaker_out, 'tier2_tags.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tier2_tags, f)
            portalocker.unlock(f)
    
    return {'tier2_tags': json_path}