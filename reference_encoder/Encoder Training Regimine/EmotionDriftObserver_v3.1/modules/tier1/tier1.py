# modules/tier1/tier1.py
"""
Tier 1 emotion tagging using VADER with TextBlob fallback.
Outputs tier1_tags.json.
"""

import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import portalocker
import os

# Keyword-based fallback
pos_keywords = ['love', 'happy', 'joy', 'excellent', 'great']
neg_keywords = ['hate', 'sad', 'anger', 'terrible', 'bad']

def run(context):
    cfg     = context['config']['tier1']
    t1_auto = cfg['auto_accept_conf']
    t1_min  = cfg['min_conf']
    pos_th  = cfg['compound_pos']
    neg_th  = cfg['compound_neg']
    low_conf = cfg['confidence_thresh']
    
    analyzer = SentimentIntensityAnalyzer()
    
    for speaker_id in context['speaker_ids']:
        speaker_out = os.path.join(context['output_dir'], 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        
        tags = []
        compounds = []
        for slice_data in transcript['slices']:
            text = slice_data['text']
            vs   = analyzer.polarity_scores(text)
            compound = vs['compound']
         
            # Tier1 bucket
            if compound >= t1_min:
                tag = 'positive' if compound > pos_th else 'negative' if compound < neg_th else 'neutral'
            else:
                tag = 'neutral'
         
            # auto‑accept if very confident
            auto_accept = abs(compound) >= t1_auto
            source = 'vader' if auto_accept else 'vader_low'
         
            # fallback if too close to neutral
            if abs(compound) < low_conf:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                tag = 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'
                source = 'textblob'
            
            # Keyword-based fallback if still neutral or low conf
            text_lower = text.lower()
            if tag == 'neutral':
                if any(k in text_lower for k in pos_keywords):
                    tag = 'positive'
                    source = 'keyword_pos'
                elif any(k in text_lower for k in neg_keywords):
                    tag = 'negative'
                    source = 'keyword_neg'
            
            tags.append({'tag': tag, 'tag_source': source, 'compound': compound, 'auto_accept': auto_accept})
            compounds.append(compound)
        
        # Histogram rebalance: redistribute neutrals
        hist, bins = np.histogram(compounds, bins=3, range=(-1,1))
        neutral_idx = 1
        if hist[neutral_idx] > sum(hist) * 0.33:  # Avoid class collapse
            excess = hist[neutral_idx] - sum(hist) * 0.33
            neu_indices = [i for i in range(len(tags)) if tags[i]['tag'] == 'neutral']
            np.random.shuffle(neu_indices)
            shift_to_pos = neu_indices[:int(excess / 2)]
            shift_to_neg = neu_indices[int(excess / 2):int(excess)]
            for i in shift_to_pos:
                tags[i]['tag'] = 'positive'
                tags[i]['compound'] += 0.1  # Slight shift
            for i in shift_to_neg:
                tags[i]['tag'] = 'negative'
                tags[i]['compound'] -= 0.1
        
        json_path = os.path.join(speaker_out, 'tier1_tags.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tags, f)
            portalocker.unlock(f)
    
    return {'tier1_tags': json_path}