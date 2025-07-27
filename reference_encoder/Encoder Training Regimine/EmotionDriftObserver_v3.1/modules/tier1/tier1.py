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
    config = context['config']['tier1']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    analyzer = SentimentIntensityAnalyzer()
    
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        
        tags = []
        compounds = []
        for slice_data in transcript['slices']:
            text = slice_data['text'].lower()
            vader_scores = analyzer.polarity_scores(text)
            compound = vader_scores['compound']
            tag = 'pos' if compound > config['compound_pos'] else 'neg' if compound < config['compound_neg'] else 'neu'
            source = 'vader'
            
            if abs(compound) < config['confidence_thresh']:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                tag = 'pos' if polarity > 0 else 'neg' if polarity < 0 else 'neu'
                source = 'textblob'
            
            # Keyword-based fallback if still neutral or low conf
            if tag == 'neu' or abs(compound) < config['confidence_thresh'] / 2:
                if any(k in text for k in pos_keywords):
                    tag = 'pos'
                    source = 'keyword_pos'
                elif any(k in text for k in neg_keywords):
                    tag = 'neg'
                    source = 'keyword_neg'
            
            tags.append({'tag': tag, 'tag_source': source, 'compound': compound})
            compounds.append(compound)
        
        # Histogram rebalance: redistribute neutrals
        hist, bins = np.histogram(compounds, bins=3, range=(-1,1))
        neutral_idx = 1
        if hist[neutral_idx] > sum(hist) * 0.33:  # Avoid class collapse
            excess = hist[neutral_idx] - sum(hist) * 0.33
            neu_indices = [i for i in range(len(tags)) if tags[i]['tag'] == 'neu']
            np.random.shuffle(neu_indices)
            shift_to_pos = neu_indices[:int(excess / 2)]
            shift_to_neg = neu_indices[int(excess / 2):int(excess)]
            for i in shift_to_pos:
                tags[i]['tag'] = 'pos'
                tags[i]['compound'] += 0.1  # Slight shift
            for i in shift_to_neg:
                tags[i]['tag'] = 'neg'
                tags[i]['compound'] -= 0.1
        
        json_path = os.path.join(speaker_out, 'tier1_tags.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tags, f)
            portalocker.unlock(f)
    
    return {'tier1_tags': json_path}