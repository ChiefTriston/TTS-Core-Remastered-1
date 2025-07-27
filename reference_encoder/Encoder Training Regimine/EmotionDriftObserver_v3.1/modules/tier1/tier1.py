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
            text = slice_data['text']
            vader_scores = analyzer.polarity_scores(text)
            compound = vader_scores['compound']
            tag = 'pos' if compound > config['compound_pos'] else 'neg' if compound < config['compound_neg'] else 'neu'
            source = 'vader'
            
            if abs(compound) < config['confidence_thresh']:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                tag = 'pos' if polarity > 0 else 'neg' if polarity < 0 else 'neu'
                source = 'textblob'
            
            tags.append({'tag': tag, 'tag_source': source, 'compound': compound})
            compounds.append(compound)
        
        # Histogram rebalance
        hist, bins = np.histogram(compounds, bins=3, range=(-1,1))
        neutral_idx = 1
        if hist[neutral_idx] > sum(hist) * 0.5:  # If neutrals dominate
            excess = hist[neutral_idx] - sum(hist) * 0.33
            shift = excess / 2 / hist[neutral_idx]  # Shift to pos/neg
            for i in range(len(compounds)):
                if bins[neutral_idx] < compounds[i] < bins[neutral_idx+1]:
                    if np.random.rand() < 0.5:
                        compounds[i] += shift
                    else:
                        compounds[i] -= shift
                tag = 'pos' if compounds[i] > config['compound_pos'] else 'neg' if compounds[i] < config['compound_neg'] else 'neu'
                tags[i]['tag'] = tag
                tags[i]['compound'] = compounds[i]
        
        json_path = os.path.join(speaker_out, 'tier1_tags.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tags, f)
            portalocker.unlock(f)
    
    return {'tier1_tags': json_path}