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

def classify_tier1(text, vader_scores, config):
    compound = vader_scores['compound']
    vs = {'pos': vader_scores['pos'], 'neu': vader_scores['neu'], 'neg': vader_scores['neg']}
    vs_conf = max(vs.values())
    if vs_conf >= config['min_conf']:
        tag = max(vs, key=vs.get)
        source = 'vader'
        if vs_conf >= config['auto_accept_conf']:
            source = 'vader_auto'
    else:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        tag = 'pos' if polarity > 0 else 'neg' if polarity < 0 else 'neu'
        source = 'textblob'
    
    text_lower = text.lower()
    if tag == 'neu' or abs(compound) < config['confidence_thresh'] / 2:
        if any(k in text_lower for k in pos_keywords):
            tag = 'pos'
            source = 'keyword_pos'
        elif any(k in text_lower for k in neg_keywords):
            tag = 'neg'
            source = 'keyword_neg'
    
    return {'tag': tag, 'tag_source': source, 'compound': compound}

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
            tag_data = classify_tier1(text, vader_scores, config)
            tags.append(tag_data)
            compounds.append(tag_data['compound'])
        
        # Histogram rebalance after you’ve built your tags list
        neut = sum(1 for t in tags if t['tag']=='neu')
        if neut/len(tags) > 0.5:
            for t in tags:
                if t['tag']=='neu':
                    t['fallback'] = True
        
        json_path = os.path.join(speaker_out, 'tier1_tags.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tags, f)
            portalocker.unlock(f)
    
    return {'tier1_tags': json_path}