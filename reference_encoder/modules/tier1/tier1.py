# tier1.py (Full - No Change Needed)
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
    results = {}  # Dictionary to store speaker_id: json_path mappings
    
    for speaker_id in context['speaker_ids']:
        speaker_out = os.path.join(context['output_dir'], 'emotion_tags', speaker_id)
        transcript_path = os.path.join(speaker_out, 'transcript.json')
        json_path = os.path.join(speaker_out, 'tier1_tags.json')
        
        # Load transcript
        try:
            with open(transcript_path, 'r') as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                transcript = json.load(f)
                portalocker.unlock(f)
        except Exception as e:
            print(f"[Tier1] Failed to load transcript for {speaker_id}: {e}")
            results[speaker_id] = json_path  # Still return a path for consistency
            continue
        
        tags = []
        compounds = []
        for slice_data in transcript.get('slices', []):
            text = slice_data.get('text', '')
            if not text:
                continue
            vs   = analyzer.polarity_scores(text)
            compound = vs['compound']
         
            # Tier1 bucket
            if abs(compound) >= t1_min:
                tag = 'positive' if compound > pos_th else 'negative' if compound < neg_th else 'neutral'
            else:
                tag = 'neutral'
         
            # Decide status
            status = 'needs-review'
            if abs(compound) >= t1_auto:
                status = 'auto-accept'
            elif abs(compound) < t1_min:
                status = 'force-manual'
         
            source = 'vader' if abs(compound) >= low_conf else 'vader_low'
         
            # Fallback if too close to neutral
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
            
            tags.append({'tag': tag, 'tag_source': source, 'compound': compound, 'status': status})
            compounds.append(compound)
        
        # Histogram rebalance: redistribute neutrals
        tag_codes = [1 if t['tag'] == 'neutral' else 2 if t['tag'] == 'positive' else 0 for t in tags]
        counts = np.bincount(tag_codes, minlength=3)
        total = len(tags)
        neutral_idx = 1
        if counts[neutral_idx] > total * 0.5:  # Avoid class collapse
            excess = counts[neutral_idx] - int(total * 0.5)
            neu_indices = [i for i in range(len(tags)) if tags[i]['tag'] == 'neutral']
            np.random.shuffle(neu_indices)
            shift_to_pos = neu_indices[:excess // 2]
            shift_to_neg = neu_indices[excess // 2:excess]
            for i in shift_to_pos:
                tags[i]['tag'] = 'positive'
                tags[i]['compound'] += 0.1  # Slight shift
            for i in shift_to_neg:
                tags[i]['tag'] = 'negative'
                tags[i]['compound'] -= 0.1
        
        # Write tags (even if empty)
        os.makedirs(speaker_out, exist_ok=True)
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tags, f, indent=2)
            portalocker.unlock(f)
        
        results[speaker_id] = json_path
    
    return {'tier1_tags': results}

