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
import torch
import numpy as np

nlp_spacy = spacy.load("en_core_web_sm")
nlp_stanza = stanza.Pipeline('en')

class EmotionClassifier(torch.nn.Module):
    def __init__(self, input_dim=23, num_emotions=6, hidden_dim=128):
        super().__init__()
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dropout=0.1),
            num_layers=2
        )
        self.fc = torch.nn.Linear(input_dim, num_emotions)
    
    def forward(self, emotion_vector):
        trans_out = self.transformer(emotion_vector.unsqueeze(1)).squeeze(1)
        return torch.sigmoid(self.fc(trans_out))

class EmotionWeightLearner(torch.nn.Module):
    def __init__(self, vader_dim=4, prosody_dim=19):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(vader_dim + prosody_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, vader, prosody):
        return self.mlp(torch.cat([vader, prosody], dim=-1))

emotion_names = ['joy', 'sadness', 'anger', 'neutral', 'surprise', 'fear']

def run(context):
    config = context['config']['tier2']
    global_config = context['config']['global']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    device = 'cuda' if global_config['use_gpu'] and torch.cuda.is_available() else 'cpu'
    
    classifier = EmotionClassifier().to(device)
    classifier.eval()  # Assume loaded weights
    
    weight_learner = EmotionWeightLearner().to(device)
    weight_learner.eval()
    
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
            vader_scores = torch.tensor([tier1[idx]['compound'], tier1[idx].get('pos', 0.0), tier1[idx].get('neu', 0.0), tier1[idx].get('neg', 0.0)], dtype=torch.float32).to(device)
            
            # Stub prosody for slice
            prosody_features = torch.tensor(np.random.rand(19), dtype=torch.float32).to(device)  # Replace with actual
            
            w = weight_learner(vader_scores, prosody_features)
            emotion_vector = w * vader_scores + (1 - w) * prosody_features
            
            probs = classifier(emotion_vector.unsqueeze(0)).squeeze(0)
            
            primary_idx, secondary_idx = torch.topk(probs, k=2)[1]
            label = emotion_names[primary_idx.item()]
            confidence = probs[primary_idx.item()].item()
            
            doc_spacy = nlp_spacy(text)
            doc_stanza = nlp_stanza(text)
            rule_id = 'dl_model'
            if any(token.dep_ == 'neg' for token in doc_spacy):
                rule_id = 'negation_invert'
                if label == 'joy':
                    label = 'sadness'
                elif label == 'surprise':
                    label = 'fear'
                confidence *= config['negation_weight']
            # Inversion handling with Stanza (example: sentiment contradiction)
            sentiments = [sent.sentiment for sent in doc_stanza.sentences]
            if len(sentiments) > 1 and sentiments[0] != sentiments[1]:
                rule_id += '_inversion'
                label = 'neutral'  # Example
            
            tier2_tags.append({'label': label, 'confidence': confidence, 'rule_id': rule_id, 'confidences': probs.tolist()})
        
        json_path = os.path.join(speaker_out, 'tier2_tags.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tier2_tags, f)
            portalocker.unlock(f)
    
    return {'tier2_tags': json_path}    return {'tier2_tags': json_path}