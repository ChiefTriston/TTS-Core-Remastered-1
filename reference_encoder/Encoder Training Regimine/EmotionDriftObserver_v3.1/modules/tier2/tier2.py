import json
import spacy
import stanza
import portalocker
import os
import numpy as np
import torch
import torchaudio
from modules.utils.emotion_utils import (
    emotion_rules, GROUP_MAP,
    T1_AUTO, T1_MIN,
    T2_AUTO, T2_MIN,
    SENTIMENT_STD_THRESHOLD
)
from sklearn.metrics.pairwise import cosine_similarity

nlp_spacy = spacy.load("en_core_web_sm")
nlp_stanza = stanza.Pipeline('en')

# Heuristic rule engine
def apply_rules(text, base_tag, conf, prosody_score):
    rule_id = 'base'
    # Prosody + sentiment interplay
    if base_tag == 'positive' and prosody_score > 1.0:  # High prosody -> excited/happy
        label = 'excited'
        rule_id = 'high_prosody_pos'
    elif base_tag == 'negative' and prosody_score > 1.0:  # High -> angry
        label = 'angry'
        rule_id = 'high_prosody_neg'
    elif base_tag == 'positive' and prosody_score < -1.0:  # Low -> calm pleasant
        label = 'pleasant'
        rule_id = 'low_prosody_pos'
    elif base_tag == 'negative' and prosody_score < -1.0:  # Low -> sad
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
    neg_w    = cfg['negation_weight']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']
    
    device = 'cuda' if context['config']['global']['use_gpu'] and torch.cuda.is_available() else 'cpu'
    sr = context['config']['global']['sample_rate']
    
    model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder')
    model.to(device)
    model.eval()
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
    else:
        resampler = None
    
    emb_cache = {}
    
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
        with open(os.path.join(speaker_out, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)
        
        wav_path = os.path.join(speaker_out, f'{speaker_id}.wav')
        waveform, _ = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        time = np.array(prosody['frame_series']['time'])
        f0_z = np.array(prosody['frame_series']['f0_z'])
        energy_z = np.array(prosody['frame_series']['energy_z'])
        prosody_combined = (f0_z + energy_z) / 2  # Simple combine for score
        
        compounds = [t['compound'] for t in tier1]
        sentiment_amp = np.ptp(compounds) if compounds else 0.0
        sentiment_std = np.std(compounds) if compounds else 0.0
        drift_score = np.mean(np.abs(drift['deltas'])) if drift['deltas'] else 0.0
        
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
                if base_tag == 'positive':
                    base_tag = 'negative'
                elif base_tag == 'negative':
                    base_tag = 'positive'
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
            
            # ─── embed & update speaker cache ────────────────────────────────
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            waveform_mono = waveform[:, start_sample:end_sample]
            if resampler is not None:
                waveform_mono = resampler(waveform_mono)
            waveform_mono = waveform_mono.to(device)
            try:
                with torch.no_grad():
                    emb = model(waveform_mono).cpu().numpy().squeeze(0)
            except:
                emb = np.zeros(256, dtype=np.float32)
    
            # update running-mean embedding
            stats = emb_cache.setdefault(speaker_id, {"mean": None, "n": 0})
            if stats["mean"] is None:
                stats["mean"], stats["n"] = emb, 1
            else:
                n = stats["n"]
                stats["mean"] = (stats["mean"] * n + emb) / (n + 1)
                stats["n"] += 1
    
            # cosine-similarity ESR score
            avg_emb = stats["mean"]
            cos_sim = float(cosine_similarity([emb], [avg_emb])[0][0]) if stats["n"] > 1 else 0.0
            esr_score = max(conf, cos_sim)
    
            # ─── compute Tier-2 confidence, incorporate drift+sentiment_amp ──
            tier2_conf = conf * (1
                                 + min(0.3, drift_score)
                                 + min(0.2, sentiment_amp))
    
            # ─── original pass/review/auto-reject logic ────────────────────────
            status = "auto-reject"
            if tier2_conf >= T2_AUTO:
                status = "auto-accepted"
            elif tier2_conf >= T2_MIN:
                status = "needs-review"
    
            # ─── FORCE REVIEW if sentiment-map noisy ─────────────────────────
            if status == "auto-accepted" and sentiment_std > SENTIMENT_STD_THRESHOLD:
                status = "needs-review"
            
            tier2_tags.append({'label': label, 'confidence': tier2_conf, 'rule_id': rule_id, 'status': status})
        
        json_path = os.path.join(speaker_out, 'tier2_tags.json')
        with open(json_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tier2_tags, f)
            portalocker.unlock(f)
    
    return {'tier2_tags': json_path}