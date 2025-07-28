# modules/tier2/tier2.py
"""
Tier 2 emotion refinement with NLP, heuristics, ESR, and dynamic thresholds.
Outputs tier2_tags.json per speaker.
"""

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
    T2_AUTO, T2_MIN,
    SENTIMENT_STD_THRESHOLD
)
from sklearn.metrics.pairwise import cosine_similarity

nlp_spacy  = spacy.load("en_core_web_sm")
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

def run(context):
    cfg       = context['config']['tier2']
    gcfg      = context['config']['global']
    neg_w     = cfg['negation_weight']
    sr        = gcfg['sample_rate']
    use_gpu   = gcfg['use_gpu'] and torch.cuda.is_available()
    device    = 'cuda' if use_gpu else 'cpu'

    # speaker embedder
    model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder')
    model.to(device).eval()
    resampler = torchaudio.transforms.Resample(sr, 16000) if sr != 16000 else None

    results = {}
    for sp in context['speaker_ids']:
        spk_dir = os.path.join(context['output_dir'], 'emotion_tags', sp)
        # load dependencies
        with open(os.path.join(spk_dir, 'tier1_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH); tier1 = json.load(f); portalocker.unlock(f)
        with open(os.path.join(spk_dir, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH); transcript = json.load(f); portalocker.unlock(f)
        with open(os.path.join(spk_dir, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH); prosody = json.load(f); portalocker.unlock(f)
        with open(os.path.join(spk_dir, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH); drift = json.load(f); portalocker.unlock(f)

        # load audio
        wav_path = os.path.join(spk_dir, f"{sp}.wav")
        waveform,_ = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0,keepdim=True)
        time      = np.array(prosody['frame_series']['time'])
        f0_z      = np.array(prosody['frame_series']['f0_z'])
        energy_z  = np.array(prosody['frame_series']['energy_z'])
        pros_comb = (f0_z + energy_z)/2

        compounds = [abs(t['compound']) for t in tier1]
        sent_amp  = np.ptp(compounds) if compounds else 0.0
        sent_std  = np.std(compounds) if compounds else 0.0
        drift_score = np.mean(np.abs(drift['deltas'])) if drift['deltas'] else 0.0

        emb_cache = {}
        tier2_tags = []
        for idx, seg in enumerate(transcript['slices']):
            text     = seg['text']
            base_tag = tier1[idx]['tag']
            conf     = abs(tier1[idx]['compound'])
            rule_id  = 'base'

            # negation invert
            doc_sp = nlp_spacy(text)
            if any(tok.dep_=='neg' for tok in doc_sp):
                rule_id = 'negation_invert'
                base_tag = 'negative' if base_tag=='positive' else 'positive'
                conf *= neg_w

            # contradiction
            doc_st = nlp_stanza(text)
            for sent in doc_st.sentences:
                words = [w.text.lower() for w in sent.words]
                if 'should' in words and 'happy' in words:
                    rule_id = 'contradiction'
                    base_tag = 'despair'
                    conf *= 0.8

            # prosody slice score
            s_time, e_time = seg['start'], seg['end']
            si = np.searchsorted(time, s_time)
            ei = np.searchsorted(time, e_time)
            pros_score = float(np.mean(pros_comb[si:ei])) if ei>si else 0.0

            # heuristic rules
            for emo,(fn,_) in emotion_rules.items():
                if fn({'pos':conf,'neg':1-conf,'pitch_mean':pros_score}):
                    label = emo
                    rule_id = f"rule_{emo}"
                    break
            else:
                label = base_tag

            # speaker embedding & ESR
            start_s = int(s_time*sr); end_s = int(e_time*sr)
            slice_wave = waveform[:,start_s:end_s]
            if resampler: slice_wave = resampler(slice_wave)
            slice_wave = slice_wave.to(device)
            with torch.no_grad():
                emb = model(slice_wave).cpu().numpy().squeeze(0)
            stats = emb_cache.setdefault(sp,{"mean":None,"n":0})
            if stats["mean"] is None:
                stats["mean"],stats["n"] = emb,1
            else:
                stats["mean"] = (stats["mean"]*stats["n"] + emb)/(stats["n"]+1)
                stats["n"] += 1
            cos_sim = float(cosine_similarity([emb],[stats["mean"]])[0][0]) if stats["n"]>1 else 0.0
            esr_score = max(conf, cos_sim)

            # Tier‑2 confidence
            t2_conf = conf*(1 + min(0.3, drift_score) + min(0.2, sent_amp))

            # decide status
            if t2_conf >= T2_AUTO:
                status = "auto-accepted"
            elif t2_conf >= T2_MIN:
                status = "needs-review"
            else:
                status = "auto-reject"
            if status=="auto-accepted" and sent_std > SENTIMENT_STD_THRESHOLD:
                status = "needs-review"

            tier2_tags.append({
                'label':      label,
                'confidence': t2_conf,
                'rule_id':    rule_id,
                'esr_score':  esr_score,
                'status':     status
            })

        out_path = os.path.join(spk_dir, 'tier2_tags.json')
        with open(out_path,'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tier2_tags, f, indent=2)
            portalocker.unlock(f)

        results[sp] = out_path

    return {'tier2_tags': results}
