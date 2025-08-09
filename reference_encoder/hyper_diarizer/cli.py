# -*- coding: utf-8 -*-
"""
Orchestrator for HyperDiarizer modules with CLI, logging, and feedback loops.
"""
import os
import json
import argparse
import torchaudio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm
import time
import torch
import numpy as np
from multiprocessing import Pool, cpu_count

from .slicer import dynamic_slice
from .embedding import extract_emb
from .cluster import time_aware_sim, temporal_cluster
from .reid import ReIDMemory as ReidMemory
from .audio_rebuilder import reconstruct_audio
from .overlap import detect_overlaps

import logging
from logging.handlers import RotatingFileHandler
from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity
import yaml

SAMPLE_RATE = 16000

# Configure logging
logging.basicConfig(
    handlers=[RotatingFileHandler('diarizer.log', maxBytes=10**6, backupCount=5)],
    level=logging.INFO
)

# Load config
dirpath = os.path.dirname(__file__)
config_file = os.path.join(dirpath, 'HyperDiarizer config.yaml')
with open(config_file, 'r') as f:
    cfg = yaml.safe_load(f)['hyperdiazer']
MIN_SLICE_DUR = cfg.get('min_slice_dur', 1.5)


def visualize_results(sim, labels, output_dir, slices):
    plt.imshow(sim, cmap='viridis')
    plt.colorbar()
    plt.title('Similarity Matrix')
    plt.savefig(os.path.join(output_dir, 'sim_matrix.png'))

    fig = go.Figure()
    for s, e, _ in slices:
        fig.add_vrect(x0=s, x1=e, fillcolor="blue", opacity=0.2)
    fig.write_html(os.path.join(output_dir, 'timeline.html'))


class DiarizerController:
    def __init__(self, args):
        self.args = args
        self.reid = ReidMemory(thresh=args.voiceprint_thresh)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buffer = []  # For streaming

    def process_chunk(self, audio_chunk):
        slices, stats = dynamic_slice(audio_chunk, self.device)
        embs = extract_emb(audio_chunk, slices, stats['noise_amp'])
        sim = time_aware_sim(embs, slices)
        labels = temporal_cluster(sim, slices)
        labels, _, certainties = self.reid.re_id(embs, labels)
        return labels, certainties, slices, embs, stats

    def diarize_single(self, audio_path, output_dir, visualize=False, streaming=False):
        try:
            waveform, orig_sr = torchaudio.load(audio_path)
            if orig_sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)(waveform)
            audio = waveform.squeeze().numpy()
            audio_dur = len(audio) / SAMPLE_RATE

            # Chunked vs single-pass
            if audio_dur > 3600 or streaming:
                logging.info(f"Long audio detected ({audio_dur}s); processing in chunks.")
                chunk_sec = 60
                chunk_len = SAMPLE_RATE * chunk_sec
                all_slices, all_labels, all_embs, all_overlaps, all_certs = [], [], [], [], []
                all_stats = {}

                for i in tqdm.tqdm(range(0, len(audio), chunk_len)):
                    chunk = audio[i: i+chunk_len]
                    labels, certs, slices, embs, stats = self.process_chunk(chunk)
                    offset = i / SAMPLE_RATE
                    slices = [(s+offset, e+offset, p) for s, e, p in slices]
                    overlaps = detect_overlaps(chunk, slices, labels, embs)
                    overlaps = [(gs+offset, ge+offset, l1, l2, conf) for gs, ge, l1, l2, conf in overlaps]

                    all_slices.extend(slices)
                    all_labels.extend(labels)
                    all_embs.extend(embs)
                    all_overlaps.extend(overlaps)
                    all_certs.extend(certs)
                    all_stats.update(stats)

                slices = all_slices
                labels = np.array(all_labels)
                embs = np.array(all_embs)
                overlaps = all_overlaps
                certainties = np.array(all_certs)
                stats = all_stats
            else:
                times = {}
                t0 = time.time()
                slices, stats = dynamic_slice(audio, self.device)
                times['slicing'] = time.time() - t0

                t0 = time.time()
                embs = extract_emb(audio, slices, stats['noise_amp'])
                times['embedding'] = time.time() - t0

                t0 = time.time()
                sim = time_aware_sim(embs, slices)
                times['sim'] = time.time() - t0

                t0 = time.time()
                labels = temporal_cluster(sim, slices)
                times['clustering'] = time.time() - t0

                t0 = time.time()
                labels, label_map, certainties = self.reid.re_id(
                    embs, labels, use_transformer=not self.args.hmm_off)
                times['reid'] = time.time() - t0

                if certainties.mean() < 0.7:
                    logging.warning("Low certainty; reslicing with adjusted params")
                    slices, stats = dynamic_slice(
                        audio, self.device,
                        min_slice_dur=MIN_SLICE_DUR * 0.8)

                t0 = time.time()
                overlaps = detect_overlaps(audio, slices, labels, embs)
                times['overlap'] = time.time() - t0
                sim  # keep for outputs

            # Reconstruction
            t0 = time.time()
            speaker_paths = reconstruct_audio(
                audio_path, slices, labels, overlaps, output_dir, certainties)
            times['rebuild'] = time.time() - t0

            # Save results
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'speaker_mapping.json'), 'w') as f:
                json.dump(speaker_paths, f, indent=2)
            np.save(os.path.join(output_dir, 'similarity_matrix.npy'), sim)
            np.save(os.path.join(output_dir, 'certainties.npy'), certainties)
            with open(os.path.join(output_dir, 'diarization_log.json'), 'w') as f:
                json.dump({'overlaps': overlaps,
                           'label_map': locals().get('label_map', {})}, f, indent=2)
            with open(os.path.join(output_dir, 'step_times.json'), 'w') as f:
                json.dump(times, f, indent=2)
            with open(os.path.join(output_dir, 'slicer_stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)

            if visualize:
                visualize_results(sim, labels, output_dir, slices)

            return {'speaker_paths': speaker_paths}
        except Exception as e:
            logging.error(f"Diarization error: {e}", exc_info=True)
            return {}


def diarize_wrapper(tup):
    controller, audio_path, out_sub, vis = tup
    os.makedirs(out_sub, exist_ok=True)
    controller.diarize_single(audio_path, out_sub, visualize=vis)


def calculate_der(gt, hyp):
    try:
        ref = Annotation.from_rttm(gt)
        hyp_ann = Annotation.from_rttm(hyp)
        der = DiarizationErrorRate()(ref, hyp_ann)
        purity = DiarizationPurity()(ref, hyp_ann)
        return {'der': der, 'purity': purity}
    except Exception as e:
        logging.error(f"DER error: {e}")
        return {'der': 0.0, 'purity': 0.0}


def main():
    p = argparse.ArgumentParser(description="HyperDiarizer CLI")
    p.add_argument('input', help="Input audio file or directory")
    p.add_argument('output_dir', help="Output directory")
    p.add_argument('--visualize', action='store_true')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--eval', help="GT RTTM for DER eval")
    p.add_argument('--hmm_off', action='store_true')
    p.add_argument('--voiceprint_thresh', type=float, default=0.6)
    p.add_argument('--streaming', action='store_true')
    args = p.parse_args()

    controller = DiarizerController(args)
    if args.batch:
        files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                 if f.lower().endswith(('.wav', '.mp3'))]
        pool = Pool(min(cpu_count(), len(files)))
        tasks = [(controller, af, os.path.join(args.output_dir, os.path.splitext(af)[0]), args.visualize)
                 for af in files]
        pool.map(diarize_wrapper, tasks)
    else:
        controller.diarize_single(args.input, args.output_dir,
                                  visualize=args.visualize,
                                  streaming=args.streaming)

    if args.eval:
        hyp_file = os.path.join(args.output_dir, 'diarization.rttm')
        res = calculate_der(args.eval, hyp_file)
        print(f"DER: {res['der']}, Purity: {res['purity']}")

if __name__ == '__main__':
    main()
