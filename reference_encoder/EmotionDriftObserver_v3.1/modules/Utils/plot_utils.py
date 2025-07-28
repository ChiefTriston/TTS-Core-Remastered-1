# modules/utils/plot_utils.py

import os
import matplotlib.pyplot as plt

def generate_segment_plot_map(
    full_scores,
    segment_start,
    segment_end,
    clip_id,
    tier1_transition,
    drift_reason,
    save_dir
):
    """
    full_scores: list of {"time": float, "vader_compound": float}
    """
    os.makedirs(save_dir, exist_ok=True)

    times = [pt['time'] for pt in full_scores]
    comps = [pt['vader_compound'] for pt in full_scores]

    plt.figure(figsize=(12, 4))
    plt.plot(times, comps, label="VADER Compound")
    plt.axvspan(segment_start, segment_end, color='red', alpha=0.3, label="Current Segment")
    plt.title(f"{clip_id} | {tier1_transition} | {drift_reason}")
    plt.xlabel("Time (s)")
    plt.ylabel("Compound Score")
    plt.legend()
    out_path = os.path.join(save_dir, f"{clip_id}.png")
    plt.savefig(out_path)
    plt.close()

    return out_path
