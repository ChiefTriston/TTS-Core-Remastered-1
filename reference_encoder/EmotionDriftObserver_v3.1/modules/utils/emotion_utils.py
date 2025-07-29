# emotion_utils.py

# ─── Emotion‐rule definitions ─────────────────────────────────────────────────
# Each rule returns True/False and has an initial confidence score.
emotion_rules = {
    "Happy":    (lambda x: x["pos"] > 0.8   and x["pitch_mean"] > 150, 0.5),
    "Sad":      (lambda x: x["neg"] > 0.8   and x["pitch_mean"] < 120, 0.5),
    "Angry":    (lambda x: x["neg"] > 0.7   and x["pitch_mean"] > 180, 0.5),
    "Neutral":  (lambda x: x["neu"] > 0.7   and x["pitch_std"]  <  20, 0.5),
    "Surprise": (lambda x: x["pos"] > 0.7   and x["pitch_std"]  >  50, 0.5)
}

# ─── Emotion‐group mapping ───────────────────────────────────────────────────
# Used downstream to route auto‐accepted JSON into Tier1 folders.
GROUP_MAP = {
    "Happy":    "Positive",
    "Sad":      "Negative",
    "Angry":    "Negative",
    "Neutral":  "Neutral",
    "Surprise": "Positive"
}

# ─── Tier thresholds ─────────────────────────────────────────────────────────
# These mirror T1_AUTO, T1_MIN, T2_AUTO, T2_MIN, SENTIMENT_STD_THRESHOLD
T1_AUTO = 0.90    # Tier‑1 auto‐accept
T1_MIN  = 0.80    # Tier‑1 minimum pass

T2_AUTO = 0.90    # Tier‑2 auto‐accept
T2_MIN  = 0.65    # Tier‑2 review threshold

# If sentiment_std above this, even auto‐accepted segments get flagged for review
SENTIMENT_STD_THRESHOLD = 0.30
