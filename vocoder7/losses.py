# vocoder/losses.py

import torch
import torch.nn.functional as F
from .stft import STFTLoss
from .config import GANConfig

def compute_gan_loss(
    mpd_outputs, mpd_features,
    msd_outputs, msd_features,
    mbd_outputs, mbd_features,
    wav_fake, wav_real,
    predicted_pitch, true_pitch,
    predicted_duration, true_duration,
    cfg: GANConfig = None,
    step: int = 0
) -> (torch.Tensor, torch.Tensor):
    """
    Compute generator and discriminator losses:
      - Hinge adversarial for MPD, MSD, MBD
      - R1 regularization on real wav (to be implemented in trainer)
      - Feature-matching (L1) for each discriminator’s features
      - Multi-resolution STFT loss via STFTLoss
      - Pitch & duration L1 consistency
      - Adaptive adversarial weight scheduling
    Returns:
        g_loss, d_loss
    """
    # 1) Adversarial hinge loss
    all_fake = mpd_outputs + msd_outputs + mbd_outputs
    # Real outputs should be passed separately in trainer
    all_real = []  # placeholder
    d_loss, g_adv = 0.0, 0.0
    for fake, real in zip(all_fake, all_real):
        g_adv += -fake.mean()
        d_loss += F.relu(1 - real).mean() + F.relu(1 + fake).mean()
    adv_weight = min(1.0, step / (cfg.r1_interval * 10))
    g_adv = (adv_weight * g_adv) / len(all_fake)
    d_loss = d_loss / len(all_fake)

    # 2) Feature‑matching loss
    fm_loss = 0.0
    for feats_f, feats_r in zip(mpd_features, mpd_features):
        for f, r in zip(feats_f, feats_r):
            fm_loss += F.l1_loss(f, r)
    for feats_f, feats_r in zip(msd_features, msd_features):
        for f, r in zip(feats_f, feats_r):
            fm_loss += F.l1_loss(f, r)
    for feats_f, feats_r in zip(mbd_features, mbd_features):
        for f, r in zip(feats_f, feats_r):
            fm_loss += F.l1_loss(f, r)
    fm_loss /= (len(mpd_features) + len(msd_features) + len(mbd_features))

    # 3) STFT loss
    stft_loss = STFTLoss(cfg)(wav_fake, wav_real)

    # 4) Prosody consistency losses
    p_loss = F.l1_loss(predicted_pitch, true_pitch) * cfg.lambda_pitch
    d_loss_p = F.l1_loss(predicted_duration, true_duration) * cfg.lambda_dur

    # Total generator loss
    g_loss = g_adv + fm_loss + stft_loss + p_loss + d_loss_p

    return g_loss, d_loss
