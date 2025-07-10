
#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
from .config import AcousticConfig
from .model import AcousticModel

def run_ci_benchmarks(cfg: AcousticConfig):
    """
    Microbenchmarks the AcousticModel against a simple MLP baseline.
    Raises if model latency exceeds cfg.ci_latency_factor × baseline.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T = 2, 128

    # Random inputs
    text = torch.randn(B, T, cfg.text_emb_dim, device=device)
    f0, eng = [torch.randn(B, T, device=device) for _ in range(2)]
    spk = torch.randn(B, cfg.speaker_dim, device=device) if cfg.speaker_dim > 0 else None

    # Instantiate model
    model = AcousticModel(cfg).to(device)

    # Build a comparable multilayer MLP baseline
    layers = []
    dim_in = cfg.text_emb_dim + cfg.cond_dim + cfg.speaker_dim
    for _ in range(cfg.num_layers):
        layers += [nn.Linear(dim_in, cfg.hidden_channels), nn.ReLU()]
        dim_in = cfg.hidden_channels
    layers.append(nn.Linear(cfg.hidden_channels, 80))
    mlp = nn.Sequential(*layers).to(device)

    # Warm up GPU / caches
    _ = model(text, f0, eng, spk)

    # Time the AcousticModel
    t0 = time.time()
    for _ in range(20):
        _ = model(text, f0, eng, spk)
    t_model = (time.time() - t0) / 20

    # Reconstruct exactly the same concatenated features for the MLP baseline
    parts = [
        text.transpose(1, 2),   # [B, emb_dim, T]
        f0.unsqueeze(1),        # [B, 1, T]
        eng.unsqueeze(1)        # [B, 1, T]
    ]
    if cfg.speaker_dim > 0 and spk is not None:
        spk_t = spk.unsqueeze(2).expand(-1, -1, T)  # [B, speaker_dim, T]
        parts.append(spk_t)

    feats = torch.cat(parts, dim=1)                   # [B, emb+cond+spk, T]
    flat = feats.transpose(1, 2).reshape(B * T, -1)   # [B*T, emb+cond+spk]

    # Time the MLP baseline
    t1 = time.time()
    for _ in range(20):
        _ = mlp(flat)
    t_base = (time.time() - t1) / 20

    print(f"[CI] AcousticModel: {t_model*1000:.2f} ms | Baseline MLP: {t_base*1000:.2f} ms")
    if t_model > cfg.ci_latency_factor * t_base:
        raise RuntimeError(
            f"CI Failure: model {t_model:.4f}s exceeds "
            f"{cfg.ci_latency_factor}× baseline {t_base:.4f}s"
        )

if __name__ == "__main__":
    # Example config for local CI runs
    cfg = AcousticConfig(text_emb_dim=128, speaker_dim=16, profile=True)
    run_ci_benchmarks(cfg)
    print("✔ CI benchmarks passed.")
