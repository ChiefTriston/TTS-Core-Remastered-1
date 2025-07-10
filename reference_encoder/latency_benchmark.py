import time
import torch
from .config import RefEncConfig
from .encoder import ReferenceEncoder

if __name__ == '__main__':
    cfg = RefEncConfig()
    model = ReferenceEncoder(cfg).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # random mel input: (1, T_frames, n_mels)
    T = int(2 * cfg.sample_rate / cfg.hop_length)
    mel = torch.randn(1, T, cfg.n_mels).to(device)
    # warm-up
    for _ in range(10):
        _ = model(mel)
    # benchmark
    runs = 100
    start = time.time()
    for _ in range(runs):
        _ = model(mel)
    elapsed = (time.time() - start) / runs * 1000
    print(f"Avg latency: {elapsed:.2f} ms")