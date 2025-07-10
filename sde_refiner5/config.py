from dataclasses import dataclass, field
import json
from typing import List, Dict, Any


@dataclass
class ScoreSDEConfig:
    # architecture
    bands:            List[int]
    levels:           int
    flows:            int
    cond_dim:         int
    time_dim:         int
    beta_hidden:      int

    # ── new fields for HSF & VQ wiring ──
    hidden_channels:  int
    hsflayer_hidden:  int
    hsflayer_layers:  int
    hsflayer_kernel:  int

    # style & VQ
    style_dim:        int
    vq_dims:          List[int]
    vq_codes:         List[int]

    # CNF / misc
    cnf_dim:          int
    max_avg_time:     float
    benchmark_runs:   int
    profile:          bool

    # S4 hyper-params
    s4:               Dict[str, Any] = field(default_factory=dict)


def load_score_sde_config(path: str) -> ScoreSDEConfig:
    with open(path, 'r') as f:
        data: Any = json.load(f)

    required = [
        "bands", "levels", "flows", "style_dim", "cond_dim",
        "time_dim", "beta_hidden", "hidden_channels", "hsflayer_hidden",
        "hsflayer_layers", "hsflayer_kernel", "vq_dims", "vq_codes",
        "cnf_dim", "max_avg_time", "benchmark_runs", "profile", "s4"
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required config field: {key}")

    cfg = ScoreSDEConfig(**data)
    if sum(cfg.bands) != cfg.cnf_dim:
        raise ValueError(f"Sum of bands {sum(cfg.bands)} != cnf_dim {cfg.cnf_dim}")
    if cfg.benchmark_runs < 1:
        raise ValueError("benchmark_runs must be >= 1")

    return cfg
