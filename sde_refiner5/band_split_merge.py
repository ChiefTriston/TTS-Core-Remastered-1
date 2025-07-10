import torch
import torch.nn as nn
from typing import List

class BandSplitMerge(nn.Module):
    """
    Splits along channel dim into specified bands and merges back.
    """
    def __init__(self, input_dim: int, bands: List[int]):
        super().__init__()
        self.bands = bands
        # Precompute cumulative sums for slicing
        self.cum = [0] + torch.cumsum(torch.tensor(bands, dtype=torch.long), dim=0).tolist()
        if sum(bands) != input_dim:
            raise ValueError(f"Sum of bands {sum(bands)} != input_dim {input_dim}")

    def split(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, C, T]
        return [x[:, self.cum[i]:self.cum[i+1], :] for i in range(len(self.bands))]

    def merge(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # xs: list of [B, b_i, T]
        return torch.cat(xs, dim=1)
