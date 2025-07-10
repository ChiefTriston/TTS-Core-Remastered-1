import torch.nn as nn

class HSFLayer(nn.Module):
    """
    Harmonic-Source-Filter block: models the residual harmonic structure.
    Input/output shape: (B, C, T)
    """
    def __init__(self, channels, hidden=None, layers=3, kernel_size=3):
        super().__init__()
        hidden = hidden or channels
        ks = kernel_size // 2
        seq = []
        seq.append(nn.Conv1d(channels,   hidden,   kernel_size, padding=ks))
        seq.append(nn.ReLU(inplace=True))
        for _ in range(layers - 2):
            seq.append(nn.Conv1d(hidden, hidden, kernel_size, padding=ks))
            seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv1d(hidden, channels, kernel_size, padding=ks))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)
