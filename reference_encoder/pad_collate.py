import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    """
    Pads a batch of (input, label) pairs for either
    - raw 1D waveforms → (B, N)
    - 2D mel-spectrograms → (B, n_mels, T)
    """
    inputs, labels = zip(*batch)

    # If mel came back as shape (1, n_mels, T), squeeze to (n_mels, T)
    if inputs[0].dim() == 3 and inputs[0].shape[0] == 1:
        inputs = [x.squeeze(0) for x in inputs]

    labels = torch.tensor(labels, dtype=torch.long)

    # raw-waveform case: each inp is 1D → pad to (B, N)
    if inputs[0].dim() == 1:
        padded = pad_sequence(inputs, batch_first=True)
        return padded, labels

    # melspectrogram case: each inp is (T, n_mels)
    # 1) pad to (B, T_max, n_mels)
    padded = pad_sequence(inputs, batch_first=True)
    # 2) transpose to (B, n_mels, T_max)
    padded = padded.transpose(1, 2)
    return padded, labels
