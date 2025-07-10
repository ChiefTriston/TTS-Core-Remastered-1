import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """
    ArcFace margin-based classification loss.
    """
    def __init__(self, in_features: int, num_classes: int, margin: float = 0.3, scale: float = 30.0, margin_schedule=None):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.margin_schedule = margin_schedule
        # weight shape: (num_classes, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, step: int = None) -> torch.Tensor:
        """
        embeddings: (B, D)
        labels:     (B,)
        """
        if self.margin_schedule is not None and step is not None:
            self.margin = self.margin_schedule(step)
        # normalize embeddings and weights
        emb_n = F.normalize(embeddings, p=2, dim=1)  # (B, D)
        Wn = F.normalize(self.weight, p=2, dim=1)    # (C, D)
        # cosine similarity
        cos = F.linear(emb_n, Wn)                    # (B, C)
        # add margin only to target logits
        one_hot = F.one_hot(labels, num_classes=cos.size(1)).float()
        phi = cos - one_hot * self.margin
        logits = phi * self.scale
        return self.ce(logits, labels)

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End loss for speaker embeddings.
    """
    def __init__(self, w: float = 10.0, b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w))
        self.b = nn.Parameter(torch.tensor(b))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D), labels: (B,)
        B distinct speakers, M utterances per speaker => B*M = batch size
        """
        unique_labels = labels.unique()
        B_spk = unique_labels.numel()
        M = embeddings.size(0) // B_spk
        D = embeddings.size(1)
        e = embeddings.view(B_spk, M, D)
        c = (e.sum(dim=1, keepdim=True) - e) / (M - 1)
        sims = []
        for i in range(B_spk):
            sim_i = self.w * F.cosine_similarity(e[i].unsqueeze(1), c[i], dim=-1) + self.b
            sims.append(sim_i)
        sim_mat = torch.cat(sims, dim=0)
        target = labels
        return F.cross_entropy(sim_mat, target)