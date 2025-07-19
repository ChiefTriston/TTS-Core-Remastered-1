# emotion_classifier.py
import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=23, num_emotions=6, hidden_dim=128):  # 4 VADER + 19 prosody
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dropout=0.1),
            num_layers=2
        )
        self.fc = nn.Linear(input_dim, num_emotions)
    
    def forward(self, emotion_vector):
        trans_out = self.transformer(emotion_vector.unsqueeze(1)).squeeze(1)
        return torch.sigmoid(self.fc(trans_out))

class EmotionWeightLearner(nn.Module):
    def __init__(self, vader_dim=4, prosody_dim=19):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vader_dim + prosody_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, vader, prosody):
        return self.mlp(torch.cat([vader, prosody], dim=-1))