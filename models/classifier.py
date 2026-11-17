"""Fully connected classification head."""
from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits; softmax is applied at inference / inside CrossEntropyLoss
