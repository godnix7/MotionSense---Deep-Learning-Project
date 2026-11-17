"""MobileNetV3-Large (spatial) + LSTM (temporal) activity recognition network.

    clip (B,T,3,H,W) -> MobileNetV3 features per frame (B,T,960)
                     -> LSTM over time            (B,T,hidden)
                     -> [last state ; temporal mean] -> FC head -> logits (B,C)

The two stages are exposed separately (`extract_features`, `classify_sequence`) so the
live engine can compute each frame's CNN feature once and reuse it across overlapping
temporal windows.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .classifier import ClassifierHead

FEATURE_DIM = 960


class MobileNetLSTM(nn.Module):
    def __init__(self, num_classes: int, lstm_hidden: int = 256, lstm_layers: int = 1,
                 dropout: float = 0.4, pretrained: bool = True, unfreeze_blocks: int = 4,
                 use_deltas: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.features = backbone.features            # conv trunk -> (N,960,h,w)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dropout = nn.Dropout(dropout / 2)
        # use_deltas: LSTM input = [f_t ; f_t - f_{t-1}]. Static scene content cancels in the
        # difference, pushing the temporal model towards motion rather than background.
        self.use_deltas = use_deltas
        self.lstm = nn.LSTM(FEATURE_DIM * (2 if use_deltas else 1), lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = ClassifierHead(2 * lstm_hidden, num_classes, hidden=lstm_hidden, dropout=dropout)
        self.unfreeze_blocks = unfreeze_blocks
        self.set_backbone_trainable(unfreeze_blocks)

    # ---- freezing ---------------------------------------------------------------------
    def set_backbone_trainable(self, last_n_blocks: int) -> None:
        n = len(self.features)
        for i, block in enumerate(self.features):
            req = i >= n - last_n_blocks
            for p in block.parameters():
                p.requires_grad = req
        self.unfreeze_blocks = last_n_blocks

    def train(self, mode: bool = True):
        super().train(mode)
        # keep BatchNorm statistics of frozen blocks fixed (small video batches)
        n = len(self.features)
        for i, block in enumerate(self.features):
            if i < n - self.unfreeze_blocks:
                block.eval()
        return self

    # ---- forward stages -----------------------------------------------------------------
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (N,3,H,W) normalized -> (N,960)."""
        return self.pool(self.features(frames)).flatten(1)

    def classify_sequence(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B,T,960) -> logits (B,C)."""
        if self.use_deltas:
            d = torch.diff(feats, dim=1, prepend=feats[:, :1])
            feats = torch.cat([feats, d], dim=-1)
        out, _ = self.lstm(self.feat_dropout(feats))
        z = torch.cat([out[:, -1], out.mean(1)], dim=1)
        return self.head(z)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        b, t = clips.shape[:2]
        feats = self.extract_features(clips.flatten(0, 1)).view(b, t, -1)
        return self.classify_sequence(feats)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parts = {
        "backbone": sum(p.numel() for p in model.features.parameters()),
        "lstm": sum(p.numel() for p in model.lstm.parameters()),
        "head": sum(p.numel() for p in model.head.parameters()),
    }
    return {"total": total, "trainable": trainable, **parts}
