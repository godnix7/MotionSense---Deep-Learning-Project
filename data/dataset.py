"""PyTorch datasets over the preprocessed frame cache."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import IMAGENET_MEAN, IMAGENET_STD, MANIFEST_PATH, TrainConfig  # noqa: E402

_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def center_crop_resize(clip: np.ndarray, size: int) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (T,3,size,size) uint8 tensor: square center crop + resize."""
    t = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)
    h, w = t.shape[-2:]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    t = t[:, :, y:y + s, x:x + s]
    if s != size:
        t = torch.nn.functional.interpolate(t.float(), size=(size, size), mode="bilinear",
                                            antialias=True, align_corners=False).round().clamp(0, 255).byte()
    return t


def normalize(t: torch.Tensor) -> torch.Tensor:
    """uint8 (...,3,H,W) -> normalized float."""
    mean, std = _MEAN.to(t.device), _STD.to(t.device)
    return (t.float() / 255.0 - mean) / std


class VideoClipDataset(Dataset):
    """Samples fixed-length clips (seq_len frames) from cached videos.

    train=True : random temporal offset, random temporal stride (speed jitter), clip-consistent
                 random-resized crop, horizontal flip and brightness/contrast jitter.
    train=False: deterministic clips (uniformly spaced `eval_clips` per video) with center crop.
    """

    def __init__(self, df: pd.DataFrame, cfg: TrainConfig, train: bool, eval_clips: int = 3):
        self.df = df.reset_index(drop=True)
        self.cfg, self.train, self.eval_clips = cfg, train, eval_clips
        if train:
            self.index = [(i, None) for i in range(len(self.df)) for _ in range(cfg.clips_per_video)]
        else:
            self.index = [(i, k) for i in range(len(self.df)) for k in range(eval_clips)]

    def __len__(self) -> int:
        return len(self.index)

    def _temporal_indices(self, n: int, k: int | None) -> np.ndarray:
        L = self.cfg.seq_len
        if self.train:
            stride = random.choices([1, 1, 1, 2], k=1)[0] if n >= 2 * L else 1
            span = (L - 1) * stride + 1
            start = random.randint(0, max(n - span, 0))
            idx = start + np.arange(L) * stride
        else:
            span = L
            starts = np.linspace(0, max(n - span, 0), self.eval_clips).round().astype(int)
            idx = starts[k] + np.arange(L)
        return np.clip(idx, 0, n - 1)  # short videos: repeat last frame

    def _augment(self, frames: np.ndarray) -> torch.Tensor:
        size = self.cfg.img_size
        t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
        h, w = t.shape[-2:]
        area = h * w
        for _ in range(10):
            target = area * random.uniform(0.45, 1.0)
            ar = np.exp(random.uniform(np.log(3 / 4), np.log(4 / 3)))
            cw, ch = int(round((target * ar) ** 0.5)), int(round((target / ar) ** 0.5))
            if cw <= w and ch <= h:
                break
        else:
            cw = ch = min(h, w)
        y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        t = t[:, :, y:y + ch, x:x + cw]
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        if random.random() < 0.5:
            t = t.flip(-1)
        b, c = random.uniform(-25, 25), random.uniform(0.75, 1.25)
        t = ((t - 128) * c + 128 + b)
        if random.random() < 0.15:
            t = t.mean(1, keepdim=True).expand_as(t)
        return t.clamp(0, 255).byte()

    def __getitem__(self, i: int):
        row_i, k = self.index[i]
        row = self.df.iloc[row_i]
        video = np.load(row["cache"], mmap_mode="r")
        frames = video[self._temporal_indices(video.shape[0], k)]
        if self.train:
            # raw frames at a fixed size; spatial/photometric augmentation runs batched on the GPU
            fh, fw = TRAIN_RAW
            if frames.shape[1:3] != (fh, fw):
                frames = np.stack([cv2.resize(f, (fw, fh), interpolation=cv2.INTER_AREA) for f in frames])
            clip = torch.from_numpy(np.ascontiguousarray(frames))
        else:
            clip = center_crop_resize(frames, self.cfg.img_size)
        return clip, int(row["label"]), row_i  # uint8; normalization happens on the GPU


TRAIN_RAW = (160, 213)  # cached UCF101 frames (320x240 source, short side 160)


def gpu_augment(clips: torch.Tensor, size: int) -> torch.Tensor:
    """Clip-consistent augmentation on the GPU.
    clips: (B,T,H,W,3) uint8 -> (B,T,3,size,size) uint8.
    Random resized crop (area 45-100%, aspect 3/4-4/3) via one batched roi_align, horizontal
    flip, brightness/contrast jitter and random grayscale — identical within a clip."""
    from torchvision.ops import roi_align
    B, T, H, W, _ = clips.shape
    dev = clips.device
    x = clips.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W).float()
    area = torch.empty(B, device=dev).uniform_(0.45, 1.0) * H * W
    ar = torch.exp(torch.empty(B, device=dev).uniform_(np.log(3 / 4), np.log(4 / 3)))
    cw = torch.sqrt(area * ar).clamp(max=W)
    ch = torch.sqrt(area / ar).clamp(max=H)
    x0 = torch.rand(B, device=dev) * (W - cw)
    y0 = torch.rand(B, device=dev) * (H - ch)
    boxes = torch.stack([x0, y0, x0 + cw, y0 + ch], 1).repeat_interleave(T, 0)
    boxes = torch.cat([torch.arange(B * T, device=dev, dtype=torch.float32)[:, None], boxes], 1)
    x = roi_align(x, boxes, output_size=(size, size), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    x = x.view(B, T, 3, size, size)
    flip = torch.rand(B, device=dev) < 0.5
    x[flip] = x[flip].flip(-1)
    c = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    b = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(-25, 25)
    x = (x - 128) * c + 128 + b
    gray = torch.rand(B, device=dev) < 0.15
    x[gray] = x[gray].mean(2, keepdim=True).expand_as(x[gray])
    return x.clamp(0, 255).round().byte()


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found. Run:  python -m data.preprocess")
    return pd.read_csv(MANIFEST_PATH)


def make_loaders(cfg: TrainConfig):
    df = load_manifest()
    df = df[df["class_name"].isin(cfg.classes)]
    tr, va = df[df.subset == "train"], df[df.subset == "val"]
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True,
              persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(VideoClipDataset(tr, cfg, train=True), shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(VideoClipDataset(va, cfg, train=False), shuffle=False, **kw)
    return train_loader, val_loader, df
