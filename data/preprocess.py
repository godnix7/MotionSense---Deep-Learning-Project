"""Preprocess UCF101 videos into a compact frame cache + split manifest.

For each video of the selected classes:
  * frames are sampled at a fixed rate (SAMPLE_FPS) regardless of the source FPS,
  * resized so the short side equals CACHE_SHORT_SIDE (aspect ratio preserved),
  * stored as a uint8 array  (T, H, W, 3)  RGB  in  datasets/frames/<Class>/<video>.npy

Splits follow the official UCF101 train/test lists (split 1 by default). A validation
set is carved out of the *training* list by holding out whole groups (VAL_GROUPS), so
clips of the same actor/scene never appear in two subsets.

Usage:  python -m data.preprocess [--split 1] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import (  # noqa: E402
    CACHE_SHORT_SIDE, CLASSES, DISTRIBUTION_PATH, FRAMES_DIR, MANIFEST_PATH, REPORTS_DIR,
    SAMPLE_FPS, SPLITS_DIR, UCF_DIR, VAL_GROUPS,
)

GROUP_RE = re.compile(r"_g(\d+)_c(\d+)")


def resize_short(frame: np.ndarray, short: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = short / min(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def sample_video(path: str, sample_fps: float = SAMPLE_FPS, short: int = CACHE_SHORT_SIDE) -> np.ndarray:
    """Decode a video and return frames sampled at `sample_fps` as uint8 RGB (T,H,W,3)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = src_fps / sample_fps
    frames, idx, next_t = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= next_t:
            frames.append(resize_short(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), short))
            next_t += step
        idx += 1
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    return np.stack(frames)


def _process(src: str, dst: str) -> tuple[str, int]:
    dst_p = Path(dst)
    if dst_p.exists():
        return dst, int(np.load(dst, mmap_mode="r").shape[0])
    arr = sample_video(src)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr)
    return dst, int(arr.shape[0])


def build_manifest(classes: list[str], split: int) -> pd.DataFrame:
    train_list = SPLITS_DIR / f"trainlist0{split}.txt"
    test_list = SPLITS_DIR / f"testlist0{split}.txt"
    for f in (train_list, test_list):
        if not f.exists():
            raise SystemExit(f"Missing split file {f}. Run:  python -m data.download")
    cls_set = set(classes)
    rows = []
    for subset, path in (("train", train_list), ("test", test_list)):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rel = line.split()[0].replace("\\", "/")
            cls, fname = rel.split("/")
            if cls not in cls_set:
                continue
            group = int(GROUP_RE.search(fname).group(1))
            sub = subset
            if subset == "train" and group in VAL_GROUPS:
                sub = "val"
            rows.append({
                "video": fname, "class_name": cls, "label": classes.index(cls),
                "subset": sub, "group": group,
                "src": str(UCF_DIR / cls / fname),
                "cache": str(FRAMES_DIR / cls / fname.replace(".avi", ".npy")),
            })
    df = pd.DataFrame(rows)
    # leakage guard: groups must not be shared between subsets of the same class
    for cls, g in df.groupby("class_name"):
        sets = {s: set(x["group"]) for s, x in g.groupby("subset")}
        names = list(sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert not (sets[names[i]] & sets[names[j]]), f"group leakage in {cls}"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--classes", nargs="+", default=CLASSES)
    args = ap.parse_args()

    missing = [c for c in args.classes if not (UCF_DIR / c).exists()]
    if missing:
        raise SystemExit(f"Videos for {missing} not found in {UCF_DIR}. Run:  python -m data.download")

    df = build_manifest(args.classes, args.split)
    absent = df[~df["src"].map(lambda p: Path(p).exists())]
    if len(absent):
        raise SystemExit(f"{len(absent)} videos listed in the split files are missing on disk, e.g. {absent.iloc[0]['src']}")

    print(f"Caching {len(df)} videos at {SAMPLE_FPS} fps, short side {CACHE_SHORT_SIDE}px ...")
    lengths = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process, r.src, r.cache): r.cache for r in df.itertuples()}
        for k, fut in enumerate(as_completed(futs), 1):
            dst, n = fut.result()
            lengths[dst] = n
            if k % 50 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)}", flush=True)
    df["n_frames"] = df["cache"].map(lengths)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)

    dist = (df.groupby(["class_name", "subset"]).size().unstack(fill_value=0)
              .reindex(args.classes).reindex(columns=["train", "val", "test"], fill_value=0))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DISTRIBUTION_PATH.write_text(json.dumps({
        "split": args.split, "val_groups": list(VAL_GROUPS),
        "counts": {c: {k: int(v) for k, v in dist.loc[c].items()} for c in args.classes},
        "mean_cached_frames": float(df["n_frames"].mean()),
        "sample_fps": SAMPLE_FPS,
    }, indent=2))
    print(dist)
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
