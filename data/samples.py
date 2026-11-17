"""Create demo assets from the official TEST split (never used for training):
  * samples/<Class>_<video>.avi   — sample clips for the demo stream / video analysis
  * assets/<Class>.jpg            — one authentic frame per class for the overview page

Usage:  python -m data.samples [--per-class 2]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import load_manifest  # noqa: E402
from utils.config import ASSETS_DIR, SAMPLES_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()
    df = load_manifest()
    test = df[df.subset == "test"]
    SAMPLES_DIR.mkdir(exist_ok=True); ASSETS_DIR.mkdir(exist_ok=True)
    for cls, g in test.groupby("class_name"):
        g = g.sort_values("n_frames", ascending=False)
        # pick long clips from different groups
        picks = g.drop_duplicates("group").head(args.per_class)
        for r in picks.itertuples():
            dst = SAMPLES_DIR / f"{cls}_{Path(r.src).stem}.avi"
            if not dst.exists():
                shutil.copy(r.src, dst)
        r = picks.iloc[0]
        cap = cv2.VideoCapture(r.src)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read(); cap.release()
        if ok:
            cv2.imwrite(str(ASSETS_DIR / f"{cls}.jpg"), cv2.resize(frame, (480, 360)), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{cls}: {len(picks)} samples")
    print(f"Samples in {SAMPLES_DIR}, thumbnails in {ASSETS_DIR}")


if __name__ == "__main__":
    main()
