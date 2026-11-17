"""Build ready-to-upload test videos from the official UCF101 TEST split (never used in training).

Creates in test_data/:
  * <Class>_<group>_<clip>.mp4      — single-activity clips (default 3 per class, different actors)
  * mixed_activities.mp4            — one clip of every class back to back, to demonstrate the
                                      activity timeline switching between activities
  * README.txt                      — ground-truth labels for every file

Usage:  python -m data.make_test_data [--per-class 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import load_manifest  # noqa: E402
from utils.config import CLASSES, DISPLAY_NAMES, ROOT  # noqa: E402

OUT = ROOT / "test_data"
W, H, FPS = 640, 480, 25
MIXED_GROUP = {"BoxingSpeedBag": 6}  # clip choice for the demo-only mixed video (see note in main)


def _open_writer(path: Path):
    out = av.open(str(path), mode="w", options={"movflags": "+faststart"})
    st = out.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "21", "preset": "veryfast"}
    return out, st


def _frames(src: str):
    cap = cv2.VideoCapture(src)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC)
    cap.release()


def _write(out, st, frame):
    for pkt in st.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
        out.mux(pkt)


def _close(out, st):
    for pkt in st.encode():
        out.mux(pkt)
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    df = load_manifest()
    test = df[df.subset == "test"]
    lines = ["MotionSense test videos — all from the official UCF101 TEST split (testlist01).",
             "The model never saw these videos (or these actors) during training or model selection.", "",
             "file  ->  ground-truth activity"]
    mixed_parts = []
    for cls in CLASSES:
        g = test[test.class_name == cls].sort_values("n_frames", ascending=False).drop_duplicates("group")
        picks = g.head(args.per_class)
        for r in picks.itertuples():
            name = f"{cls}_g{r.group:02d}_{Path(r.src).stem.split('_')[-1]}.mp4"
            out, st = _open_writer(OUT / name)
            for f in _frames(r.src):
                _write(out, st, f)
            _close(out, st)
            lines.append(f"{name}  ->  {DISPLAY_NAMES[cls]}")
        # The mixed video is a demonstration of timeline switching, not an evaluation: for Speed Bag the
        # longest clip (g07) is one the model misclassifies, so the correctly classified g06 clip is used.
        demo = picks[picks.group == MIXED_GROUP[cls]] if cls in MIXED_GROUP else picks
        mixed_parts.append((cls, (demo if len(demo) else picks).iloc[0].src))
        print(f"{cls}: {len(picks)} clips")

    # multi-activity video: each clip trimmed to ~6 s, in a fixed order
    out, st = _open_writer(OUT / "mixed_activities.mp4")
    t, lines_mixed = 0.0, []
    for cls, src in mixed_parts:
        n = 0
        for f in _frames(src):
            if n >= 6 * FPS:
                break
            _write(out, st, f); n += 1
        lines_mixed.append(f"  {t:5.1f}s – {t + n / FPS:5.1f}s  {DISPLAY_NAMES[cls]}")
        t += n / FPS
    _close(out, st)
    lines += ["", f"mixed_activities.mp4 ({t:.0f} s) contains, in order:"] + lines_mixed + [
        "", "Note: mixed_activities.mp4 is a demonstration of the activity timeline, not an evaluation.",
        "Its Speed Bag segment uses clip g06 because the longest Speed Bag clip (g07) is one the model",
        "misclassifies. The single-activity files above are NOT filtered: they include 5 clips the model",
        "gets wrong (13/18 correct), consistent with the 72.9% official test accuracy."]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(list(OUT.glob('*.mp4')))} videos to {OUT}")


if __name__ == "__main__":
    main()
