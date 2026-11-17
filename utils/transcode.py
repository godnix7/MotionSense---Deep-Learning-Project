"""Browser-playable H.264 MP4 export (original + prediction-annotated copies)."""
from __future__ import annotations

from typing import Callable

import av
import cv2
import numpy as np


def write_mp4(src: str, dst: str, annotate: Callable[[np.ndarray, float], np.ndarray] | None = None,
              max_width: int = 640) -> str:
    """Re-encode `src` to H.264/yuv420p MP4 (plays in every browser). Optionally draw on
    each BGR frame with annotate(frame, t_seconds)."""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, max_width / max(w, 1))
    ow, oh = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    out = av.open(dst, mode="w", options={"movflags": "+faststart"})
    stream = out.add_stream("libx264", rate=round(fps))
    stream.width, stream.height, stream.pix_fmt = ow, oh, "yuv420p"
    stream.options = {"crf": "23", "preset": "veryfast"}
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (ow, oh):
                frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            if annotate is not None:
                frame = annotate(frame, i / fps)
            vf = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for pkt in stream.encode(vf):
                out.mux(pkt)
            i += 1
        for pkt in stream.encode():
            out.mux(pkt)
    finally:
        out.close()
        cap.release()
    return dst


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    c = hex_color.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


def prediction_annotator(windows: list[dict], label_fn: Callable[[str], str], color_fn: Callable[[str], str]):
    """Returns annotate(frame, t) drawing the prediction of the window centred nearest to t."""
    centers = np.array([w["t_center"] for w in windows])

    def annotate(frame: np.ndarray, t: float) -> np.ndarray:
        w = windows[int(np.argmin(np.abs(centers - t)))]
        h, wd = frame.shape[:2]
        bar = max(28, h // 11)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar), (wd, h), (61, 38, 20), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        cv2.rectangle(frame, (0, h - bar), (6, h), hex_to_bgr(color_fn(w["label"])), -1)
        fs = bar / 48
        cv2.putText(frame, label_fn(w["label"]).upper(), (16, h - bar // 3), cv2.FONT_HERSHEY_DUPLEX, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)
        txt = f"{w['confidence']:.0%}  t={t:4.1f}s"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, 1)
        cv2.putText(frame, txt, (wd - tw - 10, h - bar // 3), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85,
                    (220, 228, 236), 1, cv2.LINE_AA)
        return frame

    return annotate
