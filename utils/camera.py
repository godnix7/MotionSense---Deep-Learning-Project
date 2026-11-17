"""Frame sources for live recognition.

* LocalCamera  — OpenCV capture on the machine running the app (most reliable for a
                 classroom demo on the same laptop; no browser/WebRTC negotiation).
* VideoStream  — plays a video file in real time through the same streaming pipeline
                 (demonstration mode when no webcam is available). Predictions still come
                 from the trained network.
Both run a capture thread that pushes frames into a StreamingRecognizer and keep the
most recent annotated frame for display.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from models.inference import UNKNOWN


def draw_overlay(frame: np.ndarray, snap: dict | None) -> np.ndarray:
    """Minimal technical HUD drawn onto a BGR frame."""
    if snap is None:
        return frame
    h, w = frame.shape[:2]
    out = frame
    bar_h = max(34, h // 12)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (61, 38, 20), -1)  # #14263D in BGR
    out = cv2.addWeighted(overlay, 0.78, out, 0.22, 0)
    label = snap.get("label", "—")
    conf = snap.get("confidence", 0.0)
    status = snap.get("status", "")
    color = (180, 188, 199) if label in (UNKNOWN, "—") else (161, 166, 22)  # grey / teal (BGR)
    fs = bar_h / 52
    cv2.putText(out, label.upper(), (12, h - bar_h // 3), cv2.FONT_HERSHEY_DUPLEX, fs, color, 1, cv2.LINE_AA)
    right = f"{conf:5.1%}  {status}" if status not in ("BUFFERING", "IDLE") else \
        f"BUFFER {snap.get('buffer_fill', 0)}/{snap.get('seq_len', 16)}"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, 1)
    cv2.putText(out, right, (w - tw - 12, h - bar_h // 3), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8,
                (230, 235, 240), 1, cv2.LINE_AA)
    # buffer fill indicator
    fill = snap.get("buffer_fill", 0) / max(snap.get("seq_len", 16), 1)
    cv2.rectangle(out, (0, h - bar_h - 3), (int(w * fill), h - bar_h), (170, 105, 23), -1)  # #1769AA
    return out


class _ThreadedSource:
    def __init__(self, recognizer):
        self.rec = recognizer
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error: str | None = None
        self.fps = 0.0
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def _publish(self, frame: np.ndarray):
        # encode on the capture thread (OpenCV releases the GIL) so the UI thread only ships bytes
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._lock:
            self._frame = frame
            self._jpeg = buf.tobytes() if ok else None

    def _loop(self):
        raise NotImplementedError


class LocalCamera(_ThreadedSource):
    def __init__(self, recognizer, index: int = 0, width: int = 640, height: int = 480, mirror: bool = True):
        super().__init__(recognizer)
        self.index, self.width, self.height, self.mirror = index, width, height, mirror

    def _open(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
        for be in backends:
            cap = cv2.VideoCapture(self.index, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
        return None

    def _loop(self):
        cap = self._open()
        if cap is None:
            self.error = (f"Camera {self.index} could not be opened. It may be in use by another application "
                          "(Teams, Zoom, browser tab), disabled in Windows privacy settings, or not connected.")
            return
        times = []
        fails = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:
                        self.error = "The camera stopped delivering frames."
                        return
                    time.sleep(0.03)
                    continue
                fails = 0
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                now = time.time()
                times = [t for t in times if now - t < 2] + [now]
                self.fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 2 else 0.0
                self.rec.push_frame(frame, now)
                self._publish(draw_overlay(frame, self.rec.overlay_state()))
        finally:
            cap.release()


class VideoStream(_ThreadedSource):
    """Real-time playback of a file through the streaming recognizer (loops)."""

    def __init__(self, recognizer, path: str, loop: bool = True, display_width: int = 640):
        super().__init__(recognizer)
        self.path, self.loop, self.display_width = path, loop, display_width

    def _loop(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.error = f"Could not open {self.path}"
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dt = 1.0 / fps
        t_next = time.perf_counter()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if not self.loop:
                        return
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                h, w = frame.shape[:2]
                if w != self.display_width:
                    frame = cv2.resize(frame, (self.display_width, round(h * self.display_width / w)))
                self.fps = fps
                self.rec.push_frame(frame, time.time())  # paced in real time, like a camera
                self._publish(draw_overlay(frame, self.rec.overlay_state()))
                t_next += dt
                sleep = t_next - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    t_next = time.perf_counter()
        finally:
            cap.release()


def list_cameras(max_index: int = 4) -> list[int]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found
