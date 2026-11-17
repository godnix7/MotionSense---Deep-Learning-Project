"""Inference: model loading, streaming (webcam) recognition and offline video analysis.

Real-time strategy
------------------
* Camera frames arrive at 25-60 fps; they are sub-sampled to SAMPLE_FPS (8 fps), the same
  temporal rate used for training, so a 16-frame window always spans ~2 s of motion.
* Each sampled frame passes through MobileNetV3 exactly once; its 960-d feature is kept in a
  rolling buffer. Every `infer_every` new frames the LSTM + head classify the last
  `seq_len` features. This avoids re-running the CNN on overlapping windows.
* All network work happens on a background worker thread so the video path never blocks.
* Window probabilities are smoothed with an EMA and a change needs two consecutive
  agreeing windows (hysteresis), which prevents label flicker. Below the confidence
  threshold the system reports UNKNOWN / UNCERTAIN ACTIVITY instead of forcing a class.
"""
from __future__ import annotations

import collections
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from utils.config import CHECKPOINT_PATH, IMAGENET_MEAN, IMAGENET_STD, InferenceConfig

from .mobilenet_lstm import FEATURE_DIM, MobileNetLSTM

UNKNOWN = "UNKNOWN / UNCERTAIN ACTIVITY"

_MODEL_CACHE: dict = {}
_LOCK = threading.Lock()


class ModelNotFoundError(FileNotFoundError):
    pass


def pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(path: Path = CHECKPOINT_PATH):
    """Load the trained checkpoint once per process (thread-safe)."""
    with _LOCK:
        key = str(path)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        if not Path(path).exists():
            raise ModelNotFoundError(
                f"Trained model not found at {path}.\n"
                "Prepare the dataset and train first:\n"
                "  python -m data.download\n  python -m data.preprocess\n"
                "  python -m training.train\n  python -m training.evaluate")
        device = pick_device()
        meta = torch.load(path, map_location="cpu", weights_only=False)
        tc = meta["train_config"]
        model = MobileNetLSTM(len(meta["classes"]), tc["lstm_hidden"], tc["lstm_layers"], tc["dropout"],
                              pretrained=False, unfreeze_blocks=tc["unfreeze_blocks"],
                              use_deltas=tc.get("use_deltas", False))
        model.load_state_dict(meta["state_dict"])
        model.eval().to(device)
        if device.type == "cuda":
            # Inference sees many input shapes (1 frame live, variable batches for files). cuDNN
            # autotuning would re-benchmark for every new shape and cause multi-second stalls.
            torch.backends.cudnn.benchmark = False
            S = meta["inference"]["img_size"]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):  # warm-up kernels
                model(torch.zeros(1, meta["inference"]["seq_len"], 3, S, S, device=device))
                model.extract_features(torch.zeros(1, 3, S, S, device=device))
                model.extract_features(torch.zeros(32, 3, S, S, device=device))
                model.classify_sequence(torch.zeros(64, meta["inference"]["seq_len"], 960, device=device))
            torch.cuda.synchronize()
        _attach_live_graphs(model, meta, device)
        meta.pop("state_dict")
        _MODEL_CACHE[key] = (model, meta, device)
        return _MODEL_CACHE[key]


class _FrameEncoder(torch.nn.Module):
    """uint8 RGB frame (1,S,S,3) -> 960-d feature. Normalisation + FP16 MobileNetV3 in one graph."""

    def __init__(self, features: torch.nn.Module, half: bool):
        super().__init__()
        self.features, self.half = features, half
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, frame_u8: torch.Tensor) -> torch.Tensor:
        x = (frame_u8.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
        if self.half:
            x = x.half()
        return torch.nn.functional.adaptive_avg_pool2d(self.features(x), 1).flatten(1).float()


class _SequenceHead(torch.nn.Module):
    """(1,T,960) features -> class probabilities (LSTM + FC + softmax)."""

    def __init__(self, model: MobileNetLSTM):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.model.classify_sequence(feats).softmax(-1)


def _attach_live_graphs(model: MobileNetLSTM, meta: dict, device: torch.device) -> None:
    """TorchScript-compile the per-frame encoder and the sequence head for the live path.

    A scripted module runs its whole graph in C++ and releases the Python GIL while doing so.
    In the Streamlit app the UI thread is busy (video refresh, panels, charts); an eager
    forward pass needs the GIL ~150 times and stalls behind it (measured: 11 ms outside the
    app vs 300+ ms inside). One GIL-free call per frame removes that contention."""
    import copy
    S, T = meta["inference"]["img_size"], meta["inference"]["seq_len"]
    half = device.type == "cuda"
    try:
        enc = _FrameEncoder(copy.deepcopy(model.features), half).to(device).eval()
        if half:
            enc.features.half()
        head = _SequenceHead(model).eval()
        with torch.no_grad():
            model.live_encoder = torch.jit.trace(enc, torch.zeros(1, S, S, 3, dtype=torch.uint8, device=device))
            model.live_head = torch.jit.trace(head, torch.zeros(1, T, FEATURE_DIM, device=device), check_trace=False)
            for _ in range(3):  # warm-up / profile the scripted graphs
                model.live_encoder(torch.zeros(1, S, S, 3, dtype=torch.uint8, device=device))
                model.live_head(torch.zeros(1, T, FEATURE_DIM, device=device))
        if half:
            torch.cuda.synchronize()
    except Exception as e:  # fall back to the eager path; never block the app
        print(f"[motionsense] TorchScript live path unavailable ({e}); using eager inference")
        model.live_encoder = model.live_head = None


class Preprocessor:
    """BGR uint8 frame -> normalized (3,S,S) tensor on device. Mirrors training preprocessing:
    resize short side to S (INTER_AREA) then square center crop."""

    def __init__(self, img_size: int, device: torch.device):
        self.size, self.device = img_size, device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def crop(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.size / min(h, w)
        img = cv2.resize(bgr, (max(self.size, round(w * s)), max(self.size, round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        y, x = (h - self.size) // 2, (w - self.size) // 2
        return cv2.cvtColor(img[y:y + self.size, x:x + self.size], cv2.COLOR_BGR2RGB)

    def to_tensor(self, rgb_crops: list[np.ndarray]) -> torch.Tensor:
        t = torch.from_numpy(np.stack(rgb_crops)).to(self.device, non_blocking=True)
        t = t.permute(0, 3, 1, 2).float().div_(255.0)
        return (t - self.mean) / self.std


@torch.no_grad()
def _features(model, x: torch.Tensor, device) -> torch.Tensor:
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        return model.extract_features(x).float()


@torch.no_grad()
def _classify(model, feats: torch.Tensor, device) -> torch.Tensor:
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        return model.classify_sequence(feats).float().softmax(-1)


# ======================================================================================
# Streaming recognizer (webcam / simulated stream)
# ======================================================================================
@dataclass
class Segment:
    label: str
    start: float
    end: float
    conf_sum: float = 0.0
    n: int = 0

    @property
    def confidence(self) -> float:
        return self.conf_sum / max(self.n, 1)


@dataclass
class StreamState:
    label: str = "—"
    confidence: float = 0.0
    status: str = "IDLE"
    probs: list = field(default_factory=list)
    raw_probs: list = field(default_factory=list)
    buffer_fill: int = 0
    latency_ms: float = 0.0
    cnn_ms: float = 0.0
    lstm_ms: float = 0.0
    infer_fps: float = 0.0
    camera_fps: float = 0.0
    motion: float = 0.0
    windows: int = 0


class StreamingRecognizer:
    def __init__(self, cfg: InferenceConfig | None = None, model_bundle=None):
        self.model, self.meta, self.device = model_bundle or load_model()
        self.classes = self.meta["classes"]
        self.names = self.meta.get("display_names", {c: c for c in self.classes})
        self.cfg = cfg or InferenceConfig()
        self.cfg.seq_len = self.meta["inference"]["seq_len"]
        self.cfg.img_size = self.meta["inference"]["img_size"]
        self.cfg.sample_fps = self.meta["inference"]["sample_fps"]
        self.pre = Preprocessor(self.cfg.img_size, self.device)
        # PyTorch issues ~150 small Python->CUDA calls per forward pass. With the default 5 ms
        # GIL switch interval, a busy capture thread makes the inference thread wait up to 5 ms
        # per call (GIL convoy), turning a ~5 ms forward into ~1 s. A short interval fixes this.
        sys.setswitchinterval(0.0002)
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.paused = False
        self._reset_buffers()
        self._worker = threading.Thread(target=self._run, daemon=True, name="motionsense-infer")
        self._worker.start()

    # ---- public API ---------------------------------------------------------------------
    def _reset_buffers(self):
        self.feats: collections.deque = collections.deque(maxlen=self.cfg.seq_len)
        self.motion_hist: collections.deque = collections.deque(maxlen=self.cfg.seq_len)
        self.ema: np.ndarray | None = None
        self.committed: str | None = None
        self.pending: tuple[str, int] | None = None
        self.segments: list[Segment] = []
        self.timeline: collections.deque = collections.deque(maxlen=600)  # (t, label, conf, probs)
        self.state = StreamState()
        self._since_infer = 0
        self._last_sample_t = -1e9
        self._prev_small: np.ndarray | None = None
        self._cam_times: collections.deque = collections.deque(maxlen=60)
        self._inf_times: collections.deque = collections.deque(maxlen=20)
        self._lat: collections.deque = collections.deque(maxlen=30)
        self.t0 = time.time()

    def reset(self):
        with self._lock:
            while not self._q.empty():
                try: self._q.get_nowait()
                except queue.Empty: break
            self._reset_buffers()

    def pause(self):
        self.paused = True
        with self._lock:
            self.state.status = "PAUSED"

    def resume(self):
        with self._lock:
            self.feats.clear(); self.motion_hist.clear(); self._prev_small = None
        self.paused = False

    def close(self):
        self._stop.set()

    def push_frame(self, bgr: np.ndarray, t: float | None = None) -> None:
        """Called for every camera frame (any rate). Cheap: only sampled frames are queued."""
        t = time.time() if t is None else t
        self._cam_times.append(t)
        interval = 1.0 / self.cfg.sample_fps
        if self.paused or t - self._last_sample_t < interval - 0.02:
            return
        # drift-free schedule: keeps the average rate at sample_fps for any camera frame rate
        self._last_sample_t = max(self._last_sample_t + interval, t - interval)
        crop = self.pre.crop(bgr)
        try:
            self._q.put_nowait((t, crop))
        except queue.Full:  # worker is behind: drop the oldest to stay real-time
            try: self._q.get_nowait()
            except queue.Empty: pass
            self._q.put_nowait((t, crop))

    @torch.no_grad()
    def _encode(self, crop_rgb: np.ndarray) -> torch.Tensor:
        enc = getattr(self.model, "live_encoder", None)
        if enc is not None:  # TorchScript: one GIL-free call
            return enc(torch.from_numpy(np.ascontiguousarray(crop_rgb))[None].to(self.device))
        return _features(self.model, self.pre.to_tensor([crop_rgb]), self.device)

    @torch.no_grad()
    def _head(self, seq: torch.Tensor) -> torch.Tensor:
        head = getattr(self.model, "live_head", None)
        if head is not None:
            return head(seq.float())
        return _classify(self.model, seq, self.device)

    def overlay_state(self) -> dict:
        """Tiny state for per-frame video overlays (called ~30x/s by capture threads).
        Avoids copying the timeline/segments, which would contend with the inference thread."""
        s = self.state
        return {"label": s.label, "confidence": s.confidence, "status": s.status,
                "buffer_fill": len(self.feats), "seq_len": self.cfg.seq_len}

    def snapshot(self) -> dict:
        with self._lock:
            s = self.state
            cam = self._cam_times
            cam_fps = (len(cam) - 1) / (cam[-1] - cam[0]) if len(cam) > 2 and cam[-1] > cam[0] else 0.0
            if cam and time.time() - cam[-1] > 1.5:
                cam_fps = 0.0
            segs = [dict(label=g.label, start=g.start, end=g.end, confidence=g.confidence) for g in self.segments[-50:]]
            return dict(
                label=s.label, confidence=s.confidence, status=s.status, probs=list(s.probs),
                raw_probs=list(s.raw_probs), buffer_fill=len(self.feats), seq_len=self.cfg.seq_len,
                latency_ms=s.latency_ms, cnn_ms=s.cnn_ms, lstm_ms=s.lstm_ms, infer_fps=s.infer_fps,
                camera_fps=cam_fps, motion=s.motion, windows=s.windows, segments=segs,
                timeline=list(self.timeline)[-240:], classes=self.classes, names=self.names,
                device=str(self.device), paused=self.paused,
            )

    # ---- worker ---------------------------------------------------------------------------
    def _run(self):
        # cuDNN / cuBLAS handles are created per thread: warm them up here, not on the first frame
        try:
            S = self.cfg.img_size
            self._encode(np.zeros((S, S, 3), np.uint8))
            self._head(torch.zeros(1, self.cfg.seq_len, FEATURE_DIM, device=self.device))
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                t, crop = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._step(t, crop)
            except Exception as e:  # keep the worker alive whatever happens
                with self._lock:
                    self.state.status = f"ERROR: {e}"

    def _step(self, t: float, crop: np.ndarray):
        small = cv2.cvtColor(cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
        motion = float(np.abs(small - self._prev_small).mean()) if self._prev_small is not None else 0.0
        self._prev_small = small

        sync = self.device.type == "cuda"
        t0 = time.perf_counter()
        f = self._encode(crop)
        if sync: torch.cuda.synchronize()
        cnn_ms = (time.perf_counter() - t0) * 1000

        with self._lock:
            if self.paused:
                return
            self.feats.append(f[0])
            self.motion_hist.append(motion)
            self._since_infer += 1
            fill = len(self.feats)
            self.state.buffer_fill = fill
            self.state.cnn_ms = 0.8 * self.state.cnn_ms + 0.2 * cnn_ms if self.state.cnn_ms else cnn_ms
            if fill < self.cfg.seq_len:
                self.state.status = "BUFFERING"
                return
            if self._since_infer < self.cfg.infer_every:
                return
            self._since_infer = 0
            seq = torch.stack(list(self.feats))[None]

        t1 = time.perf_counter()
        probs = self._head(seq)[0].cpu().numpy()
        lstm_ms = (time.perf_counter() - t1) * 1000
        self._update(t, probs, cnn_ms, lstm_ms)

    def _update(self, t: float, probs: np.ndarray, cnn_ms: float, lstm_ms: float):
        cfg = self.cfg
        with self._lock:
            a = cfg.smoothing_alpha
            self.ema = probs if self.ema is None else a * probs + (1 - a) * self.ema
            k = int(self.ema.argmax())
            conf = float(self.ema[k])
            motion = float(np.mean(self.motion_hist)) if self.motion_hist else 0.0
            if cfg.motion_gate and motion < cfg.motion_threshold:
                cand, status = UNKNOWN, "NO SIGNIFICANT MOTION"
            elif conf < cfg.confidence_threshold:
                cand, status = UNKNOWN, "UNCERTAIN"
            else:
                cand, status = self.classes[k], "RECOGNIZING"

            # hysteresis: switch label after 2 agreeing windows (or immediately if very confident)
            if self.committed is None or cand == self.committed or conf > 0.9:
                self.committed, self.pending = cand, None
            else:
                if self.pending and self.pending[0] == cand:
                    self.pending = (cand, self.pending[1] + 1)
                else:
                    self.pending = (cand, 1)
                if self.pending[1] >= 2:
                    self.committed, self.pending = cand, None

            label = self.committed
            rel = t - self.t0
            if self.segments and self.segments[-1].label == label:
                seg = self.segments[-1]; seg.end = rel
            else:
                seg = Segment(label, rel, rel); self.segments.append(seg)
            seg.conf_sum += conf; seg.n += 1
            self.timeline.append((rel, label, conf, self.ema.tolist()))

            self._inf_times.append(time.time())
            it = self._inf_times
            s = self.state
            s.label = label if label == UNKNOWN else self.names.get(label, label)
            s.confidence, s.status, s.motion = conf, status, motion
            s.probs, s.raw_probs = self.ema.tolist(), probs.tolist()
            s.lstm_ms = lstm_ms
            self._lat.append(cnn_ms * cfg.infer_every + lstm_ms)
            s.latency_ms = float(np.mean(self._lat))
            s.infer_fps = (len(it) - 1) / (it[-1] - it[0]) if len(it) > 2 and it[-1] > it[0] else 0.0
            s.windows += 1


# ======================================================================================
# Offline video analysis
# ======================================================================================
SUPPORTED_EXT = {".mp4", ".avi", ".mov"}


def probe_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("The file could not be opened as a video (unsupported codec or corrupt file).")
    info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 25.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise ValueError("No decodable frames were found in this file.")
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    return info


@torch.no_grad()
def analyze_video(path: str, cfg: InferenceConfig | None = None, model_bundle=None,
                  progress: Callable[[float, str], None] | None = None, max_seconds: float = 600,
                  thumb_width: int = 320) -> dict:
    """Run the model over a whole video with sliding temporal windows."""
    model, meta, device = model_bundle or load_model()
    cfg = cfg or InferenceConfig()
    L, S, fps_s = meta["inference"]["seq_len"], meta["inference"]["img_size"], meta["inference"]["sample_fps"]
    classes, names = meta["classes"], meta.get("display_names", {})
    info = probe_video(path)
    pre = Preprocessor(S, device)
    cap = cv2.VideoCapture(path)
    src_fps = info["fps"]
    step = src_fps / fps_s
    feats, times, thumbs, motions = [], [], [], []
    batch, batch_t = [], []
    idx, next_t, prev_small = 0, 0.0, None
    t_start = time.perf_counter()
    cnn_time = 0.0
    total = max(info["frames"], 1)

    def flush():
        nonlocal cnn_time
        if not batch:
            return
        t0 = time.perf_counter()
        n = len(batch)
        padded_batch = batch + [batch[-1]] * (32 - n)  # fixed shape -> no per-shape kernel setup
        feats.append(_features(model, pre.to_tensor(padded_batch), device)[:n].cpu())
        if device.type == "cuda": torch.cuda.synchronize()
        cnn_time += time.perf_counter() - t0
        times.extend(batch_t); batch.clear(); batch_t.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx / src_fps > max_seconds:
            break
        if idx >= next_t:
            crop = pre.crop(frame)
            batch.append(crop); batch_t.append(idx / src_fps)
            small = cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_RGB2GRAY).astype(np.float32)
            motions.append(float(np.abs(small - prev_small).mean()) if prev_small is not None else 0.0)
            prev_small = small
            h, w = frame.shape[:2]
            thumbs.append(cv2.cvtColor(cv2.resize(frame, (thumb_width, round(h * thumb_width / w)),
                                                  interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            next_t += step
            if len(batch) == 32:
                flush()
                if progress: progress(min(idx / total, 1.0), f"Extracting MobileNetV3 features — {len(times)} frames")
        idx += 1
    flush()
    cap.release()
    if not times:
        raise ValueError("No frames could be decoded from this video.")

    F = torch.cat(feats)                      # (N,960)
    N = F.shape[0]
    padded = N < L
    if padded:
        F = torch.cat([F, F[-1:].expand(L - N, -1)])
    starts = list(range(0, max(F.shape[0] - L, 0) + 1, cfg.video_window_stride))
    if starts[-1] != F.shape[0] - L:
        starts.append(F.shape[0] - L)
    if progress: progress(1.0, f"Running LSTM over {len(starts)} temporal windows")
    t0 = time.perf_counter()
    probs = []
    for i in range(0, len(starts), 64):
        sel = starts[i:i + 64]
        chunk = torch.stack([F[s:s + L] for s in sel + [sel[-1]] * (64 - len(sel))]).to(device)
        probs.append(_classify(model, chunk, device)[:len(sel)].cpu())
    if device.type == "cuda": torch.cuda.synchronize()
    lstm_time = time.perf_counter() - t0
    P = torch.cat(probs).numpy()

    times_arr = np.array(times + [times[-1]] * (F.shape[0] - N))
    windows, ema = [], None
    for s, p in zip(starts, P):
        ema = p if ema is None else cfg.smoothing_alpha * p + (1 - cfg.smoothing_alpha) * ema
        k = int(ema.argmax()); conf = float(ema[k])
        mot = float(np.mean(motions[s:s + L])) if motions[s:s + L] else 0.0
        if cfg.motion_gate and mot < cfg.motion_threshold:
            lab = UNKNOWN
        elif conf < cfg.confidence_threshold:
            lab = UNKNOWN
        else:
            lab = classes[k]
        windows.append({"t_start": float(times_arr[s]), "t_end": float(times_arr[min(s + L - 1, len(times_arr) - 1)]),
                        "t_center": float(times_arr[min(s + L // 2, len(times_arr) - 1)]),
                        "probs": p.tolist(), "smoothed": ema.tolist(), "label": lab,
                        "label_raw": classes[int(p.argmax())], "confidence": conf, "motion": mot})
    # merge consecutive identical labels into segments
    segments = []
    for w in windows:
        if segments and segments[-1]["label"] == w["label"]:
            segments[-1]["end"] = w["t_end"]; segments[-1]["confs"].append(w["confidence"])
        else:
            segments.append({"label": w["label"], "start": w["t_start"] if not segments else segments[-1]["end"],
                             "end": w["t_end"], "confs": [w["confidence"]]})
    for g in segments:
        g["confidence"] = float(np.mean(g.pop("confs")))
    mean_p = P.mean(0)
    k = int(mean_p.argmax())
    elapsed = time.perf_counter() - t_start
    return {
        "classes": classes, "names": names, "windows": windows, "segments": segments,
        "overall": {"label": classes[k], "confidence": float(mean_p[k]), "probs": mean_p.tolist()},
        "thumbs": thumbs, "thumb_times": times,
        "info": {**info, "sampled_frames": N, "sample_fps": fps_s, "windows": len(starts), "seq_len": L,
                 "padded_short_video": padded, "processing_s": elapsed, "cnn_s": cnn_time, "lstm_s": lstm_time,
                 "realtime_factor": (times[-1] + 1 / fps_s) / elapsed if elapsed else 0.0,
                 "device": str(device)},
    }
