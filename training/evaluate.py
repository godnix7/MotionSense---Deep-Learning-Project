"""Evaluate the trained model on the official UCF101 TEST split.

Reports clip-level and video-level accuracy, macro precision / recall / F1, the confusion
matrix, a per-class report, and measured inference latency / throughput on this machine.

Usage:  python -m training.evaluate [--clips 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import VideoClipDataset, load_manifest, normalize  # noqa: E402
from models.inference import load_model  # noqa: E402
from training.metrics import compute_metrics  # noqa: E402
from training.train import evaluate_loader  # noqa: E402
from utils.config import CHARTS_DIR, TEST_METRICS_PATH, TrainConfig  # noqa: E402


@torch.no_grad()
def benchmark(model, device, seq_len, img_size, runs=50):
    """Latency of one full window (seq_len frames through CNN + LSTM) and of the incremental
    live step (1 new frame through CNN + LSTM over cached features)."""
    x = torch.randn(1, seq_len, 3, img_size, img_size, device=device)
    f = torch.randn(1, seq_len, 960, device=device)
    one = torch.randn(1, 3, img_size, img_size, device=device)
    amp = device.type == "cuda"

    def timeit(fn):
        for _ in range(10): fn()
        if amp: torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(runs): fn()
        if amp: torch.cuda.synchronize()
        return (time.perf_counter() - t) / runs * 1000

    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        full = timeit(lambda: model(x))
        inc = timeit(lambda: model.classify_sequence(torch.cat([f[:, 1:], model.extract_features(one)[:, None].float()], 1)))
    return {"full_window_ms": full, "full_window_fps": 1000 / full,
            "incremental_step_ms": inc, "incremental_steps_per_s": 1000 / inc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=3, help="clips per test video (video-level = mean softmax)")
    args = ap.parse_args()

    model, meta, device = load_model()
    classes = meta["classes"]
    cfg = TrainConfig(**{k: v for k, v in meta["train_config"].items() if k in TrainConfig.__dataclass_fields__})
    df = load_manifest()
    test = df[(df.subset == "test") & df.class_name.isin(classes)].reset_index(drop=True)
    ds = VideoClipDataset(test, cfg, train=False, eval_clips=args.clips)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    print(f"Evaluating on {len(test)} official test videos ({len(ds)} clips) on {device} ...")
    t0 = time.time()
    loss, clip_acc, y_true, y_pred = evaluate_loader(model, loader, device, len(classes))
    m = compute_metrics(y_true, y_pred, classes)
    m.update({
        "split": f"UCF101 official testlist0{cfg.split}",
        "test_loss_clip": loss, "clip_accuracy": clip_acc, "clips_per_video": args.clips,
        "eval_time_s": time.time() - t0,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "latency": benchmark(model, device, cfg.seq_len, cfg.img_size),
        "classes": classes,
        "checkpoint_epoch": meta.get("epoch"),
    })
    TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_METRICS_PATH.write_text(json.dumps(m, indent=2))

    # static chart exports (the app renders interactive Plotly versions of the same data)
    try:
        from utils.visualization import export_static_charts
        export_static_charts(CHARTS_DIR)
    except Exception as e:  # charts are a convenience; metrics are already saved
        print("chart export skipped:", e)

    print(f"\nVideo-level accuracy : {m['accuracy']:.4f}")
    print(f"Clip-level accuracy  : {clip_acc:.4f}")
    print(f"Macro precision      : {m['precision_macro']:.4f}")
    print(f"Macro recall         : {m['recall_macro']:.4f}")
    print(f"Macro F1             : {m['f1_macro']:.4f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(np.array(m["confusion_matrix"]))
    print("Latency:", json.dumps(m["latency"], indent=1))
    print(f"Saved {TEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
