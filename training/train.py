"""Train MobileNetV3-Large + LSTM on the UCF101 six-class subset.

Usage:  python -m training.train [--epochs 15] [--batch-size 8] [--allow-cpu]

Training requires a CUDA GPU by default (pass --allow-cpu to override, very slow).
The best checkpoint (by validation macro-F1) is saved to checkpoints/motionsense_best.pt and
the per-epoch history to outputs/reports/training_history.json. The official test split is
NOT used here — run  python -m training.evaluate  afterwards.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import gpu_augment, make_loaders, normalize  # noqa: E402
from models.mobilenet_lstm import MobileNetLSTM, count_parameters  # noqa: E402
from training.metrics import compute_metrics  # noqa: E402
from utils.config import (CHECKPOINT_PATH, DISPLAY_NAMES, HISTORY_PATH, InferenceConfig,  # noqa: E402
                          TrainConfig)


def require_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__}, CUDA {torch.version.cuda})")
        return dev
    msg = (f"CUDA is not available (torch {torch.__version__}). Install the CUDA build:\n"
           "  pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128")
    if not allow_cpu:
        raise SystemExit(msg)
    print("WARNING:", msg, "\nContinuing on CPU because --allow-cpu was given.")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model, loader, device, num_classes, amp=True):
    """Clip-level loss/accuracy and video-level predictions (softmax averaged over clips)."""
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum, n, correct = 0.0, 0, 0
    vid_probs, vid_label = {}, {}
    for clips, labels, vids in loader:
        clips = normalize(clips.to(device, non_blocking=True))
        labels = labels.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(clips)
        logits = logits.float()
        loss_sum += crit(logits, labels).item() * len(labels)
        n += len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        probs = logits.softmax(1).cpu().numpy()
        for p, y, v in zip(probs, labels.cpu().numpy(), vids.numpy()):
            vid_probs.setdefault(int(v), np.zeros(num_classes)); vid_probs[int(v)] += p
            vid_label[int(v)] = int(y)
    keys = sorted(vid_probs)
    y_true = [vid_label[k] for k in keys]
    y_pred = [int(vid_probs[k].argmax()) for k in keys]
    return loss_sum / n, correct / n, y_true, y_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--seq-len", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--unfreeze", type=int, help="trailing MobileNetV3 blocks to fine-tune (0 = frozen)")
    ap.add_argument("--no-deltas", action="store_true", help="feed raw features only to the LSTM")
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--tag", default="", help="save to checkpoints/exp_<tag>.pt (experiments); empty = final model")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.unfreeze is not None: cfg.unfreeze_blocks = args.unfreeze
    if args.no_deltas: cfg.use_deltas = False
    if args.dropout is not None: cfg.dropout = args.dropout
    global CHECKPOINT_PATH, HISTORY_PATH
    if args.tag:
        CHECKPOINT_PATH = CHECKPOINT_PATH.parent / f"exp_{args.tag}.pt"
        HISTORY_PATH = HISTORY_PATH.parent / f"history_exp_{args.tag}.json"
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.seq_len: cfg.seq_len = args.seq_len
    if args.workers is not None: cfg.num_workers = args.workers
    set_seed(cfg.seed)
    device = require_device(args.allow_cpu)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, df = make_loaders(cfg)
    n_cls = len(cfg.classes)
    print(f"Train videos: {(df.subset=='train').sum()}  Val videos: {(df.subset=='val').sum()}  "
          f"(test videos held out: {(df.subset=='test').sum()})")

    model = MobileNetLSTM(n_cls, cfg.lstm_hidden, cfg.lstm_layers, cfg.dropout, pretrained=True,
                          unfreeze_blocks=cfg.unfreeze_blocks, use_deltas=cfg.use_deltas).to(device)
    params = count_parameters(model)
    print(f"Parameters: {params}")

    backbone_p = [p for p in model.features.parameters() if p.requires_grad]
    head_p = list(model.lstm.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": backbone_p, "lr": cfg.lr_backbone},
        {"params": head_p, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_loader)
    warm = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    history, best_f1 = [], -1.0
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, loss_sum, correct, n = time.time(), 0.0, 0, 0
        for step, (clips, labels, _) in enumerate(train_loader, 1):
            clips = normalize(gpu_augment(clips.to(device, non_blocking=True), cfg.img_size))
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(clips)
                loss = crit(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            loss_sum += loss.item() * len(labels); n += len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
        tr_loss, tr_acc = loss_sum / n, correct / n
        va_loss, va_acc, y_true, y_pred = evaluate_loader(model, val_loader, device, n_cls, cfg.amp)
        vm = compute_metrics(y_true, y_pred, cfg.classes)
        rec = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
               "val_clip_acc": va_acc, "val_video_acc": vm["accuracy"], "val_f1_macro": vm["f1_macro"],
               "lr_head": opt.param_groups[1]["lr"], "epoch_time_s": time.time() - t0}
        history.append(rec)
        improved = vm["f1_macro"] > best_f1
        if improved:
            best_f1 = vm["f1_macro"]
            torch.save({
                "state_dict": model.state_dict(),
                "classes": cfg.classes,
                "display_names": {c: DISPLAY_NAMES.get(c, c) for c in cfg.classes},
                "train_config": cfg.to_dict(),
                "inference": {"seq_len": cfg.seq_len, "img_size": cfg.img_size, "sample_fps": cfg.sample_fps},
                "epoch": epoch, "val_f1_macro": vm["f1_macro"], "val_video_acc": vm["accuracy"],
                "parameters": params,
                "torch_version": torch.__version__,
            }, CHECKPOINT_PATH)
            # lightweight metadata so the UI can describe the model without importing PyTorch
            card = {k: v for k, v in torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False).items()
                    if k != "state_dict"}
            card_name = "model_card.json" if not args.tag else f"{CHECKPOINT_PATH.stem}_card.json"
            CHECKPOINT_PATH.with_name(card_name).write_text(json.dumps(card, indent=2))
        print(f"epoch {epoch:02d}/{cfg.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} clip-acc {va_acc:.3f} video-acc {vm['accuracy']:.3f} "
              f"F1 {vm['f1_macro']:.3f} | {rec['epoch_time_s']:.0f}s {'*' if improved else ''}", flush=True)
        HISTORY_PATH.write_text(json.dumps({
            "config": cfg.to_dict(), "history": history, "best_val_f1_macro": best_f1,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "total_time_s": time.time() - t_start, "parameters": params,
        }, indent=2))
    print(f"Done in {(time.time()-t_start)/60:.1f} min. Best val macro-F1 {best_f1:.3f}. Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
