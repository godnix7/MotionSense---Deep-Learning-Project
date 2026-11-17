"""PAGE 5 — System Information."""
from __future__ import annotations

import platform

import pandas as pd
import psutil
import streamlit as st
import torch

from interface.ui import _model_loader, experiment_files, model_meta, model_status, page_header, section, strip


def _gpu_stats():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
        return {"util": util.gpu, "mem_used": mem.used / 2**30, "mem_total": mem.total / 2**30, "temp": temp, "power": power}
    except Exception:
        return None


def render():
    page_header("System", "System Information", "Runtime environment, model footprint and live resource usage.")
    loader = _model_loader()
    bundle = loader["bundle"] if model_status() == "ready" else None
    err = loader["error"] or ("The model is still loading onto the GPU; refresh in a few seconds."
                              if model_status() == "loading" else None)
    _, test, dist = experiment_files()
    cuda = torch.cuda.is_available()

    section("01", "Compute device")
    if not cuda:
        st.warning("CUDA is not available to PyTorch in this environment — inference is running on the CPU. "
                   "Install the CUDA build:  pip install torch==2.8.0 torchvision==0.23.0 --index-url "
                   "https://download.pytorch.org/whl/cu128")

    @st.fragment(run_every=2.0)
    def live_usage():
        g = _gpu_stats() if cuda else None
        vm = psutil.virtual_memory()
        cells = [
            ("Device", "CUDA" if cuda else "CPU", "", torch.cuda.get_device_name(0) if cuda else platform.processor()[:40]),
            ("GPU utilisation", f"{g['util']}" if g else "—", "%" if g else "", f"{g['temp']} °C · {g['power']:.0f} W" if g else "NVML unavailable"),
            ("GPU memory", f"{g['mem_used']:.2f}" if g else "—", f"/ {g['mem_total']:.1f} GB" if g else "",
             f"PyTorch allocated {torch.cuda.memory_allocated() / 2**20:.0f} MB" if cuda else ""),
            ("CPU utilisation", f"{psutil.cpu_percent(interval=0.2):.0f}", "%", f"{psutil.cpu_count(logical=True)} logical cores"),
            ("System RAM", f"{vm.used / 2**30:.1f}", f"/ {vm.total / 2**30:.0f} GB", f"{vm.percent:.0f}% used"),
        ]
        st.markdown(strip(cells), unsafe_allow_html=True)
        rec = st.session_state.get("recognizer")
        if rec is not None:
            s = rec.snapshot()
            st.write("")
            st.markdown(strip([
                ("Live latency", f"{s['latency_ms']:.1f}", "ms", "current session"),
                ("CNN / frame", f"{s['cnn_ms']:.1f}", "ms", "MobileNetV3"),
                ("LSTM / window", f"{s['lstm_ms']:.2f}", "ms", "LSTM + head"),
                ("Inference rate", f"{s['infer_fps']:.1f}", "win/s", ""),
                ("Camera FPS", f"{s['camera_fps']:.1f}", "fps", "0 when no source is running"),
            ]), unsafe_allow_html=True)
    live_usage()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("02", "Model")
        if bundle:
            model, meta, device = bundle
            p = meta.get("parameters", {})
            st.markdown(strip([("Total parameters", f"{p.get('total', 0) / 1e6:.2f}", "M", ""),
                               ("Backbone", f"{p.get('backbone', 0) / 1e6:.2f}", "M", "MobileNetV3-Large trunk"),
                               ("LSTM", f"{p.get('lstm', 0) / 1e6:.2f}", "M", ""),
                               ("FC head", f"{p.get('head', 0) / 1e3:.0f}", "K", "")]), unsafe_allow_html=True)
            rows = []
            for name, mod in model.named_children():
                if name in ("live_encoder", "live_head"):  # TorchScript copies for the live path, not extra parameters
                    continue
                n = sum(x.numel() for x in mod.parameters())
                rows.append({"Module": name, "Type": type(mod).__name__, "Parameters": f"{n:,}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            with st.expander("Full module structure"):
                st.code("\n".join(f"({n}): {m}" for n, m in model.named_children()
                                  if n not in ("live_encoder", "live_head")), language="text")
            st.caption(f"Checkpoint epoch {meta.get('epoch')} · val macro-F1 {meta.get('val_f1_macro', 0):.3f} · "
                       f"trained with torch {meta.get('torch_version')}")
        else:
            st.info(err)
    with c2:
        section("03", "Inference benchmark", "from training/evaluate.py")
        lat = (test or {}).get("latency")
        if lat:
            st.dataframe(pd.DataFrame([
                ("Full window (16 frames, CNN + LSTM)", f"{lat['full_window_ms']:.2f} ms", f"{lat['full_window_fps']:.0f} windows/s"),
                ("Incremental live step (1 frame + LSTM)", f"{lat['incremental_step_ms']:.2f} ms", f"{lat['incremental_steps_per_s']:.0f} steps/s"),
            ], columns=["Measurement", "Latency", "Throughput"]), hide_index=True, width="stretch")
            st.caption(f"Measured on {test.get('device')} with FP16 autocast, batch 1.")
        else:
            st.info("Run  python -m training.evaluate  to benchmark latency.")

        section("04", "Software environment")
        import cv2
        import streamlit
        vers = [("Python", platform.python_version()), ("PyTorch", torch.__version__),
                ("CUDA (PyTorch build)", torch.version.cuda or "none"), ("cuDNN", str(torch.backends.cudnn.version()) if cuda else "—"),
                ("OpenCV", cv2.__version__), ("Streamlit", streamlit.__version__), ("OS", f"{platform.system()} {platform.release()}")]
        st.dataframe(pd.DataFrame(vers, columns=["Component", "Version"]), hide_index=True, width="stretch")

        section("05", "Dataset")
        if dist:
            st.dataframe(pd.DataFrame([{"Class": c, **v} for c, v in dist["counts"].items()]), hide_index=True,
                         width="stretch")
            st.caption(f"UCF101 official split {dist['split']} · frames sampled at {dist['sample_fps']:g} fps · "
                       f"mean {dist['mean_cached_frames']:.0f} sampled frames per video.")
