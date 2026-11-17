"""PAGE 3 — Video Analysis (upload or demo samples)."""
from __future__ import annotations

import hashlib
import html
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from interface.ui import missing_model_notice, model_or_none, page_header, pill, section, strip
from models.inference import SUPPORTED_EXT, UNKNOWN, analyze_video
from utils.config import InferenceConfig, test_videos, video_label
from utils.visualization import color_map, prob_bars, probability_timeline, segment_timeline

MAX_MB = 400


def _name(lab, names, classes):
    return names.get(lab, lab) if lab in classes else "Uncertain / unknown"


@st.cache_data(show_spinner=False, max_entries=6)
def _analyze_cached(path: str, digest: str, thr: float, alpha: float, gate: bool, motion_thr: float, stride: int):
    from interface.ui import get_model_bundle
    cfg = InferenceConfig(confidence_threshold=thr, smoothing_alpha=alpha, motion_gate=gate,
                          motion_threshold=motion_thr, video_window_stride=stride)
    bar = st.progress(0.0, text="Decoding video …")
    res = analyze_video(path, cfg, get_model_bundle(), progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
    bar.empty()
    return res


def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def render():
    page_header("Video analysis", "Recorded Video Analysis",
                "Upload an MP4, AVI or MOV file. The video is sampled at the training frame rate, encoded by MobileNetV3, "
                "and classified by the LSTM over sliding temporal windows.")
    bundle, err = model_or_none()
    if err:
        missing_model_notice(err)
        return

    st.write("")
    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        src = st.segmented_control("Input", ["Upload a video", "Demo samples"], default="Upload a video",
                                   key="va_src") or "Upload a video"
        path, data = None, None
        if src == "Upload a video":
            up = st.file_uploader("Video file", type=["mp4", "avi", "mov"], label_visibility="collapsed")
            if up is not None:
                ext = Path(up.name).suffix.lower()
                if ext not in SUPPORTED_EXT:
                    st.error(f"Unsupported file type '{ext}'. Supported: MP4, AVI, MOV.")
                elif up.size > MAX_MB * 1024 * 1024:
                    st.error(f"File is larger than {MAX_MB} MB.")
                else:
                    data = up.getvalue()
                    tmp = Path(tempfile.gettempdir()) / f"motionsense_{_digest(data)}{ext}"
                    if not tmp.exists():
                        tmp.write_bytes(data)
                    path = str(tmp)
        else:
            samples = test_videos()
            if not samples:
                st.info("No demo samples found. Create them with  python -m data.samples")
            else:
                pick = st.selectbox("Sample (UCF101 test split)", samples,
                                    format_func=video_label)
                path = str(pick)
                data = Path(path).read_bytes()
                st.caption("Sample clips come from the official test split — the network never saw them during training.")
    with c2:
        with st.expander("Analysis parameters", expanded=False):
            thr = st.slider("Confidence threshold", 0.30, 0.95, 0.60, 0.05)
            alpha = st.slider("Smoothing α (EMA across windows)", 0.1, 1.0, 0.40, 0.05)
            stride = st.slider("Window stride (sampled frames)", 1, 16, 4)
            gate = st.toggle("Motion gate", True)
            mthr = st.slider("Motion threshold", 0.0, 8.0, 0.8, 0.1, disabled=not gate)

    if path is None:
        st.markdown('<div class="ms-panel ms-note" style="margin-top:16px">Select or upload a video to begin. '
                    'Activities are predicted for every ~2-second window; the overall prediction averages all windows.</div>',
                    unsafe_allow_html=True)
        return

    try:
        res = _analyze_cached(path, _digest(data), thr, alpha, gate, mthr, stride)
    except ValueError as e:
        st.error(f"This file could not be processed: {e}")
        return
    except Exception as e:  # never crash the page during a demo
        st.error(f"Analysis failed: {e}")
        return

    classes, names, info = res["classes"], res["names"], res["info"]
    cmap = color_map(classes)
    ov = res["overall"]

    # ---- summary --------------------------------------------------------------------------
    section("01", "Result", f"{info['windows']} windows · {info['sampled_frames']} sampled frames")
    known = [w for w in res["windows"] if w["label"] != UNKNOWN]
    n_win = len(res["windows"])
    # activities that occupy a meaningful share of the video (>= 10% of windows)
    share = {l: sum(w["label"] == l for w in known) / n_win for l in set(w["label"] for w in known)}
    present = sorted([l for l, s in share.items() if s >= 0.10], key=lambda l: next(
        i for i, w in enumerate(res["windows"]) if w["label"] == l))
    if len(present) >= 2:  # multi-activity video: a single "overall" label would be meaningless
        first = ("Activities detected", str(len(present)), "", " → ".join(names.get(l, l) for l in present), True, True)
        second = ("Longest activity", html.escape(_name(max(present, key=share.get), names, classes)), "",
                  f"{max(share.values()):.0%} of windows · see timeline below", True)
    elif present:
        dominant = present[0]
        first = ("Predicted activity", html.escape(_name(dominant, names, classes)), "",
                 f"{share.get(dominant, 0):.0%} of windows agree", True)
        second = ("Mean probability", f"{ov['confidence']:.1%}", "", f"averaged over all {n_win} windows")
    else:  # every window below threshold: report the mean-probability argmax, flagged as low confidence
        first = ("Predicted activity", html.escape(names.get(ov["label"], ov["label"])), "",
                 "low confidence · no window passed the threshold", True)
        second = ("Mean probability", f"{ov['confidence']:.1%}", "", f"averaged over all {n_win} windows")
    st.markdown(strip([
        first, second,
        ("Duration", f"{info['duration']:.1f}", "s", f"{info['width']}×{info['height']} @ {info['fps']:.1f} fps"),
        ("Processing time", f"{info['processing_s']:.2f}", "s", f"{info['realtime_factor']:.1f}× real time · {info['device'].upper()}"),
    ]), unsafe_allow_html=True)
    if info["padded_short_video"]:
        st.warning(f"Video is shorter than one window ({info['seq_len']} frames at {info['sample_fps']:g} fps); "
                   "the last frame was repeated to fill the window.")

    # ---- side-by-side comparison --------------------------------------------------------
    section("02", "Original vs. prediction", "both videos are H.264 re-encodes that play in any browser")
    key = f"{_digest(data)}_{thr}_{alpha}_{gate}_{mthr}_{stride}"
    try:
        orig_mp4, ann_mp4 = _playable_videos(path, _digest(data), key, res, names, classes)
    except Exception as e:
        orig_mp4 = ann_mp4 = None
        st.caption(f"Video re-encoding unavailable ({e}); showing sampled frames instead.")
    v1, v2 = st.columns(2, gap="large")
    with v1:
        st.markdown('<div class="ms-label">Original video</div>', unsafe_allow_html=True)
        if orig_mp4:
            st.video(orig_mp4)
        else:
            _thumb_strip(res)
    with v2:
        st.markdown('<div class="ms-label">Model output (label + confidence per frame)</div>', unsafe_allow_html=True)
        if ann_mp4:
            st.video(ann_mp4)

    section("03", "Window inspector", "scrub through time to inspect the class probabilities of each window")
    i1, i2 = st.columns(2, gap="large")
    with i1:
        times = res["thumb_times"]
        t_max = float(times[-1]) if times else 0.0
        t = st.slider("Time (s)", 0.0, max(t_max, 0.1), 0.0, 0.125, key=f"scrub_{_digest(data)}")
        fi = int(np.argmin(np.abs(np.array(times) - t)))
        win = min(res["windows"], key=lambda w: abs(w["t_center"] - t))
        frame = res["thumbs"][fi].copy()
        lab = _name(win["label"], names, classes)
        col = cmap.get(win["label"], "#B4BCC7").lstrip("#")
        bgr = tuple(int(col[i:i + 2], 16) for i in (4, 2, 0))
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 30), (w, h), (20, 38, 61)[::-1], -1)
        cv2.rectangle(frame, (0, h - 30), (6, h), bgr[::-1], -1)
        cv2.putText(frame, f"{lab.upper()}  {win['confidence']:.0%}", (14, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        st.markdown(f'<div class="ms-label">Model view · window {win["t_start"]:.1f}–{win["t_end"]:.1f} s</div>',
                    unsafe_allow_html=True)
        st.image(frame, width="stretch")
    with i2:
        k = classes.index(win["label"]) if win["label"] in classes else None
        st.markdown('<div class="ms-label">Smoothed class probabilities for this window</div>', unsafe_allow_html=True)
        st.plotly_chart(prob_bars(win["smoothed"], classes, names, highlight=k, height=240), width="stretch",
                        config={"displayModeBar": False})

    # ---- timelines ----------------------------------------------------------------------
    section("04", "Activity timeline", "committed label per window after smoothing + threshold")
    st.plotly_chart(segment_timeline(res["segments"], classes, {**names, UNKNOWN: "Uncertain"}, height=140, cursor=t),
                    width="stretch", config={"displayModeBar": False})
    st.plotly_chart(probability_timeline([w["t_center"] for w in res["windows"]], [w["smoothed"] for w in res["windows"]],
                                         classes, names, threshold=thr, height=260, cursor=t),
                    width="stretch", config={"displayModeBar": False})

    # ---- tables -------------------------------------------------------------------------
    t1, t2 = st.columns([1, 1.2], gap="large")
    with t1:
        section("05", "Detected segments")
        seg_df = pd.DataFrame([{"Activity": _name(g["label"], names, classes), "Start (s)": round(g["start"], 2),
                                "End (s)": round(g["end"], 2), "Mean confidence": f"{g['confidence']:.1%}"}
                               for g in res["segments"]])
        st.dataframe(seg_df, hide_index=True, width="stretch")
    with t2:
        section("06", "Processing information")
        st.markdown(strip([("Sampled frames", info["sampled_frames"], "", f"at {info['sample_fps']:g} fps"),
                           ("Windows", info["windows"], "", f"{info['seq_len']} frames each, stride {stride}"),
                           ("CNN time", f"{info['cnn_s'] * 1000:.0f}", "ms", "MobileNetV3 (batched)"),
                           ("LSTM time", f"{info['lstm_s'] * 1000:.0f}", "ms", "all windows")], n=4),
                    unsafe_allow_html=True)
        with st.expander("Per-window predictions"):
            wdf = pd.DataFrame([{"t start": round(w["t_start"], 2), "t end": round(w["t_end"], 2),
                                 "label": _name(w["label"], names, classes),
                                 "raw argmax": names.get(w["label_raw"], w["label_raw"]),
                                 "confidence": round(w["confidence"], 3), "motion": round(w["motion"], 2)}
                                for w in res["windows"]])
            st.dataframe(wdf, hide_index=True, width="stretch", height=260)


def _playable_videos(path: str, digest: str, key: str, res: dict, names: dict, classes: list) -> tuple[str, str]:
    """Original + annotated H.264 MP4s, cached on disk per input file / analysis settings."""
    from utils.transcode import prediction_annotator, write_mp4
    tmp = Path(tempfile.gettempdir())
    orig = tmp / f"motionsense_{digest}_orig.mp4"
    ann = tmp / f"motionsense_{hashlib.sha1(key.encode()).hexdigest()[:16]}_ann.mp4"
    cmap = color_map(classes)
    if not orig.exists():
        with st.spinner("Re-encoding video for the browser …"):
            write_mp4(path, str(orig))
    if not ann.exists():
        with st.spinner("Rendering prediction overlay …"):
            write_mp4(path, str(ann), prediction_annotator(
                res["windows"], lambda l: _name(l, names, classes), lambda l: cmap.get(l, "#B4BCC7")))
    return str(orig), str(ann)


def _thumb_strip(res):
    thumbs = res["thumbs"]
    idx = np.linspace(0, len(thumbs) - 1, min(6, len(thumbs))).round().astype(int)
    cols = st.columns(3)
    for j, i in enumerate(idx):
        cols[j % 3].image(thumbs[i], caption=f"{res['thumb_times'][i]:.1f} s", width="stretch")
