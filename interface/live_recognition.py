"""PAGE 2 — Live Recognition (main demonstration screen)."""
from __future__ import annotations

import html
import time
import uuid

import cv2
import streamlit as st

from interface.ui import missing_model_notice, model_or_none, page_header, pill, section, strip
from models.inference import UNKNOWN, StreamingRecognizer
from utils import mjpeg
from utils.camera import LocalCamera, VideoStream, draw_overlay
from utils.config import InferenceConfig, test_videos, video_label
from utils.visualization import color_map, probability_timeline, segment_timeline

SRC_LOCAL = "Laptop camera"
SRC_WEBRTC = "Browser camera"
SRC_DEMO = "Demo video"


def stop_sources():
    src = st.session_state.get("live_source")
    if src is not None:
        src.stop()
        st.session_state.live_source = None
    if "stream_key" in st.session_state:
        mjpeg.unpublish(st.session_state.stream_key)


def _stream_key() -> str:
    if "stream_key" not in st.session_state:
        st.session_state.stream_key = uuid.uuid4().hex
    return st.session_state.stream_key


def _recognizer(bundle) -> StreamingRecognizer:
    rec = st.session_state.get("recognizer")
    if rec is None:
        rec = StreamingRecognizer(InferenceConfig(), bundle)
        st.session_state.recognizer = rec
    return rec


def _fmt_t(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def _status(snap: dict, active: bool) -> tuple[str, str]:
    raw = snap["status"]
    if snap["paused"]:
        return "Paused", "warn"
    if not active:
        return "Idle", ""
    if raw.startswith("ERROR"):
        return "Error", "err"
    if raw in ("BUFFERING", "IDLE"):
        return f"Buffering {snap['buffer_fill']}/{snap['seq_len']}", "warn"
    if raw == "RECOGNIZING":
        return "Recognizing", "live"
    return ("No motion" if raw.startswith("NO") else "Uncertain"), "warn"


def _prediction_panel(rec: StreamingRecognizer, active: bool):
    snap = rec.snapshot()
    cfg = rec.cfg
    classes, names = snap["classes"], snap["names"]
    cmap = color_map(classes)
    status, kind = _status(snap, active)

    label = snap["label"] if (active or snap["windows"]) and snap["label"] != "—" else "Waiting for video"
    is_unk = label == UNKNOWN
    klass = "unk" if is_unk else ("idle" if label == "Waiting for video" else "")
    conf = snap["confidence"] if not klass == "idle" else 0.0
    color = "#B4BCC7"
    for c in classes:
        if names.get(c, c) == label:
            color = cmap[c]
    shown = "Unknown / uncertain activity" if is_unk else label
    gate = f" · motion gate {cfg.motion_threshold:.1f}" if cfg.motion_gate else ""
    st.markdown(f"""
    <div class="ms-pred {klass}" style="{'' if klass else f'border-left-color:{color}'}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span class="ms-label" style="margin:0">Current activity</span>{pill(status, kind)}</div>
      <div class="lab">{html.escape(shown)}</div>
      <div class="ms-label">Confidence</div>
      <div class="ms-conf"><div class="track"><div class="fill" style="width:{conf * 100:.1f}%;background:{color}"></div>
        <div class="thr" style="left:{cfg.confidence_threshold * 100:.0f}%" title="Confidence threshold"></div></div>
        <div class="pct">{conf:.1%}</div></div>
      <div class="ms-meta">threshold {cfg.confidence_threshold:.2f} · motion {snap['motion']:.1f}{gate} · {snap['windows']} windows</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div style="height:12px"></div>' + strip([
        ("Latency", f"{snap['latency_ms']:.0f}", "ms", f"CNN {snap['cnn_ms']:.0f} · LSTM {snap['lstm_ms']:.1f} ms"),
        ("Inference rate", f"{snap['infer_fps']:.1f}", "win/s", f"every {cfg.infer_every} sampled frames"),
    ]) + '<div style="height:8px"></div>' + strip([
        ("Camera", f"{snap['camera_fps']:.0f}", "fps", f"sampled at {cfg.sample_fps:g} fps"),
        ("Temporal buffer", f"{snap['buffer_fill']}/{snap['seq_len']}", "frames", "GPU · CUDA" if snap["device"] == "cuda" else "CPU"),
    ]), unsafe_allow_html=True)
    probs = snap["probs"] or [0.0] * len(classes)
    top = max(range(len(probs)), key=lambda i: probs[i]) if any(probs) else -1
    rows = "".join(
        f'<div class="r{" top" if i == top else ""}"><span>{names.get(c, c)}</span>'
        f'<span class="t"><i style="width:{probs[i] * 100:.1f}%;background:{cmap[c] if i == top else "#C3CDD8"}"></i></span>'
        f'<span>{probs[i]:.1%}</span></div>' for i, c in enumerate(classes))
    st.markdown(f'<div class="ms-panel ms-probs" style="margin-top:12px;padding:14px 18px">'
                f'<div class="ms-label">Class probabilities</div>{rows}</div>', unsafe_allow_html=True)


def _history(rec: StreamingRecognizer):
    snap = rec.snapshot()
    classes, names = snap["classes"], snap["names"]
    cmap = color_map(classes)
    segs = snap["segments"]
    tl = snap["timeline"]
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        if tl:
            now = tl[-1][0]
            recent = [t for t in tl if t[0] >= now - 45]
            st.plotly_chart(probability_timeline([t[0] for t in recent], [t[3] for t in recent], classes, names,
                                                 threshold=rec.cfg.confidence_threshold, height=230),
                            width="stretch", config={"displayModeBar": False}, key="live_prob_tl")
            st.plotly_chart(segment_timeline([dict(s) for s in segs], classes,
                                             {**names, UNKNOWN: "Uncertain"}, x_title="Session time (s)", height=130),
                            width="stretch", config={"displayModeBar": False}, key="live_seg_tl")
        else:
            st.markdown('<div class="ms-panel ms-note" style="height:200px;display:flex;align-items:center;'
                        'justify-content:center">The timeline starts once the temporal buffer holds a full '
                        '16-frame window.</div>', unsafe_allow_html=True)
    with c2:
        rows = []
        for s in reversed(segs[-40:]):
            lab = s["label"]
            nm = names.get(lab, lab) if lab in classes else "Uncertain / unknown"
            rows.append(f'<tr><td><span class="ms-sw" style="background:{cmap.get(lab, "#B4BCC7")}"></span>{html.escape(nm)}</td>'
                        f'<td class="m">{_fmt_t(s["start"])}</td><td class="m">{max(s["end"] - s["start"], 0):.1f} s</td>'
                        f'<td class="m">{s["confidence"]:.0%}</td></tr>')
        body = "".join(rows) or '<tr><td colspan="4" class="m">No activity recorded yet.</td></tr>'
        st.markdown(f'<div class="ms-hist"><table><thead><tr><th>Activity</th><th>Start</th><th>Duration</th>'
                    f'<th>Confidence</th></tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def render():
    page_header("Live recognition", "Live Activity Recognition",
                "Perform an activity in front of the camera. The model classifies a rolling two-second window of motion "
                "and smooths its output across consecutive windows.")
    bundle, err = model_or_none()
    if err:
        missing_model_notice(err)
        return
    rec = _recognizer(bundle)

    # ---- toolbar ---------------------------------------------------------------------------
    samples = test_videos()
    st.write("")
    tb = st.columns([2.1, 1.9, 0.75, 0.75, 0.8, 1.0], vertical_alignment="bottom")
    with tb[0]:
        # ?source=demo | browser | camera selects the initial source (e.g. a camera-free demo link)
        initial = {"demo": SRC_DEMO, "browser": SRC_WEBRTC}.get(st.query_params.get("source", ""), SRC_LOCAL)
        mode = st.segmented_control("Video source", [SRC_LOCAL, SRC_WEBRTC, SRC_DEMO], default=initial,
                                    key="live_mode", width="stretch") or initial
    if st.session_state.get("_last_mode") != mode:
        stop_sources()
        st.session_state._last_mode = mode
    src = st.session_state.get("live_source")
    pick, cam_idx = None, 0
    with tb[1]:
        if mode == SRC_LOCAL:
            cam_idx = st.selectbox("Camera device", [0, 1, 2, 3], format_func=lambda i: f"Camera {i}", key="cam_idx")
        elif mode == SRC_DEMO:
            if samples:
                pick = st.selectbox("Sample (UCF101 test split)", samples, key="demo_pick",
                                    format_func=video_label)
            else:
                st.caption("No samples found. Run: python -m data.samples")
        else:
            st.caption("Use the Start control on the video panel; the browser asks for camera permission.")
    running = src is not None and src.running
    with tb[2]:
        if mode != SRC_WEBRTC and st.button("Start", type="primary", disabled=running, width="stretch"):
            rec.reset(); rec.resume()
            if mode == SRC_LOCAL:
                st.session_state.live_source = LocalCamera(rec, index=int(cam_idx)).start()
            elif pick is not None:
                st.session_state.live_source = VideoStream(rec, str(pick)).start()
            st.rerun()
    with tb[3]:
        if mode != SRC_WEBRTC and st.button("Stop", disabled=not running, width="stretch"):
            stop_sources(); st.rerun()
    with tb[4]:
        if rec.paused:
            if st.button("Resume", width="stretch"):
                rec.resume(); st.rerun()
        elif st.button("Pause", width="stretch"):
            rec.pause(); st.rerun()
    with tb[5]:
        if st.button("Reset history", width="stretch"):
            rec.reset(); st.rerun()

    with st.expander("Recognition parameters"):
        p = st.columns(4, gap="large")
        rec.cfg.confidence_threshold = p[0].slider("Confidence threshold", 0.30, 0.95, rec.cfg.confidence_threshold, 0.05,
                                                   help="Below this smoothed probability the system reports Unknown / uncertain.")
        rec.cfg.smoothing_alpha = p[1].slider("Smoothing (EMA α)", 0.1, 1.0, rec.cfg.smoothing_alpha, 0.05,
                                              help="Weight of the newest window. Lower = smoother but slower to react.")
        rec.cfg.infer_every = p[2].slider("Classify every N frames", 1, 8, rec.cfg.infer_every,
                                          help=f"Frames are sampled at {rec.cfg.sample_fps:g} fps.")
        rec.cfg.motion_gate = p[3].toggle("Motion gate", rec.cfg.motion_gate,
                                          help="Report Unknown when the scene is nearly static.")
        rec.cfg.motion_threshold = p[3].slider("Motion threshold", 0.0, 8.0, rec.cfg.motion_threshold, 0.1,
                                               disabled=not rec.cfg.motion_gate)

    # ---- console + prediction ----------------------------------------------------------------
    st.write("")
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        with st.container(key="console"):
            if mode == SRC_WEBRTC:
                _console_bar(f"Browser camera · WebRTC", "")
                _webrtc(rec)
            else:
                _threaded_view(mode, pick)
    with right:
        @st.fragment(run_every=0.3)
        def panel():
            s = st.session_state.get("live_source")
            act = (s is not None and s.running) or st.session_state.get("webrtc_playing", False)
            _prediction_panel(rec, act)
        panel()

    section("—", "Activity timeline", "smoothed class probabilities (last 45 s) · committed activity segments")

    @st.fragment(run_every=2.0)  # charts are relatively expensive to build; 0.5 Hz is plenty for a timeline
    def history():
        _history(rec)
    history()


def _console_bar(left: str, right: str):
    st.markdown(f'<div class="ms-console-bar"><span>{left}</span><span>{right}</span></div>', unsafe_allow_html=True)


def _threaded_view(mode: str, pick):
    s = st.session_state.get("live_source")
    name = "Laptop camera" if mode == SRC_LOCAL else ("Demo · " + video_label(pick) if pick else "Demo")

    # status bar: the only part that refreshes (1 Hz); also surfaces camera errors raised after Start
    @st.fragment(run_every=1.0)
    def bar():
        src = st.session_state.get("live_source")
        if src is None:
            _console_bar(f"<b>●</b>&nbsp; {html.escape(name)}", "offline")
            return
        if src.error:
            _console_bar(html.escape(name), "error")
            st.error(src.error)
            if mode == SRC_LOCAL:
                st.info("Try another camera, close apps using the webcam, or switch to Browser camera / Demo video.")
            return
        rec = st.session_state.get("recognizer")
        o = rec.overlay_state() if rec else {}
        _console_bar(f'<b style="color:#16A6A1">●</b>&nbsp; <b>{html.escape(name)}</b>',
                     f"{src.fps:.0f} fps · buffer {o.get('buffer_fill', 0)}/{o.get('seq_len', 16)}")
    bar()

    if s is None:
        st.markdown('<div class="ms-offline"><div class="ring"><i></i></div>Source offline · press Start</div>',
                    unsafe_allow_html=True)
        return
    # The video itself is an MJPEG stream from a localhost-only server: the browser decodes frames
    # natively at camera rate and Streamlit does no per-frame work (see utils/mjpeg.py).
    url = mjpeg.publish(_stream_key(), s)
    st.markdown(f'<img src="{url}" alt="Live camera" style="width:100%;aspect-ratio:4/3;display:block;'
                f'object-fit:contain;background:#0E1C2E"/>', unsafe_allow_html=True)


def _webrtc(rec: StreamingRecognizer) -> bool:
    try:
        import av
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("streamlit-webrtc is not installed. Use the Laptop camera source or  pip install streamlit-webrtc av")
        return False

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rec.push_frame(img, time.time())
        return av.VideoFrame.from_ndarray(draw_overlay(img, rec.overlay_state()), format="bgr24")

    try:
        ctx = webrtc_streamer(
            key="motionsense-webrtc", mode=WebRtcMode.SENDRECV, video_frame_callback=callback,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                                "frameRate": {"ideal": 30}}, "audio": False},
            rtc_configuration={"iceServers": []},  # localhost: host candidates are sufficient, no internet needed
            async_processing=True,
        )
    except Exception as e:  # component failure must not take the page down
        st.error(f"Browser camera could not be initialised: {e}. Switch to the Laptop camera source.")
        return False
    playing = bool(ctx.state.playing)
    if playing and not st.session_state.get("webrtc_playing"):
        rec.reset(); rec.resume()
    st.session_state.webrtc_playing = playing
    if not playing:
        st.caption("If the browser blocks the camera, allow access via the camera icon in the address bar.")
    return playing
