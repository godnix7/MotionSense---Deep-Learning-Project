"""PAGE 1 — Overview."""
from __future__ import annotations

import base64

import streamlit as st

from interface.diagram import architecture_svg
from interface.ui import device_label, experiment_files, model_meta, model_status, pill, section, strip
from utils.config import ASSETS_DIR, CLASSES, DISPLAY_NAMES, SAMPLE_FPS, SEQ_LEN
from utils.visualization import CLASS_COLORS


def _img_b64(path) -> str | None:
    try:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None


def _hero_html(status, compute, acc, f1, params, tc):
    st.markdown(f"""
    <div class="ms-hero">
      <div class="main">
        <div class="k">Deep learning · Computer vision</div>
        <h1>Motion Intelligence Lab</h1>
        <div class="s">Real-time human activity recognition</div>
        <p>MotionSense recognises human activities from a live camera or recorded video. MobileNetV3-Large encodes each
        frame, and an LSTM models how those features change across a {SEQ_LEN}-frame window
        (about {SEQ_LEN / SAMPLE_FPS:.0f} seconds). Every prediction is based on motion over time, never on a single
        image. The network is trained and evaluated on UCF101 with the official train/test protocol.</p>
      </div>
      <div class="spec">
        <div class="h">System status</div>
        <div class="row"><span>Model</span><span>{status}</span></div>
        <div class="row"><span>Compute</span><span>{compute}</span></div>
        <div class="row"><span>Test accuracy</span><span>{acc}</span></div>
        <div class="row"><span>Test macro-F1</span><span>{f1}</span></div>
        <div class="row"><span>Parameters</span><span>{f"{params / 1e6:.2f} M" if params else "—"}</span></div>
        <div class="row"><span>Benchmark</span><span>UCF101 split {tc.get("split", 1)}</span></div>
      </div>
    </div>""", unsafe_allow_html=True)


def render():
    meta = model_meta()
    state = model_status()
    bundle, err = (meta or None), (None if meta else "no checkpoint")
    hist, test, dist = experiment_files()
    dev, dev_kind = device_label()
    classes = meta.get("classes", CLASSES)
    tc = meta.get("train_config", {})

    acc = f"{test['accuracy']:.1%}" if test else "—"
    f1 = f"{test['f1_macro']:.3f}" if test else "—"
    params = meta.get("parameters", {}).get("total")
    gpu = dev.split("· ")[-1].replace("NVIDIA GeForce ", "")

    # The hero re-renders every 2 s while the model is still loading, so its status flips to "Ready" by itself.
    @st.fragment(run_every=2.0 if (meta and state != "ready") else None)
    def hero():
        cur = model_status()
        status = ({"ready": "Ready", "error": "Error"}.get(cur, "Loading onto GPU…")) if meta else "Not trained"
        _hero_html(status, gpu if dev_kind == "ok" else "CPU", acc, f1, params, tc)
    hero()
    status = ({"ready": "Ready", "error": "Error"}.get(state, "Loading")) if meta else "Not trained"

    if err:
        st.write("")
        st.warning("No trained checkpoint was found. Train the model first (see README); the live and video pages "
                   "are disabled until then.")

    # ---- architecture ---------------------------------------------------------------------
    section("01", "Model architecture", "MobileNetV3-Large → LSTM → classifier · PyTorch")
    st.markdown(f'<div class="ms-panel" style="padding:18px 10px 8px">'
                f'{architecture_svg(tc.get("seq_len", SEQ_LEN), tc.get("img_size", 160), tc.get("sample_fps", SAMPLE_FPS), tc.get("lstm_hidden", 256), len(classes))}'
                f'</div>', unsafe_allow_html=True)

    # ---- activities -------------------------------------------------------------------------
    section("02", "Supported activities", "frames are real UCF101 test-split samples · bar = test F1-score")
    rep = test["classification_report"] if test else {}
    tiles = []
    for i, c in enumerate(classes):
        img = _img_b64(ASSETS_DIR / f"{c}.jpg")
        f1c = rep.get(c, {}).get("f1-score")
        img_html = f'<img src="{img}"/>' if img else '<div style="aspect-ratio:4/3;background:#EEF2F6"></div>'
        tiles.append(f"""<div class="ms-act">{img_html}<div class="b">
            <div class="n">{DISPLAY_NAMES.get(c, c)}</div>
            <div class="c"><span>Test F1</span><span>{f"{f1c:.2f}" if f1c is not None else "—"}</span></div>
            <div class="bar"><i style="width:{(f1c or 0) * 100:.0f}%;background:{CLASS_COLORS[i % 6]}"></i></div></div></div>""")
    st.markdown(f'<div class="ms-acts">{"".join(tiles)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="ms-note" style="margin-top:10px">Movements outside these six classes, and low-confidence '
                'windows, are reported as <b>Unknown / uncertain activity</b> instead of being forced into a class.</div>',
                unsafe_allow_html=True)

    # ---- dataset + performance --------------------------------------------------------------
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("03", "Dataset", "UCF101 · CRCV, University of Central Florida")
        if dist:
            counts = dist["counts"]
            tr = sum(v["train"] for v in counts.values()); va = sum(v["val"] for v in counts.values())
            te = sum(v["test"] for v in counts.values())
            st.markdown(strip([("Classes", len(counts), "of 101", "six-class configuration"),
                               ("Training", tr, "videos", "official trainlist"),
                               ("Validation", va, "videos", f"groups g{dist['val_groups'][0]:02d}–g{dist['val_groups'][-1]:02d}"),
                               ("Test", te, "videos", f"official testlist0{dist['split']}")]), unsafe_allow_html=True)
        st.markdown("""<div class="ms-note" style="margin-top:12px">UCF101 contains 13,320 real-world videos in 101 action
            classes. Clips are grouped by actor and scene; the official split keeps whole groups in either training or
            test, and validation holds out further whole groups, so no person or scene appears in two subsets.</div>""",
                    unsafe_allow_html=True)
    with c2:
        section("04", "Inference performance", "benchmarked on this machine")
        lat = (test or {}).get("latency", {})
        st.markdown(strip([
            ("Full window", f"{lat['full_window_ms']:.1f}" if lat else "—", "ms", f"{SEQ_LEN} frames, CNN + LSTM"),
            ("Live step", f"{lat['incremental_step_ms']:.1f}" if lat else "—", "ms", "1 new frame + LSTM"),
            ("Throughput", f"{lat['incremental_steps_per_s']:.0f}" if lat else "—", "win/s", "incremental mode"),
            ("Precision", "FP16", "", "mixed-precision autocast"),
        ]), unsafe_allow_html=True)
        st.markdown(f'<div class="ms-note" style="margin-top:12px">Offline test metrics and live-camera behaviour are '
                    f'reported separately. Live FPS and latency are measured continuously on the Live Recognition page.'
                    f'</div><div style="margin-top:12px;display:flex;gap:8px">{pill("Model " + status.lower(), "ok" if status == "Ready" else "warn")}'
                    f'{pill(dev, dev_kind)}</div>', unsafe_allow_html=True)

    # ---- demonstration guide -----------------------------------------------------------------
    section("05", "Demonstration guide", "how to run a live demo")
    steps = [
        ("Open Live Recognition", "Choose <b>Laptop camera</b> and press <b>Start</b>. The status reads "
         "<i>Buffering</i> until the first 16-frame window (about 2 s) is collected."),
        ("Frame the whole body", "Stand 2–3 m from the camera with your full body visible and even lighting, "
         "similar to the UCF101 recordings."),
        ("Hold each activity", "Perform one exercise for 3–4 s. The label switches once two consecutive windows "
         "agree; standing still shows <i>Unknown / uncertain</i>."),
        ("No camera? Use a video", "Pick <b>Demo video</b> on the Live page, or upload a clip on Video Analysis. "
         "<i>All six activities</i> plays every class in sequence."),
    ]
    st.markdown('<div class="ms-steps">' + "".join(
        f'<div class="ms-step"><div class="n">STEP {i + 1:02d}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, (t, d) in enumerate(steps)) + "</div>", unsafe_allow_html=True)
