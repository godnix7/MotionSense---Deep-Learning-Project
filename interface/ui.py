"""Shared interface components and design system for the Motion Intelligence Lab.

Typography: Inter for all text (set through the Streamlit theme, self-hosted fonts),
IBM Plex Mono only for numeric readouts, tensor shapes and instrument status.
"""
from __future__ import annotations

import html
import threading
import time

import streamlit as st

from utils.config import DISTRIBUTION_PATH, HISTORY_PATH, TEST_METRICS_PATH
from utils.visualization import load_json

CSS = """
<style>
:root{
  --bg:#F7F8FA; --navy:#14263D; --navy-2:#1B3150; --blue:#1769AA; --teal:#16A6A1; --ink:#202B38; --muted:#687586;
  --faint:#97A3B1; --ok:#239B75; --warn:#E6A23C; --err:#C2453D; --line:#DCE2E9; --line-2:#E8ECF1;
  --panel:#FFFFFF; --soft:#F2F5F8;
  --mono:'IBM Plex Mono', Consolas, monospace; --sans:Inter,'Helvetica Neue',Helvetica,Arial,sans-serif;
  --shadow:0 1px 2px rgba(20,38,61,.05);
}
.stApp{background:var(--bg)}
.block-container{padding-top:5.4rem !important;padding-bottom:4rem;max-width:1380px}

/* ---------- application bar (Streamlit header + top navigation) ---------- */
[data-testid="stHeader"]{background:var(--navy) !important;height:56px;border-bottom:1px solid #0E1C2E}
[data-testid="stHeader"]::before{content:"MOTIONSENSE";position:absolute;left:28px;top:13px;font:600 14px/1 var(--sans);
  letter-spacing:.16em;color:#FFFFFF}
[data-testid="stHeader"]::after{content:"Motion Intelligence Lab";position:absolute;left:28px;top:32px;
  font:500 10.5px/1 var(--mono);letter-spacing:.06em;color:#7FD3CF}
[data-testid="stHeader"] > div{padding-left:250px}
[data-testid="stTopNavLink"]{border-radius:0 !important;padding:0 14px !important;height:56px;background:transparent !important}
[data-testid="stTopNavLink"] p, [data-testid="stTopNavLink"] span:not([data-testid="stIconMaterial"]){
  font:500 13px/1 var(--sans) !important;color:#A9B8C9 !important;letter-spacing:.01em}
[data-testid="stTopNavLink"]:hover p{color:#FFFFFF !important}
[data-testid="stTopNavLink"][aria-current="page"], [data-testid="stTopNavLink"].active{box-shadow:inset 0 -2px 0 var(--teal)}
[data-testid="stTopNavLink"][aria-current="page"] p{color:#FFFFFF !important}
[data-testid="stHeader"] [data-testid="stIconMaterial"]{color:#A9B8C9}
[data-testid="stHeader"] button{color:#A9B8C9 !important}
[data-testid="stDecoration"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"],
[data-testid="stSidebar"], [data-testid="stStatusWidget"] .stDeployButton{display:none !important}
[data-testid="stIconMaterial"]{font-family:'Material Symbols Rounded' !important}

/* ---------- page + section headings ---------- */
.ms-page{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;padding-bottom:18px;
  border-bottom:1px solid var(--line);margin-bottom:6px}
.ms-crumb{font:500 12px/1 var(--sans);color:var(--muted);margin-bottom:10px}
.ms-crumb b{color:var(--blue);font-weight:600}
.ms-h1{font:700 26px/1.15 var(--sans);letter-spacing:-.02em;color:var(--ink);margin:0}
.ms-sub{font:400 14px/1.55 var(--sans);color:var(--muted);margin-top:8px;max-width:820px}
.ms-section{display:flex;align-items:baseline;gap:10px;margin:30px 0 12px}
.ms-section .idx{font:500 11px/1 var(--mono);color:var(--faint)}
.ms-section .t{font:600 15px/1.2 var(--sans);color:var(--ink)}
.ms-section .d{font:400 12.5px/1.2 var(--sans);color:var(--muted);margin-left:auto;text-align:right}
.ms-note{font:400 12.5px/1.55 var(--sans);color:var(--muted)}
.ms-mono{font-family:var(--mono)}
.ms-label{font:500 12px/1.3 var(--sans);color:var(--muted);margin-bottom:6px}
.ms-panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:16px 18px;box-shadow:var(--shadow)}

/* ---------- metric strip ---------- */
.ms-strip{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));background:var(--panel);
  border:1px solid var(--line);border-radius:4px;box-shadow:var(--shadow)}
.ms-cell{padding:14px 18px;border-right:1px solid var(--line-2);min-width:0}
.ms-cell:last-child{border-right:0}
.ms-cell .l{font:500 12px/1.3 var(--sans);color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ms-cell .v{font:500 22px/1.25 var(--mono);color:var(--ink);margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.01em}
.ms-cell .v.txt{font:600 18px/1.4 var(--sans);letter-spacing:-.01em}
.ms-cell .v small{font:400 12px var(--sans);color:var(--muted);margin-left:4px}
.ms-cell .s{font:400 12px/1.4 var(--sans);color:var(--faint);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* ---------- status pill ---------- */
.ms-pill{display:inline-flex;align-items:center;gap:7px;font:500 11px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;
  padding:6px 10px;border:1px solid var(--line);border-radius:3px;background:#fff;color:var(--ink);white-space:nowrap}
.ms-pill::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--faint)}
.ms-pill.ok::before{background:var(--ok)} .ms-pill.warn::before{background:var(--warn)} .ms-pill.err::before{background:var(--err)}
.ms-pill.live::before{background:var(--teal);box-shadow:0 0 0 3px rgba(22,166,161,.18);animation:mspulse 1.8s ease-in-out infinite}
@keyframes mspulse{0%,100%{opacity:1}50%{opacity:.35}}

/* ---------- overview hero ---------- */
.ms-hero{background:var(--navy);border-radius:4px;display:grid;grid-template-columns:minmax(0,1.55fr) minmax(0,1fr);
  overflow:hidden;position:relative}
.ms-hero .main{padding:34px 38px;position:relative}
.ms-hero .main::after{content:"";position:absolute;inset:0;pointer-events:none;opacity:.5;
  background-image:linear-gradient(rgba(255,255,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.03) 1px,transparent 1px);
  background-size:32px 32px}
.ms-hero .k{font:500 11px/1 var(--mono);letter-spacing:.14em;color:#7FD3CF;text-transform:uppercase}
.ms-hero h1{font:700 38px/1.05 var(--sans) !important;color:#FFFFFF !important;letter-spacing:-.025em;margin:14px 0 6px !important;padding:0 !important}
.ms-hero .s{font:500 16px/1.4 var(--sans);color:#C9D4E0;margin-bottom:18px}
.ms-hero p{color:#A9B8C9;font:400 14px/1.65 var(--sans);max-width:620px;margin:0}
.ms-hero .spec{background:var(--navy-2);padding:26px 30px;border-left:1px solid rgba(255,255,255,.06)}
.ms-hero .spec .h{font:500 11px/1 var(--mono);letter-spacing:.12em;color:#7FD3CF;text-transform:uppercase;margin-bottom:10px}
.ms-hero .row{display:flex;justify-content:space-between;gap:16px;padding:9px 0;border-bottom:1px solid rgba(255,255,255,.07)}
.ms-hero .row:last-child{border-bottom:0}
.ms-hero .row span:first-child{font:400 13px var(--sans);color:#93A6BC}
.ms-hero .row span:last-child{font:500 13px var(--mono);color:#FFFFFF;text-align:right}

/* ---------- activity tiles ---------- */
.ms-acts{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px}
.ms-act{background:#fff;border:1px solid var(--line);border-radius:4px;overflow:hidden;box-shadow:var(--shadow)}
.ms-act img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block}
.ms-act .b{padding:11px 13px 13px}
.ms-act .n{font:600 13.5px/1.3 var(--sans);color:var(--ink)}
.ms-act .c{display:flex;justify-content:space-between;font:400 11.5px/1 var(--mono);color:var(--muted);margin-top:6px}
.ms-act .bar{height:3px;background:var(--soft);margin-top:9px;border-radius:2px;overflow:hidden}
.ms-act .bar i{display:block;height:3px}

/* ---------- live console ---------- */
.st-key-console{background:var(--navy);border-radius:4px;padding:0 0 12px 0 !important;gap:0 !important}
.st-key-console [data-testid="stImage"] img{border-radius:0}
.ms-console-bar{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;position:relative;z-index:2;background:var(--navy);
  font:500 11px/1 var(--mono);letter-spacing:.06em;color:#93A6BC;text-transform:uppercase;border-bottom:1px solid rgba(255,255,255,.07)}
.ms-console-bar b{color:#FFFFFF;font-weight:500}
.ms-offline{aspect-ratio:4/3;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;
  color:#7C8FA6;font:500 12px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase}
.ms-offline .ring{width:54px;height:54px;border:1px solid #2C4361;border-radius:50%;display:flex;align-items:center;justify-content:center}
.ms-offline .ring i{width:12px;height:12px;border-radius:50%;background:#2C4361}
.ms-pred{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:20px 22px;box-shadow:var(--shadow);
  border-left:4px solid var(--blue)}
.ms-pred.unk{border-left-color:#B4BCC7} .ms-pred.idle{border-left-color:var(--line)}
.ms-pred .lab{font:700 32px/1.1 var(--sans);letter-spacing:-.02em;margin:6px 0 16px;color:var(--ink)}
.ms-pred.unk .lab{font-size:20px;color:var(--muted);letter-spacing:0}
.ms-pred.idle .lab{color:var(--faint)}
.ms-conf{display:flex;align-items:center;gap:14px}
.ms-conf .track{flex:1;height:6px;background:var(--soft);position:relative;border-radius:3px}
.ms-conf .fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px;transition:width .35s ease}
.ms-conf .thr{position:absolute;top:-5px;bottom:-5px;width:2px;background:var(--ink);opacity:.55}
.ms-conf .pct{font:500 20px/1 var(--mono);min-width:70px;text-align:right}
.ms-meta{font:400 11.5px/1.5 var(--mono);color:var(--faint);margin-top:12px}
.ms-probs .r{display:grid;grid-template-columns:132px 1fr 52px;align-items:center;gap:12px;padding:5px 0}
.ms-probs .r span:first-child{font:400 13px var(--sans);color:var(--muted)}
.ms-probs .r.top span:first-child{color:var(--ink);font-weight:600}
.ms-probs .r .t{height:5px;background:var(--soft);border-radius:3px;position:relative;overflow:hidden}
.ms-probs .r .t i{position:absolute;left:0;top:0;bottom:0;border-radius:3px;transition:width .3s}
.ms-probs .r span:last-child{font:500 12px var(--mono);text-align:right;color:var(--ink)}

/* ---------- history table ---------- */
.ms-hist{border:1px solid var(--line);background:#fff;border-radius:4px;max-height:300px;overflow:auto;box-shadow:var(--shadow)}
.ms-hist table{width:100%;border-collapse:collapse;font:400 13px var(--sans)}
.ms-hist th{position:sticky;top:0;background:var(--soft);font:500 12px var(--sans);color:var(--muted);text-align:left;
  padding:9px 14px;border-bottom:1px solid var(--line)}
.ms-hist td{padding:9px 14px;border-bottom:1px solid var(--line-2);color:var(--ink)}
.ms-hist td.m{font:400 12px var(--mono);color:var(--muted)}
.ms-sw{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:9px;vertical-align:middle}

/* ---------- widgets ---------- */
.stButton>button{font-weight:500;min-height:38px}
[data-testid="stExpander"] details{border-radius:4px;background:#fff}
[data-testid="stExpander"] summary p{font-weight:500}
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:4px;overflow:hidden;box-shadow:var(--shadow)}
[data-testid="stDataFrame"]{box-shadow:var(--shadow)}
[data-testid="stWidgetLabel"] p{font-weight:500;color:var(--muted)}
.ms-cell .s.wrap{white-space:normal;overflow:visible;line-height:1.45}
/* ---------- footer ---------- */
.ms-foot{margin-top:56px;border-top:1px solid var(--line);padding:22px 0 8px;display:grid;
  grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) minmax(0,1fr);gap:28px;font:400 12.5px/1.6 var(--sans);color:var(--muted)}
.ms-foot b{display:block;font:600 12px/1 var(--sans);color:var(--ink);margin-bottom:8px;letter-spacing:.02em}
.ms-foot a{color:var(--blue);text-decoration:none} .ms-foot a:hover{text-decoration:underline}
.ms-foot .brand{font:600 13px/1 var(--sans);letter-spacing:.14em;color:var(--ink);margin-bottom:8px}
/* ---------- steps ---------- */
.ms-steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}
.ms-step{background:#fff;border:1px solid var(--line);border-radius:4px;padding:16px 18px;box-shadow:var(--shadow)}
.ms-step .n{font:500 11px/1 var(--mono);color:var(--teal);margin-bottom:10px}
.ms-step .t{font:600 14px/1.3 var(--sans);color:var(--ink);margin-bottom:6px}
.ms-step .d{font:400 12.5px/1.55 var(--sans);color:var(--muted)}
@media (max-width:1200px){.ms-steps{grid-template-columns:repeat(2,minmax(0,1fr))}.ms-foot{grid-template-columns:1fr}.ms-acts{grid-template-columns:repeat(3,minmax(0,1fr))}.ms-hero{grid-template-columns:1fr}
  [data-testid="stHeader"] > div{padding-left:200px}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(crumb: str, title: str, subtitle: str = "", right_html: str = ""):
    st.markdown(
        f"""<div class="ms-page"><div><div class="ms-crumb">MotionSense&nbsp;&nbsp;/&nbsp;&nbsp;<b>{crumb}</b></div>
        <div class="ms-h1">{title}</div><div class="ms-sub">{subtitle}</div></div>
        <div style="padding-bottom:4px">{right_html}</div></div>""", unsafe_allow_html=True)


def section(idx: str, title: str, desc: str = ""):
    st.markdown(f'<div class="ms-section"><span class="idx">{idx}</span><span class="t">{title}</span>'
                f'<span class="d">{desc}</span></div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    return f'<span class="ms-pill {kind}">{html.escape(text)}</span>'


def strip(cells: list[tuple], n: int | None = None) -> str:
    """cells: (label, value, unit, sub[, text[, wrap]]) — text=True renders the value in Inter (for words);
    wrap=True lets the sub-line wrap instead of being truncated."""
    out = []
    for c in cells:
        label, value, unit, sub, text, wrap = (list(c) + ["", "", "", "", False, False])[:6]
        out.append(f'<div class="ms-cell"><div class="l">{label}</div>'
                   f'<div class="v{" txt" if text else ""}">{value}<small>{unit}</small></div>'
                   f'<div class="s{" wrap" if wrap else ""}">{sub}</div></div>')
    return f'<div class="ms-strip" style="--n:{n or len(cells)}">{"".join(out)}</div>'


_LOADER: dict | None = None
_LOADER_LOCK = threading.Lock()


def _model_loader(start: bool = True) -> dict | None:
    """Load the network (CUDA init, weights, TorchScript compile; ~15 s cold) on a background thread,
    once per server process. Pages that only need metadata never wait for it."""
    global _LOADER
    with _LOADER_LOCK:
        if _LOADER is None and start:
            state = {"bundle": None, "error": None, "done": threading.Event(), "t0": time.time()}

            def run():
                try:
                    from models.inference import load_model
                    state["bundle"] = load_model()
                except FileNotFoundError as e:
                    state["error"] = str(e)
                except Exception as e:  # corrupted checkpoint, CUDA failure, ...
                    state["error"] = f"The model checkpoint could not be loaded: {e}"
                finally:
                    state["done"].set()

            threading.Thread(target=run, daemon=True, name="motionsense-model-loader").start()
            _LOADER = state
        return _LOADER


def start_model_loading() -> None:
    _model_loader()


def model_status() -> str:
    """'ready' | 'loading' | 'error' | 'idle' (does not start loading)."""
    s = _model_loader(start=False)
    if s is None:
        return "idle"
    if not s["done"].is_set():
        return "loading"
    return "ready" if s["bundle"] is not None else "error"


def get_model_bundle():
    """Block until the model is loaded (used by pages that run inference)."""
    s = _model_loader()
    if not s["done"].is_set():
        with st.spinner("Loading the MobileNetV3-LSTM model onto the GPU (first start takes ~15 s) …"):
            s["done"].wait()
    if s["error"]:
        raise FileNotFoundError(s["error"])
    return s["bundle"]


def model_or_none():
    """Returns (bundle, error_message); waits for the background load if needed."""
    try:
        return get_model_bundle(), None
    except FileNotFoundError as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def model_meta() -> dict:
    """Checkpoint metadata (classes, config, parameter counts) without importing PyTorch:
    read from checkpoints/model_card.json (written by training), falling back to the checkpoint."""
    import json
    from utils.config import CHECKPOINT_PATH
    card = CHECKPOINT_PATH.with_name("model_card.json")
    if card.exists():
        return json.loads(card.read_text(encoding="utf-8"))
    if not CHECKPOINT_PATH.exists():
        return {}
    import torch
    meta = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    meta.pop("state_dict", None)
    return meta


def missing_model_notice(err: str):
    st.markdown(
        f"""<div class="ms-panel" style="border-left:4px solid var(--warn)">
        <div style="font-weight:600;font-size:14px">Model unavailable</div>
        <div class="ms-note" style="margin:4px 0 10px">The trained network is required for recognition. This page produces
        no predictions until a checkpoint exists.</div>
        <pre class="ms-mono" style="background:var(--soft);padding:10px 12px;font-size:12px;border:1px solid var(--line);
        border-radius:3px;white-space:pre-wrap;margin:0">{html.escape(err)}</pre></div>""", unsafe_allow_html=True)


def experiment_files():
    return load_json(HISTORY_PATH), load_json(TEST_METRICS_PATH), load_json(DISTRIBUTION_PATH)


@st.cache_resource(show_spinner=False)
def device_label() -> tuple[str, str]:
    """GPU name via NVML (fast, no PyTorch import); falls back to PyTorch."""
    try:
        import pynvml
        pynvml.nvmlInit()
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return f"CUDA · {name.decode() if isinstance(name, bytes) else name}", "ok"
    except Exception:
        import torch
        if torch.cuda.is_available():
            return f"CUDA · {torch.cuda.get_device_name(0)}", "ok"
        return "CPU (CUDA unavailable)", "warn"


REPO_URL = "https://github.com/godnix7/MotionSense---Deep-Learning-Project"


def footer():
    st.markdown(f"""<div class="ms-foot">
      <div><div class="brand">MOTIONSENSE</div>Real-time human activity recognition with a MobileNetV3-Large + LSTM
        network in PyTorch. Academic deep learning project.<br/><a href="{REPO_URL}" target="_blank">Source code on GitHub</a></div>
      <div><b>Dataset</b>UCF101 (Soomro, Zamir &amp; Shah, 2012), CRCV, University of Central Florida. Six classes,
        official split 1. Reported metrics are measured on the official test split.</div>
      <div><b>Stack</b>PyTorch · TorchScript · OpenCV · Streamlit · Plotly<br/>Inference runs locally; the camera feed
        never leaves this computer.</div>
    </div>""", unsafe_allow_html=True)
