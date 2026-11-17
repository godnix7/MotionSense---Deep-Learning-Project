"""MotionSense — Motion Intelligence Lab.   Launch with:  streamlit run app.py"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Motion Intelligence Lab · MotionSense", page_icon=str(ROOT / "static" / "favicon.png"),
                   layout="wide", initial_sidebar_state="collapsed")

from interface.ui import footer, inject_css, start_model_loading  # noqa: E402

inject_css()


def _page(module_name: str, key: str):
    def run():
        # Page modules are imported on first visit: PyTorch/torchvision (~5 s) are only needed by the
        # inference pages, so the Overview renders without waiting for them.
        if key != "live" and "interface.live_recognition" in sys.modules:
            sys.modules["interface.live_recognition"].stop_sources()  # release the camera when leaving Live
        importlib.import_module(f"interface.{module_name}").render()
    run.__name__ = f"page_{key}"
    return run


pages = [
    st.Page(_page("overview", "overview"), title="Overview", url_path="overview", default=True),
    st.Page(_page("live_recognition", "live"), title="Live Recognition", url_path="live"),
    st.Page(_page("video_analysis", "video"), title="Video Analysis", url_path="video"),
    st.Page(_page("model_insights", "insights"), title="Model Insights", url_path="insights"),
    st.Page(_page("system_info", "system"), title="System", url_path="system"),
]
nav = st.navigation(pages, position="top")
nav.run()
footer()
# Start loading the GPU model only after the first page has been sent to the browser.
start_model_loading()
