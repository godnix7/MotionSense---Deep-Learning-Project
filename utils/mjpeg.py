"""Minimal MJPEG stream server for the live video panel.

Pushing every camera frame through Streamlit (a script rerun per frame) costs CPU time under
Python's GIL and starves the capture and inference threads. Instead, the annotated frames
(already JPEG-encoded by the capture thread) are streamed as multipart/x-mixed-replace, which
browsers render natively in an <img> tag at full camera frame rate. Streamlit then only
refreshes the prediction panel a few times per second.

The server binds to 127.0.0.1 only: the stream is never reachable from the network.
"""
from __future__ import annotations

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_sources: dict[str, object] = {}       # stream key -> object with latest_jpeg() and .running
_server: ThreadingHTTPServer | None = None
_lock = threading.Lock()
BOUNDARY = b"msframe"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass

    def do_GET(self):
        key = self.path.strip("/").split("?")[0].removeprefix("stream/")
        src = _sources.get(key)
        if src is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        last = None
        try:
            while _sources.get(key) is src and getattr(src, "running", False):
                jpg = src.latest_jpeg()
                if jpg is None or jpg is last:
                    time.sleep(0.008)
                    continue
                last = jpg
                self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (ConnectionError, OSError):
            pass  # browser navigated away / stream replaced


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_server() -> int:
    global _server
    with _lock:
        if _server is None:
            _server = ThreadingHTTPServer(("127.0.0.1", _free_port()), _Handler)
            _server.daemon_threads = True
            threading.Thread(target=_server.serve_forever, daemon=True, name="motionsense-mjpeg").start()
        return _server.server_address[1]


def publish(key: str, source) -> str:
    """Register a frame source and return the stream URL for an <img> tag."""
    port = ensure_server()
    _sources[key] = source
    return f"http://127.0.0.1:{port}/stream/{key}?t={int(time.time() * 1000)}"


def unpublish(key: str) -> None:
    _sources.pop(key, None)
