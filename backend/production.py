import os
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request as UrlRequest, urlopen

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.app import app, recognizer
from backend.vision.qipedc_live import index_path, indexed_entry_ids

_qipedc_indexed_count = None
_self_ping_started = False


def _warm_qipedc_index():
    global _qipedc_indexed_count
    try:
        _qipedc_indexed_count = len(indexed_entry_ids())
    except Exception as exc:  # noqa: BLE001
        print("[production] Không tải nền được chỉ mục QIPEDC:", exc)


def _self_ping_loop(url: str, interval_sec: int):
    while True:
        time.sleep(interval_sec)
        try:
            req = UrlRequest(url, headers={"User-Agent": "vsl-keep-alive/1.0"})
            with urlopen(req, timeout=15) as resp:
                resp.read(128)
        except Exception:  # noqa: BLE001
            pass


@app.on_event("startup")
def start_production_background_tasks():
    global _self_ping_started
    if "pytest" in sys.modules or "unittest" in sys.modules:
        return
    threading.Thread(target=_warm_qipedc_index, name="qipedc-index-warm", daemon=True).start()

    external_url = (os.environ.get("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")
    enabled = os.environ.get("VSL_SELF_PING", "1").strip().lower() not in {"0", "false", "no", "off"}
    if external_url and enabled and not _self_ping_started:
        try:
            interval = max(60, int(os.environ.get("VSL_SELF_PING_INTERVAL_SEC", "600")))
        except ValueError:
            interval = 600
        _self_ping_started = True
        ping_url = f"{external_url}/healthz"
        threading.Thread(
            target=_self_ping_loop,
            args=(ping_url, interval),
            name="render-self-ping",
            daemon=True,
        ).start()
        print(f"[keep-alive] Tự ping {ping_url} mỗi {interval}s (VSL_SELF_PING=0 để tắt)")


@app.api_route("/healthz", methods=["GET", "HEAD"], include_in_schema=False)
def healthz(request: Request = None):
    ready = bool(getattr(recognizer.sstcn_model, "has_weights", False))
    status_code = 200 if ready else 503
    if request is not None and request.method == "HEAD":
        return Response(status_code=status_code)
    return JSONResponse(
        {
            "ready": ready,
            "qipedc_index_available": index_path() is not None,
            "qipedc_indexed_count": _qipedc_indexed_count,
            "commit": os.environ.get("RENDER_GIT_COMMIT"),
        },
        status_code=status_code,
    )


DIST_DIR = Path(__file__).resolve().parents[1] / "dist"
app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
