import base64
import os
from pathlib import Path
from secrets import compare_digest

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app import app, recognizer
from backend.vision.qipedc_live import indexed_entry_ids


@app.get("/healthz", include_in_schema=False)
def healthz():
    ready = bool(getattr(recognizer.sstcn_model, "has_weights", False))
    return JSONResponse(
        {"ready": ready, "qipedc_indexed_count": len(indexed_entry_ids())},
        status_code=200 if ready else 503,
    )


preview_password = os.environ.get("VSL_PREVIEW_PASSWORD")

if preview_password:
    preview_credentials = base64.b64encode(f"vsl:{preview_password}".encode())

    @app.middleware("http")
    async def require_preview_login(request: Request, call_next):
        if request.url.path == "/healthz" and request.method == "GET":
            return await call_next(request)

        scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "basic" or not compare_digest(credentials.encode(), preview_credentials):
            return JSONResponse(
                {"detail": "Đăng nhập để mở bản thử nghiệm VSL."},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="VSL Connect", charset="UTF-8"'},
            )
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Yêu cầu phải đến từ trang VSL."}, status_code=403)
        return await call_next(request)

DIST_DIR = Path(__file__).resolve().parents[1] / "dist"
app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
