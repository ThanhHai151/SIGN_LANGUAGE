from pathlib import Path

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app import app, recognizer


@app.get("/healthz", include_in_schema=False)
def healthz():
    ready = bool(getattr(recognizer.sstcn_model, "has_weights", False))
    return JSONResponse({"ready": ready}, status_code=200 if ready else 503)


DIST_DIR = Path(__file__).resolve().parents[1] / "dist"
app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
