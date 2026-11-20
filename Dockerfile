FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html vite.config.js ./
COPY src/ ./src/
COPY public/ ./public/
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    VSL_PREFETCH=0 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libportaudio2 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-cloud.txt ./
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-cloud.txt
RUN python - <<'PY'
from hashlib import sha256
from pathlib import Path
from sysconfig import get_path
from urllib.request import urlopen

with urlopen("https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite", timeout=30) as response:
    model = response.read()
if sha256(model).hexdigest() != "1150dc68a713b80660b90ef46ce4e85c1c781bb88b6e3512cc64e6a685ba5588":
    raise RuntimeError("MediaPipe Pose Lite checksum mismatch")
target = Path(get_path("purelib")) / "mediapipe/modules/pose_landmark/pose_landmark_lite.tflite"
target.write_bytes(model)
PY
COPY --chown=1000:1000 backend/ ./backend/
COPY --from=frontend /build/dist/ ./dist/
ENV MPLCONFIGDIR=/tmp/matplotlib OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;1" \
    VSL_QIPEDC_REFERENCE_DIR=/app/backend/data/qipedc_reference
USER 1000:1000
EXPOSE 7860
CMD ["sh", "-c", "exec python -m uvicorn backend.production:app --host 0.0.0.0 --port \"$PORT\" --workers 1 --no-access-log"]
