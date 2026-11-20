#!/bin/bash
# ==========================================================
# Script Khởi chạy Hệ thống Giao tiếp VSL Connect AI
# Chạy đồng thời Backend Python FastAPI và Frontend Vite React
# ==========================================================

cd "$(dirname "$0")" || exit 1

PYTHON_BIN=${VSL_PYTHON:-python3}
if [[ -z ${VSL_PYTHON:-} && -z ${VIRTUAL_ENV:-} && -x .venv-sstcn/bin/python ]]; then
    PYTHON_BIN=.venv-sstcn/bin/python
fi

if [[ ${1:-} == --production ]]; then
    if [[ ! -f dist/index.html ]]; then
        echo 'Chưa có bản build. Chạy npm run build trước.' >&2
        exit 1
    fi

    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
    exec "$PYTHON_BIN" -m uvicorn backend.production:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
fi

echo "=========================================================="
echo "🤟 Đang khởi động VSL Connect AI (Hybrid PyTorch + Web)..."
echo "=========================================================="

# 1. Khởi chạy Backend FastAPI
"$PYTHON_BIN" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
echo "[Backend] Đang chạy tại: http://127.0.0.1:8000 (PID: $BACKEND_PID)"

# 2. Khởi chạy Frontend React (Vite)
node ./node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

# Dọn cả hai tiến trình khi Vite thoát hoặc script nhận tín hiệu dừng.
cleanup() {
    kill "$FRONTEND_PID" "$BACKEND_PID" 2>/dev/null
    wait "$FRONTEND_PID" "$BACKEND_PID" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[Frontend] Đang chạy tại: http://127.0.0.1:5173 (PID: $FRONTEND_PID)"

echo "----------------------------------------------------------"
echo "👉 Mở trình duyệt tại: http://127.0.0.1:5173"
echo "👉 API Docs (Swagger): http://127.0.0.1:8000/docs"
echo "Nhấn [Ctrl + C] để dừng toàn bộ hệ thống."
echo "----------------------------------------------------------"

wait "$FRONTEND_PID"
