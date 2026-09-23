#!/bin/bash
# ==========================================================
# Script Khởi chạy Hệ thống Giao tiếp VSL Connect AI
# Chạy đồng thời Backend Python FastAPI và Frontend Vite React
# ==========================================================

cd "$(dirname "$0")"

echo "=========================================================="
echo "🤟 Đang khởi động VSL Connect AI (Hybrid PyTorch + Web)..."
echo "=========================================================="

# 1. Khởi chạy Backend FastAPI
python3 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
echo "[Backend] Đang chạy tại: http://127.0.0.1:8000 (PID: $BACKEND_PID)"

# 2. Khởi chạy Frontend React (Vite)
node ./node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!
echo "[Frontend] Đang chạy tại: http://127.0.0.1:5173 (PID: $FRONTEND_PID)"

echo "----------------------------------------------------------"
echo "👉 Mở trình duyệt tại: http://127.0.0.1:5173"
echo "👉 API Docs (Swagger): http://127.0.0.1:8000/docs"
echo "Nhấn [Ctrl + C] để dừng toàn bộ hệ thống."
echo "----------------------------------------------------------"

# Bắt tín hiệu Ctrl + C để dừng cả hai tiến trình
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM

wait
