"""
Điều phối trích xuất Landmark thật cho Số 10 từ video QIPEDC D0539.mp4.
======================================================================
"""

import os
import sys
import json
import time
import subprocess
import http.server
import socketserver
import threading

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, WORKSPACE)

from backend.training.extract_video_landmarks import validate_landmark_format, save_landmark_fixture

PORT = 8775
OUTPUT_FIXTURE = os.path.join(WORKSPACE, "backend", "tests", "fixtures", "real_digit_10_landmarks.json")
HARNESS_URL = f"http://127.0.0.1:{PORT}/backend/training/extract_digit_10_browser.html"


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE, **kwargs)

    def log_message(self, format, *args):
        pass


def run_extraction():
    print("=" * 70)
    print("VSL CONNECT AI — TRÍCH XUẤT 21 LANDMARKS THẬT TỪ D0539 (SỐ 10)")
    print("=" * 70)

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), QuietHTTPHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(f"[1/4] HTTP Server đang lắng nghe tại http://127.0.0.1:{PORT}")

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        httpd.shutdown()
        raise FileNotFoundError(f"Không tìm thấy Chrome tại: {chrome_path}")

    print(f"[2/4] Khởi động Chrome Headless để chạy MediaPipe Hands...")
    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=60000",
        "--dump-dom",
        HARNESS_URL
    ]

    dom_content = ""
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        dom_content = res.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        print("[LỖI] Chrome headless chạy quá thời gian (timeout 90s).")
    finally:
        httpd.shutdown()
        print("[3/4] Đã dừng HTTP Server.")

    if not dom_content:
        raise RuntimeError("Không thu được DOM từ trình duyệt.")

    status_start = dom_content.find('<div id="status">')
    status_end = dom_content.find('</div>', status_start)
    browser_status = dom_content[status_start + len('<div id="status">'): status_end].strip() if status_start != -1 else "UNKNOWN"
    print(f"-> Trạng thái từ Browser: {browser_status}")

    pre_start = dom_content.find('<pre id="output">')
    pre_end = dom_content.find('</pre>', pre_start)
    if pre_start == -1 or pre_end == -1:
        raise RuntimeError("Không tìm thấy thẻ <pre id='output'> trong DOM.")

    raw_json_str = dom_content[pre_start + len('<pre id="output">'): pre_end].strip()
    data = json.loads(raw_json_str)

    if "error" in data:
        raise RuntimeError(f"Browser báo lỗi: {data['error']}")

    print(f"[4/4] Kiểm định dữ liệu và lưu fixture: {OUTPUT_FIXTURE}")
    samples = data.get("samples", [])
    ok_count = 0
    for s in samples:
        if s.get("quality_status") == "ok":
            lms = s.get("landmarks")
            valid, msg = validate_landmark_format(lms)
            if not valid:
                print(f"  [CẢNH BÁO] Frame {s['frame_idx']} không đạt format: {msg}")
                s["quality_status"] = "format_error"
            else:
                ok_count += 1

    print(f"-> Tổng số frame: {len(samples)}, thành công (ok): {ok_count}/{len(samples)}")

    os.makedirs(os.path.dirname(OUTPUT_FIXTURE), exist_ok=True)
    with open(OUTPUT_FIXTURE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"-> Đã ghi thành công fixture: {OUTPUT_FIXTURE}")
    return data


if __name__ == "__main__":
    run_extraction()

