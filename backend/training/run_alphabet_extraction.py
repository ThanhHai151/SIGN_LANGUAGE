"""
Script điều phối trích xuất Landmark thật cho toàn bộ 29 chữ cái VSL từ video QIPEDC.
=====================================================================================
Khởi chạy HTTP Server phục vụ workspace, điều phối Chrome Headless chạy MediaPipe Hands,
thẩm định dữ liệu chuẩn 21 điểm mốc qua validate_landmark_format(),
lưu kết quả chính thức vào backend/tests/fixtures/real_alphabet_landmarks.json.
"""

import os
import sys
import json
import time
import subprocess
import http.server
import socketserver
import threading
from typing import Dict, Any

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, WORKSPACE)

from backend.training.extract_video_landmarks import validate_landmark_format, save_landmark_fixture

PORT = 8770
OUTPUT_FIXTURE = os.path.join(WORKSPACE, "backend", "tests", "fixtures", "real_alphabet_landmarks.json")
HARNESS_URL = f"http://127.0.0.1:{PORT}/backend/training/extract_alphabet_browser.html"


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE, **kwargs)

    def log_message(self, format, *args):
        pass


def run_full_extraction():
    print("=" * 80)
    print("VSL CONNECT AI — TRÍCH XUẤT LANDMARK THẬT CHO 29 CHỮ CÁI TIẾNG VIỆT (QIPEDC)")
    print("=" * 80)

    tasks_path = os.path.join(WORKSPACE, "backend", "training", "alphabet_extraction_tasks.json")
    if not os.path.exists(tasks_path):
        raise FileNotFoundError(f"Không tìm thấy file tác vụ: {tasks_path}")

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"[1/5] Đã nạp {len(tasks)} tác vụ khung hình từ 29 video chữ cái.")

    # Khởi động HTTP server
    print(f"[2/5] Khởi động HTTP server tại http://127.0.0.1:{PORT} ...")
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), QuietHTTPHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    # Tìm trình duyệt
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        httpd.shutdown()
        raise FileNotFoundError(f"Không tìm thấy Chrome tại: {chrome_path}")

    print(f"[3/5] Điều phối Chrome Headless xử lý 29 video qua MediaPipe Hands...")
    print(f"      (Quá trình xử lý ~200 frame, vui lòng đợi...)")

    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=150000",
        "--dump-dom",
        HARNESS_URL
    ]

    dom_content = ""
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        dom_content = res.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        print("[LỖI] Chrome headless chạy quá thời gian (timeout 180s).")
    finally:
        httpd.shutdown()
        print("[4/5] Đã dừng HTTP Server.")

    if not dom_content:
        raise RuntimeError("Không thu được nội dung DOM từ trình duyệt.")

    status_start = dom_content.find('<div id="status">')
    status_end = dom_content.find('</div>', status_start)
    browser_status = dom_content[status_start + len('<div id="status">'): status_end].strip() if status_start != -1 else "UNKNOWN"
    print(f"-> Trạng thái ghi nhận từ Browser: {browser_status}")

    pre_start = dom_content.find('<pre id="output">')
    pre_end = dom_content.find('</pre>', pre_start)
    if pre_start == -1 or pre_end == -1:
        raise ValueError("Không tìm thấy thẻ <pre id='output'> trong DOM.")

    raw_json = dom_content[pre_start + len('<pre id="output">'): pre_end].strip()
    if not raw_json:
        raise ValueError(f"Dữ liệu JSON rỗng. Status: {browser_status}")

    data = json.loads(raw_json)
    if "error" in data:
        raise RuntimeError(f"Lỗi từ Browser Harness: {data['error']}")

    samples = data.get("samples", [])
    print(f"[5/5] Kiểm định dữ liệu trích xuất ({len(samples)} mẫu)...")

    # Kiểm định từng mẫu
    valid_count = 0
    letter_stats = {}

    for s in samples:
        l = s["letter"]
        if l not in letter_stats:
            letter_stats[l] = {
                "letter": l,
                "video_id": s["video_id"],
                "total": 0,
                "ok": 0,
                "no_hand": 0,
                "invalid": 0,
                "phases": set()
            }
        st = letter_stats[l]
        st["total"] += 1
        st["phases"].add(s.get("phase", "unknown"))

        status = s.get("quality_status")
        if status == "ok":
            lms = s.get("landmarks")
            is_valid, msg = validate_landmark_format(lms)
            if is_valid:
                st["ok"] += 1
                valid_count += 1
            else:
                st["invalid"] += 1
                s["quality_status"] = f"invalid_format: {msg}"
        elif status == "no_hand_detected":
            st["no_hand"] += 1
        else:
            st["invalid"] += 1

    # In bảng thống kê chi tiết theo từng chữ cái
    print("\n" + "=" * 80)
    print(f"{'Chữ':<5} | {'Video ID':<8} | {'Tổng mẫu':<9} | {'Thành công':<11} | {'Không thấy tay':<15} | {'Tỷ lệ':<8} | {'Trạng thái'}")
    print("-" * 80)
    for l, st in letter_stats.items():
        rate = f"{(st['ok'] / st['total']) * 100:.1f}%" if st['total'] > 0 else "0%"
        status_str = "TỐT" if st['ok'] >= 2 else ("THIẾU MẪU" if st['ok'] > 0 else "THẤT BẠI")
        print(f"{st['letter']:<5} | {st['video_id']:<8} | {st['total']:<9} | {st['ok']:<11} | {st['no_hand']:<15} | {rate:<8} | {status_str}")

    data["successful_samples"] = valid_count
    data["failed_samples"] = len(samples) - valid_count

    # Lưu fixture chính thức
    save_landmark_fixture(data, OUTPUT_FIXTURE, overwrite=True)
    print("\n" + "=" * 80)
    print(f"-> ĐÃ LƯU FIXTURE CHÍNH THỨC TẠI: {OUTPUT_FIXTURE}")
    print(f"   Tổng số video: 29 | Tổng mẫu: {len(samples)} | Mẫu OK: {valid_count} | Mẫu không thấy tay: {len(samples) - valid_count}")
    print("=" * 80)
    return data, letter_stats


if __name__ == "__main__":
    run_full_extraction()

