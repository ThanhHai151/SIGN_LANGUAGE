"""
Công cụ chạy Pilot trích xuất Landmark cho chữ 'A' từ video QIPEDC D0489.mp4.
=============================================================================
Tự động khởi động HTTP server, điều phối Chrome Headless chạy MediaPipe Hands,
thu nhận và kiểm định 21 mốc landmark thật, lưu kết quả ra fixture JSON.
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

# Cấu hình encoding chuẩn utf-8 cho console Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đưa workspace vào sys.path
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, WORKSPACE)

from backend.training.extract_video_landmarks import validate_landmark_format, save_landmark_fixture

PORT = 8769
OUTPUT_FIXTURE = os.path.join(WORKSPACE, "backend", "tests", "fixtures", "pilot_real_letter_a_landmarks.json")
HARNESS_URL = f"http://127.0.0.1:{PORT}/backend/training/extract_pilot_a_browser.html"


class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request Handler hỗ trợ HTTP Range requests cho việc tua (seek) video MP4."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE, **kwargs)

    def log_message(self, format, *args):
        # Tắt log ồn ào trên console
        pass

    def send_head(self):
        """Hỗ trợ cơ bản cho 206 Partial Content khi trình duyệt yêu cầu Range."""
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header or not range_header.startswith("bytes="):
            return super().send_head()

        try:
            total_size = os.path.getsize(path)
            range_val = range_header.strip()[6:]
            parts = range_val.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else total_size - 1

            if start >= total_size or end >= total_size or start > end:
                self.send_error(416, "Requested Range Not Satisfiable")
                return None

            length = end - start + 1
            f = open(path, "rb")
            f.seek(start)

            self.send_response(206)
            self.send_header("Content-type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{total_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f
        except Exception:
            return super().send_head()


def run_pilot():
    print("=" * 70)
    print("VSL CONNECT AI — PILOT TRÍCH XUẤT LANDMARK THẬT CHO CHỮ 'A'")
    print("=" * 70)

    # 1. Kiểm tra video nguồn
    video_path = os.path.join(WORKSPACE, "backend", "data", "media_cache", "videos", "D0489.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video chữ A tại: {video_path}")
    print(f"[1/5] Xác minh video nguồn: {video_path} ({os.path.getsize(video_path):,} bytes) — OK")

    # 2. Khởi động HTTP Server cục bộ
    print(f"[2/5] Khởi động HTTP server tại http://127.0.0.1:{PORT} ...")
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), RangeHTTPRequestHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    # 3. Tìm trình duyệt Chrome hoặc Edge
    chrome_candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    browser_bin = None
    for cand in chrome_candidates:
        if os.path.exists(cand):
            browser_bin = cand
            break

    if not browser_bin:
        httpd.shutdown()
        raise FileNotFoundError("Không tìm thấy Chrome hoặc Edge trên máy trạm.")

    print(f"[3/5] Sử dụng trình duyệt: {browser_bin}")
    print(f"      Điều hướng tới: {HARNESS_URL}")

    cmd = [
        browser_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=35000",
        "--dump-dom",
        HARNESS_URL
    ]

    dom_content = ""
    try:
        # Chạy Chrome Headless và thu thập DOM kết quả
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        dom_content = res.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        print("[LỖI] Chrome headless chạy quá thời gian (timeout 60s).")
    finally:
        httpd.shutdown()
        print("[4/5] Đã dừng HTTP Server.")

    # 4. Phân tích kết quả từ DOM
    if not dom_content:
        raise RuntimeError("Không thu được nội dung DOM từ trình duyệt.")

    status_start = dom_content.find('<div id="status">')
    status_end = dom_content.find('</div>', status_start)
    browser_status = dom_content[status_start + len('<div id="status">'): status_end].strip() if status_start != -1 else "UNKNOWN"
    print(f"-> Trạng thái ghi nhận từ Browser: {browser_status}")

    pre_start = dom_content.find('<pre id="output">')
    pre_end = dom_content.find('</pre>', pre_start)
    if pre_start == -1 or pre_end == -1:
        raise ValueError("Không tìm thấy thẻ <pre id='output'> trong DOM kết quả.")

    raw_json_str = dom_content[pre_start + len('<pre id="output">'): pre_end].strip()
    if not raw_json_str:
        raise ValueError(f"Dữ liệu JSON rỗng. Status DOM: {browser_status}")

    data = json.loads(raw_json_str)
    if "error" in data:
        raise RuntimeError(f"Browser Harness gặp lỗi: {data['error']}")

    samples = data.get("samples", [])
    successful = [s for s in samples if s.get("quality_status") == "ok"]
    failed = [s for s in samples if s.get("quality_status") != "ok"]

    print(f"[5/5] Kiểm định dữ liệu: {len(samples)} mẫu quét ({len(successful)} thành công, {len(failed)} thất bại).")

    # Kiểm tra từng mẫu thành công bằng validate_landmark_format() của backend
    valid_count = 0
    for s in successful:
        lms = s.get("landmarks")
        is_valid, msg = validate_landmark_format(lms)
        if not is_valid:
            print(f"  [CẢNH BÁO] Mẫu frame {s['frame_idx']} không đạt chuẩn: {msg}")
            s["quality_status"] = f"invalid_format: {msg}"
        else:
            valid_count += 1
            # In mẫu điểm đầu tiên (cổ tay) và ngón cái
            w = lms[0]
            t = lms[4]
            print(f"  ✓ Frame {s['frame_idx']:2d} (t={s['timestamp_sec']:.2f}s | {s['phase']:11s} | {s['split']:9s}): "
                  f"Hand={s['handedness']} Conf={s['confidence']:.4f} | Wrist=({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}) ThumbTip=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")

    for f_item in failed:
        print(f"  ✗ Frame {f_item['frame_idx']:2d} (t={f_item['timestamp_sec']:.2f}s | {f_item['phase']:11s}): {f_item['quality_status']}")

    # Cập nhật số lượng chuẩn
    data["successful_samples"] = valid_count
    data["failed_samples"] = len(samples) - valid_count

    # Lưu fixture vào OUTPUT_FIXTURE
    save_landmark_fixture(data, OUTPUT_FIXTURE, overwrite=True)
    print(f"\n-> ĐÃ XUẤT THÀNH CÔNG FIXTURE TẠI: {OUTPUT_FIXTURE}")
    print(f"   Tổng số mẫu: {len(samples)} | Mẫu hợp lệ: {valid_count} | Mẫu không thấy tay: {len(failed)}")
    return data


if __name__ == "__main__":
    run_pilot()
