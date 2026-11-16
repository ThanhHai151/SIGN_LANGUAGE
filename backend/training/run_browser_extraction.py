"""
Script tự động điều phối trích xuất mốc landmark số 1-9 qua Browser MediaPipe.
=============================================================================
Quy trình:
1. Khởi chạy HTTP Server cục bộ tại cổng 8767 phục vụ workspace D:\\SIGN_LANGUAGE.
2. Khởi chạy Chrome Headless để nạp backend/training/extract_digits_browser.html.
3. MediaPipe Hands (WASM/CDN) xử lý lần lượt 18 khung hình (9 dev + 9 eval).
4. Thu nhận chuỗi JSON kết quả từ DOM, kiểm định tính hợp lệ bằng validate_landmark_format().
5. Lưu fixture vào backend/tests/fixtures/real_digit_landmarks.json.
6. Cập nhật trạng thái registry backend/tests/fixtures/digit_video_registry.json.
"""

import os
import sys
import json
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import subprocess
import http.server
import socketserver
import threading
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.training.extract_video_landmarks import validate_landmark_format, save_landmark_fixture

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PORT = 8767
FIXTURE_PATH = os.path.join(WORKSPACE, "backend", "tests", "fixtures", "real_digit_landmarks.json")
REGISTRY_PATH = os.path.join(WORKSPACE, "backend", "tests", "fixtures", "digit_video_registry.json")


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE, **kwargs)

    def log_message(self, format, *args):
        # Giảm log server cho console gọn gàng
        pass


def run_extraction():
    print("[1/5] Khởi động HTTP Server phục vụ tệp tại cổng", PORT)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), CustomHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        httpd.shutdown()
        raise FileNotFoundError(f"Không tìm thấy trình duyệt Chrome tại {chrome_path}")

    url = f"http://127.0.0.1:{PORT}/backend/training/extract_digits_browser.html"
    temp_dom_file = os.path.join(WORKSPACE, "backend", "data", "media_cache", "inspect_frames", "temp_dom_out.txt")

    print("[2/5] Khởi chạy Chrome Headless điều phối trích xuất MediaPipe...")
    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=30000",
        "--dump-dom",
        url
    ]

    try:
        with open(temp_dom_file, "w", encoding="utf-8") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=45)
    finally:
        httpd.shutdown()
        print("[3/5] Đã dừng HTTP Server.")

    if not os.path.exists(temp_dom_file):
        raise RuntimeError("Không thu được tệp kết quả DOM từ Chrome.")

    print("[4/5] Phân tích và kiểm định dữ liệu trích xuất...")
    with open(temp_dom_file, "r", encoding="utf-8") as f:
        dom_content = f.read()

    # Tìm nội dung trong <pre id="output">...</pre>
    pre_start = dom_content.find('<pre id="output">')
    pre_end = dom_content.find('</pre>', pre_start)
    if pre_start == -1 or pre_end == -1:
        raise ValueError("Không tìm thấy thẻ <pre id='output'> trong DOM kết quả.")

    json_str = dom_content[pre_start + len('<pre id="output">'): pre_end].strip()
    if not json_str:
        raise ValueError("Dữ liệu trong thẻ <pre id='output'> rỗng.")

    data = json.loads(json_str)
    samples = data.get("samples", [])
    print(f"-> Thu được tổng cộng {len(samples)} mẫu từ trình duyệt.")

    # Xác minh chất lượng từng mẫu
    valid_count = 0
    for idx, s in enumerate(samples):
        digit = s["digit"]
        split = s["split"]
        status = s["quality_status"]
        lms = s["landmarks"]
        if status == "ok" and lms:
            is_valid, msg = validate_landmark_format(lms)
            if not is_valid:
                s["quality_status"] = f"invalid_format: {msg}"
                s["landmarks"] = None
                print(f"  [X] Số {digit} ({split}): Lỗi định dạng mốc: {msg}")
            else:
                valid_count += 1
                print(f"  [V] Số {digit} ({split}): 21 mốc hợp lệ (Confidence: {s['confidence']})")
        else:
            print(f"  [X] Số {digit} ({split}): Không phát hiện bàn tay ({status})")

    data["successful_samples"] = valid_count

    # Bước 5: Lưu fixture
    print(f"[5/5] Lưu fixture kết quả vào {FIXTURE_PATH}")
    if os.path.exists(FIXTURE_PATH):
        print(f"Cảnh báo: File fixture {FIXTURE_PATH} đã tồn tại, tiến hành cập nhật.")
    save_landmark_fixture(data, FIXTURE_PATH, overwrite=True)

    # Cập nhật registry
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)

        for s in samples:
            d = s["digit"]
            split = s["split"]
            if d in registry["digits"] and s["quality_status"] == "ok":
                reg_entry = registry["digits"][d]
                reg_entry["has_ground_truth"] = True
                reg_entry["status"] = "GROUND_TRUTH_EXTRACTED"
                if split == "dev":
                    reg_entry["dev_frame"]["landmarks_verified"] = True
                elif split == "eval":
                    reg_entry["eval_frame"]["landmarks_verified"] = True

        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        print("Đã cập nhật trạng thái trong digit_video_registry.json.")

    # Xóa file DOM tạm
    try:
        os.remove(temp_dom_file)
    except OSError:
        pass

    print(f"Hoàn thành xuất sắc: {valid_count}/{len(samples)} mẫu đạt chuẩn Ground Truth.")
    return data


if __name__ == "__main__":
    run_extraction()
