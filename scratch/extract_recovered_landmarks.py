import os
import sys
import json
import time
import subprocess
import http.server
import socketserver
import threading
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = 8789
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_frames")

os.makedirs(TEMP_DIR, exist_ok=True)

class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE, **kwargs)
    def log_message(self, format, *args):
        pass

def extract_video_landmarks(video_name, letter_label, output_json):
    vpath = os.path.join(WORKSPACE, "backend", "data", "media_cache", "videos", video_name)
    if not os.path.exists(vpath):
        print(f"LỖI: Không tìm thấy {vpath}", flush=True)
        return False

    # Clear temp frames
    for f in os.listdir(TEMP_DIR):
        try: os.remove(os.path.join(TEMP_DIR, f))
        except Exception: pass

    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n=== BẮT ĐẦU TRÍCH XUẤT {letter_label} ({video_name}): {total_frames} frames, {fps:.2f} fps ===", flush=True)

    tasks = []
    for fidx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"{video_name.replace('.mp4', '')}_f{fidx:03d}.jpg"
        cv2.imwrite(os.path.join(TEMP_DIR, fname), frame)
        ts = round(fidx / fps, 3)
        tasks.append({
            "letter_label": letter_label,
            "video_id": video_name.replace(".mp4", ""),
            "source_video_file": f"backend/data/media_cache/videos/{video_name}",
            "frame_index": fidx,
            "timestamp_sec": ts,
            "signer_id": "qipedc_signer_01",
            "phase": "manual_review_needed",
            "frame_url": f"/backend/data/alphabet_v2/temp_frames/{fname}"
        })
    cap.release()
    print(f"Đã xuất {len(tasks)} ảnh frame vào {TEMP_DIR}", flush=True)

    tasks_file = os.path.join(OUTPUT_DIR, "current_batch_tasks.json")
    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f)

    # Start HTTP server
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), QuietHTTPHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    harness_url = f"http://127.0.0.1:{PORT}/backend/training/extract_alphabet_v2_browser.html"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=180000",
        "--dump-dom",
        harness_url
    ]

    print("Khởi chạy Chrome Headless trích xuất MediaPipe (timeout 180s)...", flush=True)
    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        dur = time.time() - t0
        print(f"Chrome hoàn thành trong {dur:.2f}s", flush=True)
        dom_content = res.stdout.decode("utf-8", errors="replace")
        pre_start = dom_content.find('<pre id="output">')
        pre_end = dom_content.find('</pre>', pre_start)
        if pre_start != -1 and pre_end != -1:
            json_text = dom_content[pre_start + len('<pre id="output">'): pre_end].strip()
            data = json.loads(json_text)
            ok_cnt = sum(1 for s in data if s.get("quality_status") == "ok")
            print(f"Tổng mẫu: {len(data)}, Landmarks hợp lệ (OK): {ok_cnt}/{len(data)}", flush=True)
            if ok_cnt > 0:
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Đã lưu kết quả trích xuất vào: {output_json}", flush=True)
                return True
            else:
                print("CẢNH BÁO: Không có frame nào đạt quality_status == 'ok'", flush=True)
        else:
            print("LỖI: Không tìm thấy thẻ <pre id='output'> trong output của Chrome", flush=True)
    except Exception as e:
        print(f"Lỗi khi chạy Chrome: {e}", flush=True)
    finally:
        httpd.shutdown()
        server_thread.join(timeout=2)
    return False

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    jobs = [
        ("D0490N.mp4", "Ă", "scratch/d0490n_extracted_raw.json"),
        ("D0491N.mp4", "Â", "scratch/d0491n_extracted_raw.json"),
        ("D0497N.mp4", "Ê", "scratch/d0497n_extracted_raw.json"),
    ]
    for vname, letter, out_json in jobs:
        if target != "ALL" and target != letter:
            continue
        extract_video_landmarks(vname, letter, out_json)
