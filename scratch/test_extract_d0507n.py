import os
import sys
import json
import time
import subprocess
import http.server
import socketserver
import threading
import cv2

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

def run():
    vpath = os.path.join(WORKSPACE, "backend", "data", "media_cache", "videos", "D0507N.mp4")
    if not os.path.exists(vpath):
        print(f"LỖI: Không tìm thấy {vpath}")
        return

    # Clear temp frames
    for f in os.listdir(TEMP_DIR):
        try: os.remove(os.path.join(TEMP_DIR, f))
        except Exception: pass

    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"D0507N: {total_frames} frames, {fps:.2f} fps")

    tasks = []
    for fidx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"D0507N_f{fidx:03d}.jpg"
        cv2.imwrite(os.path.join(TEMP_DIR, fname), frame)
        ts = round(fidx / fps, 3)
        tasks.append({
            "letter_label": "Ô",
            "video_id": "D0507N",
            "source_video_file": "backend/data/media_cache/videos/D0507N.mp4",
            "frame_index": fidx,
            "timestamp_sec": ts,
            "signer_id": "qipedc_signer_01",
            "phase": "manual_review_needed",
            "frame_url": f"/backend/data/alphabet_v2/temp_frames/{fname}"
        })
    cap.release()
    print(f"Extracted {len(tasks)} frames to {TEMP_DIR}")

    with open(os.path.join(OUTPUT_DIR, "current_batch_tasks.json"), "w", encoding="utf-8") as f:
        json.dump(tasks, f)

    # Start HTTP server
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), QuietHTTPHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(f"HTTP Server running at http://127.0.0.1:{PORT}")

    harness_url = f"http://127.0.0.1:{PORT}/backend/training/extract_alphabet_v2_browser.html"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=60000",
        "--dump-dom",
        harness_url
    ]

    print("Running Chrome Headless...")
    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        dur = time.time() - t0
        print(f"Chrome exited in {dur:.2f}s")
        dom_content = res.stdout.decode("utf-8", errors="replace")
        pre_start = dom_content.find('<pre id="output">')
        pre_end = dom_content.find('</pre>', pre_start)
        if pre_start != -1 and pre_end != -1:
            json_text = dom_content[pre_start + len('<pre id="output">'): pre_end].strip()
            data = json.loads(json_text)
            print(f"Result count: {len(data)}")
            ok_cnt = sum(1 for s in data if s.get("quality_status") == "ok")
            print(f"OK landmarks: {ok_cnt}/{len(data)}")
            if ok_cnt > 0:
                with open("scratch/d0507n_extracted_raw.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print("Saved scratch/d0507n_extracted_raw.json")
        else:
            print("Output pre tag not found in DOM!")
            print("DOM snippet:", dom_content[:500])
    except Exception as e:
        print(f"Chrome extraction error: {e}")
    finally:
        httpd.shutdown()

if __name__ == "__main__":
    run()

