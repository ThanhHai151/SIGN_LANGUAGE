"""
Pipeline trích xuất Landmark Dày (Dense Landmark Extraction) cho 29 chữ cái VSL.
================================================================================
Thực thi theo Nhiệm vụ 3:
- Quét toàn bộ 29 video chữ cái tiếng Việt QIPEDC.
- Stride = 1 cho 7 chữ cái động (Ă, Â, Ê, Ô, Ơ, Ư, Đ) để bảo toàn trọn vẹn chuỗi nét vẽ.
- Stride = 2 cho 22 chữ cái tĩnh (A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y).
- Sử dụng Browser Harness và MediaPipe Hands WASM.
- Tổ chức đầu ra vào backend/data/alphabet_v2/:
    + static_landmarks/
    + motion_sequences/
    + metadata/
    + extraction_report.json
"""

import os
import sys
import json
import time
import subprocess
import http.server
import socketserver
import threading
import shutil
import cv2

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PORT = 8780
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

OUTPUT_DIR = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2")
STATIC_DIR = os.path.join(OUTPUT_DIR, "static_landmarks")
MOTION_DIR = os.path.join(OUTPUT_DIR, "motion_sequences")
META_DIR = os.path.join(OUTPUT_DIR, "metadata")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_frames")

for d in [OUTPUT_DIR, STATIC_DIR, MOTION_DIR, META_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# 29 chữ cái tiếng Việt chuẩn và video tương ứng
VIDEO_MAP = {
    'A': 'D0489.mp4', 'Ă': 'D0490B.mp4', 'Â': 'D0491B.mp4', 'B': 'D0492.mp4',
    'C': 'D0493.mp4', 'D': 'D0494.mp4', 'Đ': 'D0495.mp4', 'E': 'D0496.mp4',
    'Ê': 'D0497B.mp4', 'G': 'D0498B.mp4', 'H': 'D0500B.mp4', 'I': 'D0501.mp4',
    'K': 'D0502.mp4', 'L': 'D0503.mp4', 'M': 'D0504.mp4', 'N': 'D0505.mp4',
    'O': 'D0506.mp4', 'Ô': 'D0507B.mp4', 'Ơ': 'D0508N.mp4', 'P': 'D0509B.mp4',
    'Q': 'D0510.mp4', 'R': 'D0511.mp4', 'S': 'D0512.mp4', 'T': 'D0513B.mp4',
    'U': 'D0514.mp4', 'Ư': 'D0515N.mp4', 'V': 'D0516.mp4', 'X': 'D0517B.mp4',
    'Y': 'D0518N.mp4'
}

DYNAMIC_LETTERS = {'Ă', 'Â', 'Ê', 'Ô', 'Ơ', 'Ư', 'Đ'}

class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE, **kwargs)
    def log_message(self, format, *args):
        pass

def run_extraction():
    print("=" * 80)
    print("BẮT ĐẦU TRÍCH XUẤT LANDMARK THỰC TẾ DÀY CHO 29 CHỮ CÁI TIẾNG VIỆT (QIPEDC)")
    print("=" * 80)

    # Khởi động HTTP Server phục vụ Browser Harness
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), QuietHTTPHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(f"[1/5] Khởi động HTTP Server tại http://127.0.0.1:{PORT}")

    # Chuẩn bị danh sách tác vụ cho từng video
    video_tasks = []
    videos_metadata = {}

    for letter, vname in VIDEO_MAP.items():
        vpath = os.path.join(WORKSPACE, "backend", "data", "media_cache", "videos", vname)
        if not os.path.exists(vpath):
            print(f"LỖI: Không tìm thấy video: {vpath}")
            continue

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = round(total_frames / fps, 3)
        cap.release()

        is_dynamic = letter in DYNAMIC_LETTERS
        stride = 1 if is_dynamic else 2

        vid_id = os.path.splitext(vname)[0]
        videos_metadata[letter] = {
            "letter": letter,
            "video_id": vid_id,
            "filename": vname,
            "total_frames": total_frames,
            "fps": round(fps, 3),
            "duration_sec": dur,
            "resolution": f"{w}x{h}",
            "is_dynamic": is_dynamic,
            "stride": stride,
            "signer_id": "qipedc_signer_01"
        }

        frames_indices = list(range(0, total_frames, stride))
        video_tasks.append({
            "letter": letter,
            "video_id": vid_id,
            "vname": vname,
            "vpath": vpath,
            "fps": fps,
            "total_frames": total_frames,
            "is_dynamic": is_dynamic,
            "indices": frames_indices
        })

    print(f"[2/5] Đã thiết lập kế hoạch quét cho 29 video: {sum(len(t['indices']) for t in video_tasks)} frames.")

    # Xử lý theo từng batch (4 video mỗi batch để đảm bảo bộ nhớ và tính ổn định)
    batch_size = 4
    batches = [video_tasks[i:i + batch_size] for i in range(0, len(video_tasks), batch_size)]

    all_letter_samples = {l: [] for l in VIDEO_MAP.keys()}

    total_extracted_ok = 0
    total_frames_processed = 0

    harness_url = f"http://127.0.0.1:{PORT}/backend/training/extract_alphabet_v2_browser.html"

    start_time = time.time()

    try:
        for b_idx, batch in enumerate(batches, 1):
            print(f"\n--- BATCH {b_idx}/{len(batches)} (Gồm các chữ: {', '.join(t['letter'] for t in batch)}) ---")
            
            # Xóa sạch thư mục temp_frames trước mỗi batch
            for f in os.listdir(TEMP_DIR):
                try: os.remove(os.path.join(TEMP_DIR, f))
                except Exception: pass

            batch_tasks_payload = []
            
            # Trích xuất ảnh khung hình cho batch
            for vtask in batch:
                cap = cv2.VideoCapture(vtask["vpath"])
                fps = vtask["fps"]
                total_f = vtask["total_frames"]
                vid = vtask["video_id"]
                letter = vtask["letter"]
                is_dyn = vtask["is_dynamic"]

                for fidx in vtask["indices"]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    fname = f"{vid}_f{fidx:03d}.jpg"
                    fpath = os.path.join(TEMP_DIR, fname)
                    cv2.imwrite(fpath, frame)

                    ts = round(fidx / fps, 3)
                    
                    # Xác định nhãn pha
                    if is_dyn:
                        phase = "manual_review_needed"
                    else:
                        ratio = fidx / max(1, total_f - 1)
                        if ratio < 0.20:
                            phase = "entry"
                        elif ratio > 0.80:
                            phase = "exit"
                        else:
                            phase = "stable_pose"

                    batch_tasks_payload.append({
                        "letter_label": letter,
                        "video_id": vid,
                        "source_video_file": f"backend/data/media_cache/videos/{vtask['vname']}",
                        "frame_index": fidx,
                        "timestamp_sec": ts,
                        "signer_id": "qipedc_signer_01",
                        "phase": phase,
                        "frame_url": f"/backend/data/alphabet_v2/temp_frames/{fname}"
                    })

                cap.release()

            print(f"  -> Đã trích xuất {len(batch_tasks_payload)} ảnh JPEG. Gửi tác vụ sang Chrome Headless...")

            # Lưu tác vụ của batch
            batch_tasks_file = os.path.join(OUTPUT_DIR, "current_batch_tasks.json")
            with open(batch_tasks_file, "w", encoding="utf-8") as f:
                json.dump(batch_tasks_payload, f)

            # Gọi Chrome Headless
            cmd = [
                CHROME_PATH,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--virtual-time-budget=200000",
                "--dump-dom",
                harness_url
            ]

            t_batch_start = time.time()
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
            t_batch_dur = time.time() - t_batch_start

            dom_content = res.stdout.decode("utf-8", errors="replace")
            pre_start = dom_content.find('<pre id="output">')
            pre_end = dom_content.find('</pre>', pre_start)

            if pre_start == -1 or pre_end == -1:
                print("  [LỖI] Không tìm thấy output JSON trong DOM trả về từ Chrome!")
                continue

            json_text = dom_content[pre_start + len('<pre id="output">'): pre_end].strip()
            try:
                batch_results = json.loads(json_text)
            except Exception as e:
                print(f"  [LỖI] Parse JSON thất bại: {e}")
                continue

            if isinstance(batch_results, dict) and "error" in batch_results:
                print(f"  [LỖI] Trình duyệt báo lỗi: {batch_results['error']}")
                continue

            # Phân phối kết quả về từng chữ
            b_ok = sum(1 for s in batch_results if s.get("quality_status") == "ok")
            print(f"  -> Chrome hoàn tất trong {t_batch_dur:.1f}s. Nhận diện bàn tay thành công: {b_ok} / {len(batch_results)}")

            for sample in batch_results:
                l = sample["letter_label"]
                if l in all_letter_samples:
                    all_letter_samples[l].append(sample)

            total_frames_processed += len(batch_results)
            total_extracted_ok += b_ok

    finally:
        httpd.shutdown()
        # Dọn dẹp temp_frames
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        print("\n[3/5] Đã dừng HTTP Server và dọn dẹp ảnh tạm thời.")

    # 4. Phân loại và lưu trữ theo cấu trúc thư mục chuẩn
    print("[4/5] Tổ chức và phân loại cấu trúc dữ liệu theo 22 chữ tĩnh và 7 chữ động...")

    stats_per_letter = {}

    for letter, samples in all_letter_samples.items():
        # Đảm bảo thứ tự theo frame_index
        samples.sort(key=lambda s: s["frame_index"])
        total_f = len(samples)
        ok_f = sum(1 for s in samples if s.get("quality_status") == "ok")
        err_f = total_f - ok_f
        ratio_ok = round(ok_f / max(1, total_f) * 100, 2)

        is_dynamic = letter in DYNAMIC_LETTERS

        stats_per_letter[letter] = {
            "letter": letter,
            "type": "dynamic" if is_dynamic else "static",
            "video_id": VIDEO_MAP[letter].split(".")[0],
            "total_frames_scanned": total_f,
            "valid_landmarks": ok_f,
            "failed_frames": err_f,
            "valid_ratio_pct": ratio_ok
        }

        if is_dynamic:
            # Lưu trọn vẹn chuỗi động
            m_path = os.path.join(MOTION_DIR, f"{letter}.json")
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump({
                    "letter": letter,
                    "type": "motion_sequence",
                    "video_id": VIDEO_MAP[letter].split(".")[0],
                    "total_frames": total_f,
                    "valid_frames": ok_f,
                    "sequence": samples
                }, f, ensure_ascii=False, indent=2)
        else:
            # Lưu các mẫu chữ tĩnh
            s_path = os.path.join(STATIC_DIR, f"{letter}.json")
            with open(s_path, "w", encoding="utf-8") as f:
                json.dump({
                    "letter": letter,
                    "type": "static_landmarks",
                    "video_id": VIDEO_MAP[letter].split(".")[0],
                    "total_frames": total_f,
                    "valid_frames": ok_f,
                    "samples": samples
                }, f, ensure_ascii=False, indent=2)

    # Lưu metadata
    meta_path = os.path.join(META_DIR, "videos_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(videos_metadata, f, ensure_ascii=False, indent=2)

    # 5. Lưu báo cáo tổng hợp
    total_elapsed = time.time() - start_time
    report = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(total_elapsed, 2),
        "total_videos_processed": len(VIDEO_MAP),
        "total_frames_scanned": total_frames_processed,
        "total_valid_landmarks": total_extracted_ok,
        "overall_valid_ratio_pct": round(total_extracted_ok / max(1, total_frames_processed) * 100, 2),
        "dynamic_letters_count": len(DYNAMIC_LETTERS),
        "static_letters_count": len(VIDEO_MAP) - len(DYNAMIC_LETTERS),
        "stats_per_letter": stats_per_letter
    }

    report_path = os.path.join(OUTPUT_DIR, "extraction_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[5/5] ✅ HOÀN TẤT TOÀN DIỆN!")
    print(f"      - Đã xử lý: {len(VIDEO_MAP)} / 29 video")
    print(f"      - Tổng frame quét: {total_frames_processed}")
    print(f"      - Tổng landmark hợp lệ: {total_extracted_ok} ({report['overall_valid_ratio_pct']}%)")
    print(f"      - Báo cáo đã lưu: {report_path}")

if __name__ == "__main__":
    run_extraction()

