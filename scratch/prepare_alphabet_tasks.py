import os
import sys
import json
import cv2

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

lexicon_path = r"D:\SIGN_LANGUAGE\backend\data\vsl_lexicon.json"
video_dir = r"D:\SIGN_LANGUAGE\backend\data\media_cache\videos"
inspect_dir = r"D:\SIGN_LANGUAGE\backend\data\media_cache\inspect_frames"
os.makedirs(inspect_dir, exist_ok=True)

with open(lexicon_path, "r", encoding="utf-8") as f:
    data = json.load(f)

alphabet_entries = [e for e in data.get("entries", []) if e.get("category") == "Bảng chữ cái"]
vn_letters = ['a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô', 'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y']
multi_phase_letters = {'ă', 'â', 'ê', 'ô', 'ơ', 'ư', 'đ'}

all_tasks = []

print("Trích xuất các khung hình đại diện từ 29 video chữ cái...")
for l in vn_letters:
    entry = next((e for e in alphabet_entries if e.get("word") == l), None)
    if not entry:
        print(f"BỎ QUA: Không tìm thấy '{l}' trong lexicon")
        continue

    vid_url = entry.get("video_url", "")
    vid_id = os.path.splitext(os.path.basename(vid_url))[0]
    vid_file = os.path.basename(vid_url)
    vid_path = os.path.join(video_dir, vid_file)

    if not os.path.exists(vid_path):
        print(f"LỖI: Video không tồn tại: {vid_path}")
        continue

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"LỖI: Không thể mở video bằng OpenCV: {vid_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 10:
        print(f"LỖI: Video {vid_id} quá ngắn ({total_frames} frames)")
        continue

    # Chọn 7 tỷ lệ phân bố theo thời gian: 10%, 25%, 40%, 50%, 60%, 75%, 90%
    ratios = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
    is_multi = (l in multi_phase_letters)

    for r_idx, r in enumerate(ratios):
        fidx = int(round(r * (total_frames - 1)))
        fidx = max(0, min(fidx, total_frames - 1))
        ts = round(fidx / fps, 2)
        frame_filename = f"{vid_id}_f{fidx}.jpg"
        frame_path = os.path.join(inspect_dir, frame_filename)

        # Xác định pha và split
        if r <= 0.15:
            phase = "entry"
            split = "aux"
        elif r >= 0.85:
            phase = "exit"
            split = "aux"
        elif r in (0.25, 0.75):
            phase = "transition" if not is_multi else "unknown"
            split = "aux"
        elif r == 0.40:
            phase = "stable_pose" if not is_multi else "unknown"
            split = "dev"
        elif r == 0.50:
            phase = "stable_pose" if not is_multi else "unknown"
            split = "aux"
        elif r == 0.60:
            phase = "stable_pose" if not is_multi else "unknown"
            split = "eval"
        else:
            phase = "unknown"
            split = "aux"

        all_tasks.append({
            "letter": l,
            "letter_upper": l.upper(),
            "video_id": vid_id,
            "video_path": vid_path,
            "frame_idx": fidx,
            "timestamp_sec": ts,
            "total_frames": total_frames,
            "fps": round(fps, 3),
            "frame_file": frame_filename,
            "frame_path": frame_path,
            "phase": phase,
            "split": split,
            "is_multi_phase": is_multi
        })

print(f"Tổng số frame cần trích xuất cho 29 video: {len(all_tasks)} frame (trung bình 7 frame/video).")

# Trích xuất thực tế các frame thành file ảnh JPEG bằng OpenCV
cap_cache = {}
extracted_count = 0
for task in all_tasks:
    vpath = task["video_path"]
    if vpath not in cap_cache:
        cap_cache[vpath] = cv2.VideoCapture(vpath)
    cap = cap_cache[vpath]
    cap.set(cv2.CAP_PROP_POS_FRAMES, task["frame_idx"])
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(task["frame_path"], frame)
        extracted_count += 1
    else:
        print(f"Cảnh báo: Không đọc được frame {task['frame_idx']} của {task['video_id']}")

for cap in cap_cache.values():
    cap.release()

print(f"Đã trích xuất thành công {extracted_count} / {len(all_tasks)} tệp ảnh khung hình vào {inspect_dir}.")

# Lưu danh sách tác vụ để Browser Harness nạp
tasks_json_path = os.path.join(r"D:\SIGN_LANGUAGE\backend\training", "alphabet_extraction_tasks.json")
with open(tasks_json_path, "w", encoding="utf-8") as f:
    json.dump(all_tasks, f, ensure_ascii=False, indent=2)

print(f"Đã ghi danh sách tác vụ vào: {tasks_json_path}")

