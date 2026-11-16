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

with open(lexicon_path, "r", encoding="utf-8") as f:
    data = json.load(f)

alphabet_entries = [e for e in data.get("entries", []) if e.get("category") == "Bảng chữ cái"]
# Filter to 29 standard Vietnamese letters
# Letters in order: a, ă, â, b, c, d, đ, e, ê, g, h, i, k, l, m, n, o, ô, ơ, p, q, r, s, t, u, ư, v, x, y
vn_letters = ['a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô', 'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y']

letter_entries = []
for l in vn_letters:
    match = next((e for e in alphabet_entries if e.get("word") == l), None)
    if match:
        letter_entries.append(match)
    else:
        print(f"Missing in lexicon: {l}")

print(f"Found {len(letter_entries)} / 29 letters in lexicon.")

video_metadata = []
inspect_dir = r"D:\SIGN_LANGUAGE\backend\data\media_cache\inspect_frames"
os.makedirs(inspect_dir, exist_ok=True)

for entry in letter_entries:
    word = entry["word"]
    vid_url = entry.get("video_url", "")
    vid_id = os.path.splitext(os.path.basename(vid_url))[0]
    vid_file = os.path.basename(vid_url)
    vid_path = os.path.join(video_dir, vid_file)
    
    exists = os.path.exists(vid_path)
    fps = None
    total_frames = None
    duration = None
    width = None
    height = None
    
    if exists:
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps
            cap.release()
            
    video_metadata.append({
        "letter": word,
        "letter_upper": word.upper(),
        "video_id": vid_id,
        "video_file": vid_file,
        "video_path": vid_path,
        "exists": exists,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(duration, 2) if duration else None,
        "resolution": f"{width}x{height}" if width else None,
        "how_to": entry.get("how_to", "")
    })

print(f"{'Letter':<6} | {'Video ID':<8} | {'Frames':<7} | {'FPS':<6} | {'Dur(s)':<6} | {'Resolution':<10} | {'Status'}")
print("-" * 65)
for v in video_metadata:
    status = "OK" if v["exists"] and v["total_frames"] and v["total_frames"] > 0 else "FAIL"
    print(f"{v['letter']:<6} | {v['video_id']:<8} | {str(v['total_frames']):<7} | {str(round(v['fps'], 2)):<6} | {str(v['duration_sec']):<6} | {str(v['resolution']):<10} | {status}")
