import sys
import os
import json

sys.stdout.reconfigure(encoding='utf-8')

# Ensure venv packages are available
venv_path = r"D:\SIGN_LANGUAGE\.venv\Lib\site-packages"
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

import cv2
import mediapipe as mp

print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")

lexicon_path = r"D:\SIGN_LANGUAGE\backend\data\vsl_lexicon.json"
video_dir = r"D:\SIGN_LANGUAGE\backend\data\media_cache\videos"

with open(lexicon_path, 'r', encoding='utf-8') as f:
    lexicon = json.load(f)

alphabet_items = [item for item in lexicon if item.get('category') == 'Bảng chữ cái']
print(f"Total alphabet items in lexicon: {len(alphabet_items)}")

video_files = os.listdir(video_dir) if os.path.exists(video_dir) else []
print(f"Total video files in media_cache/videos: {len(video_files)}")

results = []
for item in alphabet_items:
    word = item.get('word', '')
    url = item.get('video_url', '')
    vid_name = os.path.basename(url) if url else ''
    local_path = os.path.join(video_dir, vid_name)
    exists = os.path.exists(local_path)
    size = os.path.getsize(local_path) if exists else 0
    
    fps = None
    frame_count = None
    width = None
    height = None
    duration = None
    if exists:
        cap = cv2.VideoCapture(local_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps and fps > 0 else 0
            cap.release()
            
    results.append({
        'id': item.get('id'),
        'word': word,
        'video_file': vid_name,
        'exists': exists,
        'size_bytes': size,
        'fps': fps,
        'frame_count': frame_count,
        'resolution': f"{width}x{height}" if width else None,
        'duration_sec': round(duration, 2) if duration else None,
        'sstcn_index': item.get('sstcn_index'),
        'how_to': item.get('how_to', '')
    })

print(f"\nLocal presence summary: {sum(1 for r in results if r['exists'])} / {len(results)} present.")
for r in results:
    status = f"OK ({r['frame_count']} frames, {r['duration_sec']}s, {r['resolution']})" if r['exists'] else "MISSING"
    print(f"[{r['id']}] word: '{r['word']}' -> {r['video_file']} | Status: {status}")

