import cv2
import numpy as np
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

cap = cv2.VideoCapture('backend/data/media_cache/videos/D0539.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total / fps if fps > 0 else 0

print(f"Video D0539.mp4 (Số 10): {w}x{h}, {fps:.2f} fps, {total} frames, {duration:.2f}s")

# Đọc mô tả trong backend/models/vsl_102_dictionary.py:
from backend.models.vsl_102_dictionary import SIGN_METADATA_102
meta_10 = SIGN_METADATA_102[38]
print(f"Mô tả chuẩn VSL-102: '{meta_10['description']}'")
print(f"Video URL VSL-102: {meta_10['video_url']}")
print(f"Category: {meta_10['category']}")

# Đọc motion D0539.json:
import json
with open('backend/data/motions/D0539.json') as f:
    m = json.load(f)
print(f"Motion D0539: active={m.get('active')}, fps={m.get('fps')}, dur={m.get('dur')}, total frames={len(m.get('frames', []))}")
print(f"Hand coverage: {m.get('hand_coverage')}")
