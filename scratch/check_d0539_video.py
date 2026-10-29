import os
import sys
import cv2

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

video_path = 'backend/data/media_cache/videos/D0539.mp4'
if not os.path.exists(video_path):
    print(f"File {video_path} không tồn tại!")
    sys.exit(0)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frames / fps if fps > 0 else 0
print(f"Video D0539.mp4 (So 10): {w}x{h}, {fps} fps, {frames} frames, {duration:.2f}s")
cap.release()

