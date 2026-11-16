import cv2
import os

videos = {
    'D0507B': 'backend/data/media_cache/videos/D0507B.mp4',
    'D0507N': 'backend/data/media_cache/videos/D0507N.mp4',
    'D0507T': 'backend/data/media_cache/videos/D0507T.mp4',
    'D0508N': 'backend/data/media_cache/videos/D0508N.mp4',
}

for name, path in videos.items():
    if not os.path.exists(path):
        print(f"{name}: File not found ({path})")
        continue
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"{name}: {total} frames, {fps:.1f} fps, {w}x{h}, duration={total/fps:.2f}s")

