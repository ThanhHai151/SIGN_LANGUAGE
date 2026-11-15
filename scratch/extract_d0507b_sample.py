import cv2
import os

cap = cv2.VideoCapture("backend/data/media_cache/videos/D0507B.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"D0507B: {total_frames} frames, {fps:.2f} fps, {w}x{h}")

os.makedirs("scratch/frames_d0507b", exist_ok=True)

# Save every 5th frame
for fidx in range(0, total_frames, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"scratch/frames_d0507b/f{fidx:03d}.jpg", frame)

cap.release()
print("Saved sample frames to scratch/frames_d0507b/")

