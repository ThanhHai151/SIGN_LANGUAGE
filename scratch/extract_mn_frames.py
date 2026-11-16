"""
Trích xuất các frame stable_pose của video D0504 (M) và D0505 (N) để quan sát giải phẫu bàn tay.
"""
import cv2
import os

os.makedirs("scratch/frames_mn", exist_ok=True)

# 1. Trích xuất D0504 (M)
cap_m = cv2.VideoCapture("backend/data/media_cache/videos/D0504.mp4")
fps_m = cap_m.get(cv2.CAP_PROP_FPS)
total_m = int(cap_m.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"D0504 (M): fps={fps_m}, total_frames={total_m}")

for fidx in [20, 30, 40, 50, 60]:
    cap_m.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap_m.read()
    if ret:
        cv2.imwrite(f"scratch/frames_mn/M_f{fidx}.jpg", frame)
cap_m.release()

# 2. Trích xuất D0505 (N)
cap_n = cv2.VideoCapture("backend/data/media_cache/videos/D0505.mp4")
fps_n = cap_n.get(cv2.CAP_PROP_FPS)
total_n = int(cap_n.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"D0505 (N): fps={fps_n}, total_frames={total_n}")

for fidx in [15, 25, 35, 45, 55]:
    cap_n.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap_n.read()
    if ret:
        cv2.imwrite(f"scratch/frames_mn/N_f{fidx}.jpg", frame)
cap_n.release()

print("Trích xuất frames hoàn tất trong scratch/frames_mn/")

