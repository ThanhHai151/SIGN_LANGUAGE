import cv2
import numpy as np
import json
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Đọc motion json đã có trong dự án cho D0539
with open('backend/data/motions/D0539.json') as f:
    motion_data = json.load(f)

frames = motion_data.get('frames', [])
active = motion_data.get('active', [0, 0])
dur = motion_data.get('dur', 0)
fps = motion_data.get('fps', 8)

print(f"Motion D0539 analysis:")
print(f"  Duration: {dur}s, FPS: {fps}, Total motion frames: {len(frames)}")
print(f"  Active interval: {active[0]}s -> {active[1]}s")

# Mỗi frame trong motions/D0539.json là vector góc khớp robot
# Hãy kiểm tra sự thay đổi của các khớp tay trong khoảng active
active_start_idx = int(active[0] * fps)
active_end_idx = min(len(frames), int(active[1] * fps) + 1)
print(f"  Active frame indices: {active_start_idx} to {active_end_idx}")

# Tính độ biến thiên (variance / std) của các góc khớp trong khoảng active
active_frames = np.array(frames[active_start_idx:active_end_idx])
diffs = np.diff(active_frames, axis=0)
mean_abs_velocity = np.mean(np.abs(diffs), axis=0)
total_movement = np.sum(mean_abs_velocity)
print(f"  Total joint movement in active phase: {total_movement:.3f}")

# Kiểm tra video D0539.mp4 thực tế
cap = cv2.VideoCapture('backend/data/media_cache/videos/D0539.mp4')
v_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
v_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"\nVideo D0539.mp4 analysis:")
print(f"  Total frames: {v_total}, FPS: {v_fps:.2f}, Duration: {v_total/v_fps:.2f}s")

