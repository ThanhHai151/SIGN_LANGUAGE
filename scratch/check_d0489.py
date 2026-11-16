import os
import cv2
import json

vid_path = r"D:\SIGN_LANGUAGE\backend\data\media_cache\videos\D0489.mp4"
print(f"Path: {vid_path}")
print(f"Exists: {os.path.exists(vid_path)}")
print(f"Size: {os.path.getsize(vid_path):,} bytes")

cap = cv2.VideoCapture(vid_path)
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    print(f"Resolution: {w}x{h}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f}s")
    cap.release()
else:
    print("Failed to open video")

