import os
import cv2

vid_path = r"D:\SIGN_LANGUAGE\backend\data\media_cache\videos\D0489.mp4"
cap = cv2.VideoCapture(vid_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}, Total frames: {total_frames}")

# Check frames at interval
inspect_dir = r"D:\SIGN_LANGUAGE\backend\data\media_cache\inspect_frames"
os.makedirs(inspect_dir, exist_ok=True)

# Sample 10 frames across the video
sample_indices = [10, 20, 30, 40, 50, 60, 70, 80]
for idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        ts = idx / fps
        out_path = os.path.join(inspect_dir, f"D0489_f{idx}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved frame {idx} (t={ts:.2f}s) to {out_path}")

cap.release()

