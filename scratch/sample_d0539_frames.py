import os
import sys
import cv2

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

video_path = 'backend/data/media_cache/videos/D0539.mp4'
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Khảo sát video D0539.mp4 (Số 10): {total_frames} frames, {fps:.1f} fps")

# Trích xuất một số frame lưu vào thư mục scratch/frames_d0539/
out_dir = 'scratch/frames_d0539'
os.makedirs(out_dir, exist_ok=True)

# Trích xuất đều các frame từ đầu đến cuối
step = max(1, total_frames // 10)
saved = []
for f_idx in range(0, total_frames, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if ret:
        fn = os.path.join(out_dir, f"frame_{f_idx:03d}.jpg")
        cv2.imwrite(fn, frame)
        saved.append(f_idx)

cap.release()
print(f"Đã lưu các frame mẫu tại các chỉ số: {saved}")

