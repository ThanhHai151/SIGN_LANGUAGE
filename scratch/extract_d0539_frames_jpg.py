import cv2
import os

video_path = 'backend/data/media_cache/videos/D0539.mp4'
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video {video_path}: {total_frames} frames, {fps:.2f} fps")

out_dir = 'backend/data/media_cache/frames_d0539'
os.makedirs(out_dir, exist_ok=True)

# Lấy các frame từ giai đoạn chuẩn bị, giai đoạn xoay cổ tay (active), đến kết thúc
# 10 frames tiêu biểu trải dài suốt cử chỉ
frame_indices = [10, 20, 28, 32, 36, 40, 44, 48, 54, 60, 70]
extracted = []

for fidx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    if fidx in frame_indices:
        fn = f"D0539_f{fidx:03d}.jpg"
        p = os.path.join(out_dir, fn)
        cv2.imwrite(p, frame)
        ts = fidx / fps
        extracted.append({"fidx": fidx, "ts": round(ts, 2), "file": fn})
        print(f"  Frame {fidx}: ts={ts:.2f}s -> {fn}")

cap.release()
print(f"Total extracted: {len(extracted)} frames to {out_dir}")

