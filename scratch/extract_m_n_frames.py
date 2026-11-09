import cv2
import os

os.makedirs('scratch/m_n_frames', exist_ok=True)

def save_sample_frames(video_path, out_prefix, frame_indices):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {video_path}: FPS={fps}, Total Frames={total}")
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out_name = f"scratch/m_n_frames/{out_prefix}_f{idx:03d}.jpg"
            cv2.imwrite(out_name, frame)
            print(f"Saved {out_name}")
    cap.release()

save_sample_frames('backend/data/media_cache/videos/D0504.mp4', 'M_D0504', [10, 20, 28, 30, 35, 40])
save_sample_frames('backend/data/media_cache/videos/D0505.mp4', 'N_D0505', [10, 16, 20, 24, 28, 32])

