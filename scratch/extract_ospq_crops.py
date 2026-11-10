import cv2
import json

targets = [
    ("O", "backend/data/media_cache/videos/D0506.mp4", 36),
    ("S", "backend/data/media_cache/videos/D0512.mp4", 36),
    ("P", "backend/data/media_cache/videos/D0509B.mp4", 36),
    ("Q", "backend/data/media_cache/videos/D0510.mp4", 40),
]

for letter, vid_path, f_target in targets:
    cap = cv2.VideoCapture(vid_path)
    cur_f = 0
    saved = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cur_f == f_target:
            h, w, _ = frame.shape
            # crop around the hand / torso region
            crop = frame[int(h*0.3):int(h*0.85), int(w*0.3):int(w*0.75)]
            out_path = f"scratch/crop_{letter}.jpg"
            cv2.imwrite(out_path, crop)
            print(f"Saved {out_path} from {vid_path} frame {cur_f}")
            saved = True
            break
        cur_f += 1
    cap.release()
    if not saved:
        print(f"Failed to find frame {f_target} in {vid_path}")

