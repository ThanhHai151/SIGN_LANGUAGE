import os
import cv2
import numpy as np

def inspect_video(vpath, frames):
    cap = cv2.VideoCapture(vpath)
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"scratch/m_n_frames/{os.path.basename(vpath)}_f{f:03d}.jpg", frame)
    cap.release()

inspect_video('backend/data/media_cache/videos/D0504.mp4', [25, 35, 45, 50, 55, 65])
inspect_video('backend/data/media_cache/videos/D0505.mp4', [15, 25, 35, 45, 55, 65])
print("Done extracting inspection frames")
