import cv2
import numpy as np
import glob
import os

frames = sorted(glob.glob('scratch/frames_d0539/*.jpg'))
print(f"Frames found: {len(frames)}")
for f in frames:
    img = cv2.imread(f)
    print(f"{os.path.basename(f)}: shape {img.shape}")

