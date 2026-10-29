import cv2
import numpy as np
import glob
import os

frames = sorted(glob.glob('scratch/frames_d0539/*.jpg'))
for i in range(len(frames) - 1):
    f1 = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
    f2 = cv2.imread(frames[i+1], cv2.IMREAD_GRAYSCALE)
    diff = cv2.absdiff(f1, f2)
    mean_diff = np.mean(diff)
    print(f"Diff between {os.path.basename(frames[i])} and {os.path.basename(frames[i+1])}: {mean_diff:.2f}")

