import cv2
import numpy as np
import glob
import os

frames = sorted(glob.glob('scratch/frames_d0539/frame_*.jpg'))
print(f"Tracking hand in {len(frames)} frames:")

prev_cx = None
for f in frames:
    img = cv2.imread(f)
    h, w, _ = img.shape
    # Vùng nửa trên cơ thể
    roi = img[int(h*0.3):int(h*0.8), int(w*0.3):int(w*0.7)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Mask màu da
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Tìm contour lớn nhất trong góc phải dưới của ROI (bàn tay phải)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hand_c = max(contours, key=cv2.contourArea)
        M = cv2.moments(hand_c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            delta_x = cx - prev_cx if prev_cx is not None else 0
            prev_cx = cx
            print(f"  {os.path.basename(f)}: centroid=({cx}, {cy}), area={int(M['m00'])}, dx={delta_x}")

