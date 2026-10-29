import cv2
import numpy as np

for idx in [32, 40, 48]:
    img = cv2.imread(f'scratch/frames_d0539/hand_close_{idx}.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 25, 60], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Bỏ phần mặt (nếu có ở đỉnh ảnh)
    mask[:int(mask.shape[0]*0.2), :] = 0
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        print(f"Frame {idx}: bbox=({x},{y},{w},{h}), aspect_ratio(w/h)={aspect_ratio:.2f}, area={cv2.contourArea(c)}")

