import cv2
import numpy as np
import glob
import os

frames = sorted(glob.glob('scratch/frames_d0539/*.jpg'))
print(f"Total frames: {len(frames)}")

# Trong video QIPEDC, người ký ngồi ở giữa màn hình
# Hãy crop vùng ngực và tay (khoảng 30% đến 70% chiều cao, 30% đến 70% chiều rộng)
for f in frames:
    img = cv2.imread(f)
    h, w, _ = img.shape
    # Crop vùng tay phải của người ký (bên trái theo góc nhìn camera, hoặc trung tâm)
    crop = img[int(h*0.25):int(h*0.75), int(w*0.35):int(w*0.75)]
    out_crop = os.path.join('scratch/frames_d0539', 'crop_' + os.path.basename(f))
    cv2.imwrite(out_crop, crop)

print("Saved cropped frames to scratch/frames_d0539/crop_*.jpg")

