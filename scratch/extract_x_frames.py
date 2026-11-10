import cv2
import os

cap = cv2.VideoCapture("backend/data/media_cache/videos/D0517B.mp4")
os.makedirs("scratch/frames_x", exist_ok=True)

# Save frame 44 (lan 1) and frame 124 (lan 2)
for fidx in [40, 48, 116, 128]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"scratch/frames_x/frame_{fidx}.jpg", frame)
        print(f"Saved frame {fidx}")
cap.release()

