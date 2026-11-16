import cv2

cap = cv2.VideoCapture('backend/data/media_cache/videos/D0498B.mp4')
cur = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if cur in [36, 116]:
        cv2.imwrite(f'scratch/d0498b_f{cur}.jpg', frame)
        print(f"Saved scratch/d0498b_f{cur}.jpg")
    cur += 1
cap.release()

