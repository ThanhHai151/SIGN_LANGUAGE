import cv2
import json

cap = cv2.VideoCapture("backend/data/media_cache/videos/D0510.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"D0510.mp4: total_frames={total_frames}, fps={fps}")

# Check landmark JSON for Q
with open("backend/data/alphabet_v2/static_landmarks/Q.json", 'r', encoding='utf-8') as f:
    q_data = json.load(f)

for s in q_data['samples']:
    f_idx = s.get('frame_index') or s.get('frame_idx')
    phase = s.get('phase')
    has_lm = bool(s.get('landmarks'))
    print(f"Frame {f_idx}: phase={phase}, has_landmarks={has_lm}")

# Let's save a contact sheet of frames 20, 28, 36, 44, 52, 60
saved_frames = [20, 28, 36, 44, 52, 60]
cur_f = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if cur_f in saved_frames:
        h, w, _ = frame.shape
        crop = frame[int(h*0.3):int(h*0.85), int(w*0.3):int(w*0.75)]
        cv2.imwrite(f"scratch/q_frame_{cur_f}.jpg", crop)
        print(f"Saved scratch/q_frame_{cur_f}.jpg")
    cur_f += 1
cap.release()

