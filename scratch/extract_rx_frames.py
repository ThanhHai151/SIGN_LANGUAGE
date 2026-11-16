import json
import cv2
import os

# 1. Inspect R samples
with open('backend/data/alphabet_v2/static_landmarks/R.json', 'r', encoding='utf-8') as f:
    r_data = json.load(f)

r_stable_indices = [
    s.get('frame_index') or s.get('frame_idx')
    for s in r_data['samples']
    if s.get('phase') == 'stable_pose' and s.get('landmarks')
]
print("R stable frame indices:", r_stable_indices)

# 2. Inspect X samples
with open('backend/data/alphabet_v2/static_landmarks/X.json', 'r', encoding='utf-8') as f:
    x_data = json.load(f)

x_stable_indices = [
    s.get('frame_index') or s.get('frame_idx')
    for s in x_data['samples']
    if s.get('phase') == 'stable_pose' and s.get('landmarks')
]
print("X stable frame indices:", x_stable_indices)

# 3. Extract sample frames for visual inspection
def extract_and_crop(video_path, frame_idx, landmarks, out_path):
    cap = cv2.VideoCapture(video_path)
    cur = 0
    target_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cur == frame_idx:
            target_frame = frame
            break
        cur += 1
    cap.release()
    
    if target_frame is None:
        print(f"Could not read frame {frame_idx} from {video_path}")
        return
        
    h, w, _ = target_frame.shape
    # Draw landmarks
    xs = [int(p[0] * w) for p in landmarks]
    ys = [int(p[1] * h) for p in landmarks]
    min_x, max_x = max(0, min(xs) - 40), min(w, max(xs) + 40)
    min_y, max_y = max(0, min(ys) - 40), min(h, max(ys) + 40)
    
    vis = target_frame.copy()
    for idx, (x, y) in enumerate(zip(xs, ys)):
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(vis, str(idx), (x+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
    # Draw connections
    connections = [
        (0,1),(1,2),(2,3),(3,4), # thumb
        (0,5),(5,6),(6,7),(7,8), # index
        (5,9),(9,10),(10,11),(11,12), # middle
        (9,13),(13,14),(14,15),(15,16), # ring
        (13,17),(17,18),(18,19),(19,20), # pinky
        (0,17) # palm
    ]
    for c1, c2 in connections:
        cv2.line(vis, (xs[c1], ys[c1]), (xs[c2], ys[c2]), (255, 200, 0), 1)
        
    crop = vis[min_y:max_y, min_x:max_x]
    if crop.size > 0:
        cv2.imwrite(out_path, crop)
        print(f"Saved crop: {out_path} (size: {crop.shape})")

# Extract R frame 34
for s in r_data['samples']:
    idx = s.get('frame_index') or s.get('frame_idx')
    if idx in [28, 34, 40] and s.get('landmarks'):
        extract_and_crop('backend/data/media_cache/videos/D0511.mp4', idx, s['landmarks'], f'scratch/hand_R_f{idx}.jpg')

# Extract X frame 36, 112
for s in x_data['samples']:
    idx = s.get('frame_index') or s.get('frame_idx')
    if idx in [28, 36, 44, 112] and s.get('landmarks'):
        extract_and_crop('backend/data/media_cache/videos/D0517B.mp4', idx, s['landmarks'], f'scratch/hand_X_f{idx}.jpg')
