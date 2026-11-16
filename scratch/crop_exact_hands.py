import cv2
import json
import numpy as np

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

targets = [
    ("O", "backend/data/media_cache/videos/D0506.mp4", "backend/data/alphabet_v2/static_landmarks/O.json", 36),
    ("S", "backend/data/media_cache/videos/D0512.mp4", "backend/data/alphabet_v2/static_landmarks/S.json", 36),
    ("P", "backend/data/media_cache/videos/D0509B.mp4", "backend/data/alphabet_v2/static_landmarks/P.json", 36),
    ("Q", "backend/data/media_cache/videos/D0510.mp4", "backend/data/alphabet_v2/static_landmarks/Q.json", 40),
]

for letter, vid_path, json_path, f_target in targets:
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # find sample
    sample = None
    for s in meta['samples']:
        f_idx = s.get('frame_index') or s.get('frame_idx')
        if f_idx == f_target and s.get('landmarks'):
            sample = s
            break
    if not sample:
        # pick first stable pose with landmarks
        for s in meta['samples']:
            if s.get('phase') == 'stable_pose' and s.get('landmarks'):
                sample = s
                f_target = s.get('frame_index') or s.get('frame_idx')
                break
                
    cap = cv2.VideoCapture(vid_path)
    cur_f = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cur_f == f_target:
            h, w, _ = frame.shape
            lm = sample['landmarks']
            xs = [int(p[0] * w) for p in lm]
            ys = [int(p[1] * h) for p in lm]
            min_x, max_x = max(0, min(xs) - 40), min(w, max(xs) + 40)
            min_y, max_y = max(0, min(ys) - 40), min(h, max(ys) + 40)
            
            # draw skeleton on a copy
            annotated = frame.copy()
            for p1, p2 in CONNECTIONS:
                cv2.line(annotated, (xs[p1], ys[p1]), (xs[p2], ys[p2]), (0, 255, 0), 2)
            for i, (x, y) in enumerate(zip(xs, ys)):
                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(annotated, str(i), (x+2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                
            crop_raw = frame[min_y:max_y, min_x:max_x]
            crop_ann = annotated[min_y:max_y, min_x:max_x]
            
            combined = np.hstack([crop_raw, crop_ann])
            out_p = f"scratch/hand_{letter}_f{f_target}.jpg"
            cv2.imwrite(out_p, combined)
            print(f"Saved {out_p} (bbox: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}])")
            break
        cur_f += 1
    cap.release()

