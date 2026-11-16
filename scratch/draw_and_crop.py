import cv2
import json
import numpy as np

for letter, img_path, json_path, fidx in [
    ('T', 'backend/data/media_cache/inspect_frames/D0513B_f34.jpg', 'backend/data/alphabet_v2/static_landmarks/T.json', 34),
    ('U', 'backend/data/media_cache/inspect_frames/D0514_f38.jpg', 'backend/data/alphabet_v2/static_landmarks/U.json', 38),
    ('S', 'backend/data/media_cache/inspect_frames/D0512_f36.jpg', 'backend/data/alphabet_v2/static_landmarks/S.json', 36)
]:
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = {s['frame_index']: s for s in data['samples'] if s.get('landmarks')}

    sample = samples.get(fidx)
    if not sample:
        # get closest
        f_closest = min(samples.keys(), key=lambda k: abs(k - fidx))
        sample = samples[f_closest]

    lms = sample['landmarks']
    pts = [(int(p[0] * w), int(p[1] * h)) for p in lms]

    # Hand bounding box with padding
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = max(0, min(xs) - 40), min(w, max(xs) + 40)
    min_y, max_y = max(0, min(ys) - 40), min(h, max(ys) + 40)

    # Connections in MediaPipe hand
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    vis = img.copy()
    for c1, c2 in connections:
        cv2.line(vis, pts[c1], pts[c2], (0, 255, 0), 2)
    for i, pt in enumerate(pts):
        color = (0, 0, 255) if i in [4, 8, 12, 16, 20] else (255, 0, 0)
        cv2.circle(vis, pt, 4, color, -1)
        cv2.putText(vis, str(i), (pt[0]+3, pt[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    crop = vis[min_y:max_y, min_x:max_x]
    out_path = f'scratch/crop_{letter}.jpg'
    cv2.imwrite(out_path, crop)
    print(f"Saved {out_path}, size={crop.shape}")
