import cv2
import json

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

for s in n_data['samples']:
    fidx = s['frame_index']
    if fidx in [20, 24, 28, 38, 46]:
        lms = s['landmarks']
        img = cv2.imread(f'scratch/m_n_frames/check_N_f{fidx:03d}.jpg')
        if img is None:
            continue
        h, w = img.shape[:2]
        pts = [(int(pt[0]*w), int(pt[1]*h)) for pt in lms]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ann = img.copy()
        for idx, (px, py) in enumerate(pts):
            cv2.circle(ann, (px, py), 3, (0, 255, 0), -1)
            cv2.putText(ann, str(idx), (px+2, py-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        min_x, max_x = max(0, min(xs)-20), min(w, max(xs)+20)
        min_y, max_y = max(0, min(ys)-20), min(h, max(ys)+20)
        cv2.imwrite(f'scratch/m_n_frames/check_N_f{fidx:03d}_crop.jpg', ann[min_y:max_y, min_x:max_x])
        print(f"N Frame {fidx}: Ring tip (16) at ({lms[16][0]:.3f}, {lms[16][1]:.3f}), Middle tip (12) at ({lms[12][0]:.3f}, {lms[12][1]:.3f})")

