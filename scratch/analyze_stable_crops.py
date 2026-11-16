import cv2
import json
import os
import numpy as np

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12), (0, 9),   # Middle
    (13, 14), (14, 15), (15, 16), (0, 13), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]

def analyze_and_draw(samples, img_prefix, frames_to_check):
    print(f"\n=================== {img_prefix} ===================")
    for s in samples:
        fidx = s['frame_index']
        if fidx not in frames_to_check:
            continue
        lms = s.get('landmarks')
        if not lms:
            continue
        img_path = f"scratch/m_n_frames/{img_prefix}_f{fidx:03d}.jpg"
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        pts = [(int(pt[0]*w), int(pt[1]*h)) for pt in lms]
        
        # Crop hand
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = max(0, min(xs)-30), min(w, max(xs)+30)
        min_y, max_y = max(0, min(ys)-30), min(h, max(ys)+30)
        
        ann = img.copy()
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(ann, pts[p1], pts[p2], (0, 255, 0), 2)
        for idx, (px, py) in enumerate(pts):
            color = (0, 0, 255) if idx in [4, 8, 12, 16, 20] else (255, 0, 0)
            cv2.circle(ann, (px, py), 4, color, -1)
            cv2.putText(ann, str(idx), (px+3, py-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imwrite(f"scratch/m_n_frames/crop_{img_prefix}_f{fidx:03d}_ann.jpg", ann[min_y:max_y, min_x:max_x])
        cv2.imwrite(f"scratch/m_n_frames/crop_{img_prefix}_f{fidx:03d}_orig.jpg", img[min_y:max_y, min_x:max_x])
        
        print(f"Frame {fidx}:")
        wrist = lms[0]
        for name, tip_idx, mcp_idx in [
            ("Thumb", 4, 2),
            ("Index", 8, 5),
            ("Middle", 12, 9),
            ("Ring", 16, 13),
            ("Pinky", 20, 17)
        ]:
            dx = lms[tip_idx][0] - lms[mcp_idx][0]
            dy = lms[tip_idx][1] - lms[mcp_idx][1]
            dz = lms[tip_idx][2] - lms[mcp_idx][2]
            dist_tip_mcp = (dx*dx + dy*dy + dz*dz)**0.5
            dist_tip_wrist = ((lms[tip_idx][0]-wrist[0])**2 + (lms[tip_idx][1]-wrist[1])**2 + (lms[tip_idx][2]-wrist[2])**2)**0.5
            print(f"  {name:7s}: tip_y - mcp_y = {dy:+.4f}, dist_tip_mcp = {dist_tip_mcp:.4f}, dist_tip_wrist = {dist_tip_wrist:.4f}")

analyze_and_draw(m_data['samples'], 'M_stable', [48, 50, 52, 54])
analyze_and_draw(n_data['samples'], 'N_stable', [22, 24, 26, 28])

