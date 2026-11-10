import sys
sys.path.insert(0, ".")
import json
import numpy as np
from backend.vision.alphabet_classifier import alphabet_classifier

# Let's inspect D0501 frame 24
data = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
s24 = [s for s in data['samples'] if s['frame_index'] == 24][0]
lm24 = np.array(s24['landmarks'])

def rotate_2d(landmarks, angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s], [s, c]])
    
    # rotate around wrist (index 0)
    wrist = landmarks[0][:2]
    new_lm = []
    for pt in landmarks:
        xy = pt[:2] - wrist
        new_xy = rot_mat @ xy + wrist
        new_lm.append([float(new_xy[0]), float(new_xy[1]), float(pt[2])])
    return new_lm

def mirror_x(landmarks):
    wrist_x = landmarks[0][0]
    return [[float(2.0 * wrist_x - pt[0]), float(pt[1]), float(pt[2])] for pt in landmarks]

print("--- Testing current alphabet_classifier on rotated Frame 24 ---")
for angle in [0, 10, 20, 30, -10, -20, -30]:
    rot_lm = rotate_2d(lm24, angle)
    res = alphabet_classifier.classify_detailed(rot_lm)
    print(f"Angle {angle:+3d}°: status={res['status']:8s} letter={res.get('letter')} conf={res.get('confidence')}")

print("\n--- Testing mirrored Frame 24 (Left hand / mirrored webcam) ---")
mir_lm = mirror_x(lm24)
res_mir = alphabet_classifier.classify_detailed(mir_lm)
print(f"Mirrored: status={res_mir['status']:8s} letter={res_mir.get('letter')} conf={res_mir.get('confidence')}")

