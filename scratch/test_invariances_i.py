import sys
sys.path.insert(0, ".")
import json
import numpy as np

from backend.vision.alphabet_classifier import alphabet_classifier
from scratch.validate_new_rules import eval_rule_i_candidate, eval_rule_y_candidate

data = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
s24 = [s for s in data['samples'] if s['frame_index'] == 24][0]
lm24 = np.array(s24['landmarks'])

def rotate_2d(landmarks, angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s], [s, c]])
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

print("=== Rotations from -45° to +45° ===")
for angle in range(-45, 50, 15):
    rot_lm = rotate_2d(lm24, angle)
    feat = alphabet_classifier.extract_features(rot_lm)
    m_i = eval_rule_i_candidate(feat) if feat else False
    m_y = eval_rule_y_candidate(feat) if feat else False
    print(f"Angle {angle:+3d}°: MATCH I = {m_i}, MATCH Y = {m_y}")

print("\n=== Mirror (Left Hand) ===")
mir_lm = mirror_x(lm24)
feat_mir = alphabet_classifier.extract_features(mir_lm)
print(f"Mirror: MATCH I = {eval_rule_i_candidate(feat_mir)}, MATCH Y = {eval_rule_y_candidate(feat_mir)}")

print("\n=== Thumb folded over middle finger (forcing t_ext=True) ===")
# Shift thumb tip closer to middle finger
lm_t = np.array(lm24)
lm_t[4] = lm_t[12] * 0.7 + lm_t[4] * 0.3 # Move thumb tip closer to middle tip
feat_t = alphabet_classifier.extract_features(lm_t.tolist())
print(f"Thumb modified: t_ext = {feat_t.get('thumb_ext')}, MATCH I = {eval_rule_i_candidate(feat_t)}")

