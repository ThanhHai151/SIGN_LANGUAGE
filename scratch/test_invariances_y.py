import sys
sys.path.insert(0, ".")
import json
import numpy as np

from backend.vision.alphabet_classifier import alphabet_classifier
from scratch.validate_new_rules import eval_rule_i_candidate, eval_rule_y_candidate
from scratch.test_invariances_i import rotate_2d, mirror_x

data = json.load(open('backend/data/alphabet_v2/static_landmarks/Y.json', encoding='utf-8'))
s24 = [s for s in data['samples'] if s['frame_index'] == 24][0]
lm24 = np.array(s24['landmarks'])

print("=== Y Rotations from -30° to +30° ===")
for angle in range(-30, 35, 15):
    rot_lm = rotate_2d(lm24, angle)
    feat = alphabet_classifier.extract_features(rot_lm)
    m_i = eval_rule_i_candidate(feat) if feat else False
    m_y = eval_rule_y_candidate(feat) if feat else False
    print(f"Angle {angle:+3d}°: MATCH I = {m_i}, MATCH Y = {m_y}")

print("\n=== Y Mirror (Left Hand) ===")
mir_lm = mirror_x(lm24)
feat_mir = alphabet_classifier.extract_features(mir_lm)
print(f"Mirror: MATCH I = {eval_rule_i_candidate(feat_mir)}, MATCH Y = {eval_rule_y_candidate(feat_mir)}")

