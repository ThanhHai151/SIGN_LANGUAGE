import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import math
import numpy as np
import backend.vision.alphabet_classifier as ac
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

classifier = VSLAlphabetClassifier()

npz_path = 'backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz'
data = np.load(npz_path, allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

v_idx = classes.index("V")
v_indices = np.where(y == v_idx)[0][:13]

for cand in [0.50, 0.48, 0.47, 0.46, 0.45, 0.44, 0.42, 0.40]:
    ac.H_HORIZ_RATIO_MIN = cand
    failed_rotations = []
    for idx in v_indices:
        lm_v = landmarks[idx].tolist()
        wv = lm_v[0]
        for deg in (-25, -15, -10, 10, 15, 25):
            rad = math.radians(deg)
            c, s_val = math.cos(rad), math.sin(rad)
            rot_v = [
                [
                    wv[0] + (pt[0] - wv[0]) * c - (pt[1] - wv[1]) * s_val,
                    wv[1] + (pt[0] - wv[0]) * s_val + (pt[1] - wv[1]) * c,
                    pt[2],
                ]
                for pt in lm_v
            ]
            res_rv = classifier.classify_detailed(rot_v)
            if res_rv["letter"] == "H":
                failed_rotations.append((idx, deg, res_rv["letter"]))
    print(f"Candidate {cand:.2f}: {len(failed_rotations)} rotated V false positives as H")
    if failed_rotations:
        print(f"   Examples: {failed_rotations[:3]}")

