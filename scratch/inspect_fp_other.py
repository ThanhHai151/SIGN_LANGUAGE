import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from scratch.test_rule_m import evaluate_rule_m

npz_data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
all_labels = [npz_data['classes'][i] for i in npz_data['y']]
all_lms = npz_data['landmarks']

for idx, (lbl, lm) in enumerate(zip(all_labels, all_lms)):
    if lbl in ['A', 'H', 'S', 'V']:
        ok, reason = evaluate_rule_m(lm.tolist())
        if ok:
            print(f"Sample {idx} of {lbl} matched: {reason}")
            # check lm coordinates:
            p = lm.tolist()
            print(f"  Wrist: {p[0]}")
            print(f"  Index tip: {p[8]}, mcp: {p[5]}")
            print(f"  Middle tip: {p[12]}, mcp: {p[9]}")
            print(f"  Ring tip: {p[16]}, mcp: {p[13]}")

