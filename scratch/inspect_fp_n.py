import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from scratch.test_rule_m import dist_3d

# Inspect N frames 18, 42, 46, 52
with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

for s in n_data['samples']:
    fidx = s['frame_index']
    if fidx in [18, 42, 46, 52]:
        lm = s['landmarks']
        wrist = lm[0]
        # check curl
        # in feature_extractor, finger_curl_ratio
        # and check why N has ring finger extended in these frames:
        print(f"N Frame {fidx}:")
        d_mw = dist_3d(lm[12], wrist)
        d_rw = dist_3d(lm[16], wrist)
        print(f"  d_MW={d_mw:.4f}, d_RW={d_rw:.4f}, d_RW/d_MW={d_rw/d_mw:.4f}")
        # What is the thumb position?
        d_tw = dist_3d(lm[4], wrist)
        d_tr = dist_3d(lm[4], lm[16])
        d_tm = dist_3d(lm[4], lm[12])
        print(f"  d_TW={d_tw:.4f}, d_TR={d_tr:.4f}, d_TM={d_tm:.4f}")

