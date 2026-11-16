import json
import os
import sys

sys.path.insert(0, ".")
from backend.vision.alphabet_classifier import alphabet_classifier, O_D_TI_MAX, O_IDX_STRAIGHT_MAX

d = json.load(open("backend/data/alphabet_v2/static_landmarks/O.json"))
for s in d["samples"]:
    if s.get("landmarks"):
        feat = alphabet_classifier.extract_features(s["landmarks"])
        d_ti = feat["d_TI"]
        idx_s = feat["idx_straight"]
        i_ext = feat["index_ext"]
        m_ext = feat["middle_ext"]
        r_ext = feat["ring_ext"]
        p_ext = feat["pinky_ext"]
        outer = (not i_ext) and (not m_ext) and (not r_ext) and (not p_ext)
        res = alphabet_classifier.classify_detailed(s["landmarks"])
        print(f"Frame {s['frame_index']:2d}: d_TI={d_ti:.4f} (<=0.22:{d_ti<=O_D_TI_MAX}), idx_str={idx_s:.4f} (<=0.48:{idx_s<=O_IDX_STRAIGHT_MAX}), outer={outer}, status={res['status']}, letter={res['letter']}")

