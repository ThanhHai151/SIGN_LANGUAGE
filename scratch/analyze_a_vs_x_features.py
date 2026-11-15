import numpy as np
import json
import sys
sys.path.insert(0, ".")

import backend.vision.alphabet_classifier as ac

d = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz")
lm385 = d["landmarks"][385].tolist()
lm386 = d["landmarks"][386].tolist()
lm439 = d["landmarks"][439].tolist()

f385 = ac.alphabet_classifier.classify_detailed(lm385)["features"]
f386 = ac.alphabet_classifier.classify_detailed(lm386)["features"]
f439 = ac.alphabet_classifier.classify_detailed(lm439)["features"]

with open("scratch/d0517n_extracted_raw.json", encoding="utf-8") as f:
    d0517n_samples = json.load(f)

f_vid = [s for s in d0517n_samples if s["frame_index"] in (36, 42, 54)]

keys = [
    "d_TI", "d_TM", "d_TW", "d_IW", "d_MW", "d_RW", "d_PW",
    "index_curl", "idx_straight", "idx_tip_y", "idx_dy", "thumb_lat",
    "middle_curl", "ring_curl", "pinky_curl", "ang_IM_dir"
]

print(f"{'Key':<15} | {'Smpl 439 (A)':<12} | {'Smpl 385 (X)':<12} | {'Smpl 386 (X)':<12} | {'D0517N f42':<12}")
print("-" * 75)
f42 = ac.alphabet_classifier.classify_detailed(f_vid[1]["landmarks"])["features"]

for k in keys:
    v439 = f439.get(k)
    v385 = f385.get(k)
    v386 = f386.get(k)
    vvid = f42.get(k)
    s439 = f"{v439:.3f}" if isinstance(v439, (int, float)) else str(v439)
    s385 = f"{v385:.3f}" if isinstance(v385, (int, float)) else str(v385)
    s386 = f"{v386:.3f}" if isinstance(v386, (int, float)) else str(v386)
    svid = f"{vvid:.3f}" if isinstance(vvid, (int, float)) else str(vvid)
    print(f"{k:<15} | {s439:<12} | {s385:<12} | {s386:<12} | {svid:<12}")

