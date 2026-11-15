import json
import sys
import numpy as np
sys.path.insert(0, ".")

import backend.vision.alphabet_classifier as ac

# Test P with P_DOWN_D_TM_MAX = 1.45
orig_p_tm = ac.P_DOWN_D_TM_MAX
orig_x_tw = ac.X_D_TW_MAX

print(f"Original: P_DOWN_D_TM_MAX={orig_p_tm}, X_D_TW_MAX={orig_x_tw}")

ac.P_DOWN_D_TM_MAX = 1.45
ac.X_D_TW_MAX = 1.45

# 1. Test on 551 dataset
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks_all = data["landmarks"]
y_all = data["y"]
classes = [str(c) for c in data["classes"]]

correct = 0
total = len(y_all)
wrong_p = 0
wrong_x = 0
fp_p = []
fp_x = []

for idx in range(total):
    lm = landmarks_all[idx].tolist()
    gt = classes[y_all[idx]]
    res = ac.alphabet_classifier.classify_detailed(lm)
    pred = res.get("letter") if res.get("status") == "MATCH" else "UNKNOWN"
    
    if pred == "P" and gt != "P":
        fp_p.append((idx, gt))
    if pred == "X" and gt != "X":
        fp_x.append((idx, gt))

print(f"FP for P with new threshold: {len(fp_p)} ({fp_p})")
print(f"FP for X with new threshold: {len(fp_x)} ({fp_x})")

