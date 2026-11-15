import json
import sys
import numpy as np
sys.path.insert(0, ".")

import backend.vision.alphabet_classifier as ac

# Test P with P_DOWN_D_TM_MAX = 1.45
ac.P_DOWN_D_TM_MAX = 1.45
ac.X_D_TW_MAX = 1.45

# Patch upright rule X to require d_tm >= 0.50
orig_eval_x = ac.VSLAlphabetClassifier.evaluate_rule_x_diagnostics

def patched_eval_x(self, feat):
    res = orig_eval_x(self, feat)
    if res is None: return None
    d_tm = feat.get("d_TM")
    # require d_tm >= 0.50 for upright X
    d_tm_ok = (d_tm is not None and d_tm >= 0.50)
    if not d_tm_ok and res.get("matched"):
        res["matched"] = False
        res["failed_conditions"].append("d_tm_ge_min")
    return res

ac.VSLAlphabetClassifier.evaluate_rule_x_diagnostics = patched_eval_x

# Test on 551 dataset
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks_all = data["landmarks"]
y_all = data["y"]
classes = [str(c) for c in data["classes"]]

total = len(y_all)
fp_p = []
fp_x = []
correct_x = 0
total_x = sum(1 for c in y_all if classes[c] == "X")

for idx in range(total):
    lm = landmarks_all[idx].tolist()
    gt = classes[y_all[idx]]
    res = ac.alphabet_classifier.classify_detailed(lm)
    # Check if patched x matches
    feat = res.get("features", {})
    x_diag = ac.alphabet_classifier.evaluate_rule_x_diagnostics(feat)
    p_diag = ac.alphabet_classifier.evaluate_rule_p_diagnostics(feat)
    
    pred_x = x_diag.get("matched", False)
    pred_p = p_diag.get("matched", False)
    
    if pred_p and gt != "P":
        fp_p.append((idx, gt))
    if pred_x and gt != "X":
        fp_x.append((idx, gt))
    if pred_x and gt == "X":
        correct_x += 1

print(f"FP for P: {len(fp_p)} ({fp_p})")
print(f"FP for X: {len(fp_x)} ({fp_x})")
print(f"Correct X on static dataset: {correct_x}/{total_x}")

