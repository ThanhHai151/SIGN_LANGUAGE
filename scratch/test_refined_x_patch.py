import numpy as np
import sys
sys.path.insert(0, ".")

import backend.vision.alphabet_classifier as ac
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

# Configure:
# 1. P_DOWN_D_TM_MAX = 1.45 (for D0509N Southern P)
# 2. X_D_TW_MAX = 1.45 (for D0517N Southern X)
# 3. Remove X_D_TM_MIN (set to -999)
# 4. Add X_D_TI_MIN = 0.10 (blocks sample 439 with d_TI=0.084, keeps sample 386 with d_TI=0.126)

ac.P_DOWN_D_TM_MAX = 1.45
ac.X_D_TW_MAX = 1.45
ac.X_D_TM_MIN = -999.0
ac.X_D_TI_MIN = 0.10

orig_classify = ac.VSLAlphabetClassifier.evaluate_rules

def test_evaluate_rules(self, feat):
    matches = orig_classify(self, feat)
    # Check if upright X matched
    # if d_ti < 0.10, remove X
    d_ti = feat.get("d_TI")
    if d_ti is not None and d_ti < 0.10:
        matches = [m for m in matches if m[0] != "X"]
    return matches

ac.VSLAlphabetClassifier.evaluate_rules = test_evaluate_rules

npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks_all = data["landmarks"]
y_all = data["y"]
classes = [str(c) for c in data["classes"]]
total_samples = len(y_all)

correct = 0
fp_p = []
fp_x = []

for idx in range(total_samples):
    lm = landmarks_all[idx].tolist()
    gt = classes[y_all[idx]]
    session_id = f"eval_{idx}"
    rec.reset_diacritic_fsm(session_id)
    rec.reset()
    res = rec.predict_landmarks(lm, mode="letters", session_id=session_id, timestamp=1.0)
    pred = res.get("sign")
    if pred == gt:
        correct += 1
    if pred == "P" and gt != "P":
        fp_p.append((idx, gt))
    if pred == "X" and gt != "X":
        fp_x.append((idx, gt))

print(f"Test Accuracy: {correct}/{total_samples} ({correct/total_samples*100:.2f}%)")
print(f"FP for P: {len(fp_p)} ({fp_p})")
print(f"FP for X: {len(fp_x)} ({fp_x})")

# Check Sample 385, 386, 439
for test_idx in (385, 386, 439):
    lm = landmarks_all[test_idx].tolist()
    gt = classes[y_all[test_idx]]
    res = rec.predict_landmarks(lm, mode="letters", session_id="test", timestamp=1.0)
    print(f"Sample {test_idx} (GT: {gt}) -> Pred: {res.get('sign')}, Source: {res.get('source')}")
