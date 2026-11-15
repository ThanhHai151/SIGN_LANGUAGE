import numpy as np
import sys
sys.path.insert(0, ".")

import backend.vision.alphabet_classifier as ac
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks_all = data["landmarks"]
y_all = data["y"]
classes = [str(c) for c in data["classes"]]
total_samples = len(y_all)

# 1. Run with current settings
curr_correct = 0
curr_preds = []
for idx in range(total_samples):
    lm = landmarks_all[idx].tolist()
    gt = classes[y_all[idx]]
    session_id = f"eval_{idx}"
    rec.reset_diacritic_fsm(session_id)
    rec.reset()
    res = rec.predict_landmarks(lm, mode="letters", session_id=session_id, timestamp=1.0)
    pred = res.get("sign")
    curr_preds.append((pred, res))
    if pred == gt:
        curr_correct += 1

print(f"Current Settings Accuracy: {curr_correct}/{total_samples} ({curr_correct/total_samples*100:.2f}%)")

# 2. Run with old settings (P_DOWN_D_TM_MAX=1.10, X_D_TW_MAX=1.30, remove d_tm >= 0.50)
orig_p_tm = ac.P_DOWN_D_TM_MAX
orig_x_tw = ac.X_D_TW_MAX
orig_x_tm = getattr(ac, "X_D_TM_MIN", 0.50)

ac.P_DOWN_D_TM_MAX = 1.10
ac.X_D_TW_MAX = 1.30
ac.X_D_TM_MIN = -999.0  # disable

old_correct = 0
old_preds = []
for idx in range(total_samples):
    lm = landmarks_all[idx].tolist()
    gt = classes[y_all[idx]]
    session_id = f"eval_old_{idx}"
    rec.reset_diacritic_fsm(session_id)
    rec.reset()
    res = rec.predict_landmarks(lm, mode="letters", session_id=session_id, timestamp=1.0)
    pred = res.get("sign")
    old_preds.append((pred, res))
    if pred == gt:
        old_correct += 1

print(f"Reverted Settings Accuracy: {old_correct}/{total_samples} ({old_correct/total_samples*100:.2f}%)")

# Compare diffs
diffs = []
for idx in range(total_samples):
    gt = classes[y_all[idx]]
    c_pred, c_res = curr_preds[idx]
    o_pred, o_res = old_preds[idx]
    if c_pred != o_pred or (c_pred == gt) != (o_pred == gt):
        diffs.append({
            "idx": idx,
            "gt": gt,
            "curr_pred": c_pred,
            "old_pred": o_pred,
            "curr_status": c_res.get("source"),
            "old_status": o_res.get("source")
        })

print(f"\nDifferences between current and reverted settings ({len(diffs)}):")
for d in diffs:
    print(d)

