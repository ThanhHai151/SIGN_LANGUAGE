import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes']
y = npz['y']
lms = npz['landmarks']

from backend.vision.alphabet_classifier import AlphabetClassifier
clf = AlphabetClassifier()

# Override R constants on classifier
import backend.vision.alphabet_classifier as ac
ac.R_D_IW_MIN = 1.10
ac.R_D_MW_MIN = 1.15
ac.R_D_RW_MAX = 0.75
ac.R_D_PW_MAX = 0.75
ac.R_D_IM_MAX = 0.60
ac.R_IDX_STRAIGHT_MIN = 0.75
ac.R_MID_STRAIGHT_MIN = 0.75
ac.R_IDX_TIP_Y_MAX = -0.70
ac.R_MID_TIP_Y_MAX = -0.70
ac.R_CROSS_IND_MAX = 0.0060
ac.R_D_MW_IW_DIFF_MIN = -0.35
ac.R_MCP_WIDTH_MIN = 0.40
ac.R_THUMB_RING_MIN = 0.18

results = {c: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for c in classes}

for i in range(len(y)):
    true_label = classes[y[i]]
    results[true_label]["total"] += 1
    lm = lms[i].tolist()
    pred_sign, conf = clf.classify(lm)
    if pred_sign == true_label:
        results[true_label]["tp"] += 1
    elif pred_sign != "UNKNOWN":
        results[pred_sign]["fp"] += 1
        results[true_label]["fn"] += 1
        print(f"MISCLASSIFICATION: True={true_label}, Pred={pred_sign}")
    else:
        results[true_label]["fn"] += 1

print("\n--- RESULTS ACROSS ALL 22 CLASSES ---")
total_fp = 0
for c in classes:
    r = results[c]
    total_fp += r['fp']
    print(f"Class {c:2s}: Total={r['total']:2d}, TP={r['tp']:2d}, FP={r['fp']:2d}, FN={r['fn']:2d}")

print(f"\nTOTAL FALSE POSITIVES ACROSS ALL 22 CLASSES: {total_fp}")

