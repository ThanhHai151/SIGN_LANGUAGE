import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np

# Let's simulate the full AlphabetClassifier with these proposed changes:
from backend.vision.alphabet_classifier import AlphabetClassifier

clf = AlphabetClassifier()

# Override constants on classifier
import backend.vision.alphabet_classifier as ac
ac.S_D_TW_MAX = 1.40
ac.S_D_TI_MIN = 0.10
ac.S_D_TI_MAX = 0.48
ac.S_D_TM_MIN = 0.35
ac.S_D_TM_MAX = 0.78
ac.S_D_IW_MAX = 1.35
ac.S_D_MW_MAX = 0.85
ac.S_D_RW_MAX = 0.75
ac.S_D_PW_MIN = 0.35
ac.S_D_PW_MAX = 0.85
ac.S_THUMB_LAT_MIN = 0.55

# Test on all 551 samples in dataset
npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes']
y = npz['y']
lms = npz['landmarks']

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

# Also test on user's 60 webcam frames
webcam_file = 'backend/data/diagnostics/pilot_pilot_20261106_2347.jsonl'
with open(webcam_file, 'r', encoding='utf-8') as f:
    webcam_lines = [json.loads(l.strip()) for l in f if l.strip()]

webcam_hits = 0
for line in webcam_lines:
    if line.get('expected_sign') == 'S':
        feat = line['geometric']['features']
        # Evaluate rules directly with updated constants
        rules = clf.evaluate_rules(feat)
        if "S" in [r[0] for r in rules]:
            webcam_hits += 1

print(f"USER WEBCAM S HITS: {webcam_hits} / 60 ({webcam_hits/60*100:.1f}%)")

