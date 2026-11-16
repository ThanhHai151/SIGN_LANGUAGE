import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from backend.vision.alphabet_classifier import AlphabetClassifier

clf = AlphabetClassifier()
npz = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
classes = npz["classes"]
y = npz["y"]
lms = npz["landmarks"]
meta = json.loads(str(npz["metadata_json"]))

s_mask = (y == 16)
s_lms = lms[s_mask]
s_meta = [m for m in meta if m.get("label") == "S"]

print(f"Total S samples: {len(s_lms)}")
tp = 0
for i in range(len(s_lms)):
    lm = s_lms[i].tolist()
    feat = clf.extract_features(lm)
    rules = clf.evaluate_rules(feat)
    fidx = s_meta[i].get("frame_index")
    diag = clf.evaluate_rule_s_diagnostics(feat)
    matched_letters = [r[0] for r in rules]
    is_s = "S" in matched_letters
    if is_s: tp += 1
    print(f"Frame {fidx:2d}: S_matched={is_s}, all_rules={matched_letters}, failed_conds={diag['failed_conditions']}")

print(f"\nS True Positives: {tp}/{len(s_lms)} ({tp/len(s_lms)*100:.1f}%)")

# Check all non-S samples in dataset for False Positives of S!
fp = 0
fp_details = []
non_s_indices = np.where(~s_mask)[0]
for idx in non_s_indices:
    lm = lms[idx].tolist()
    feat = clf.extract_features(lm)
    rules = clf.evaluate_rules(feat)
    matched = [r[0] for r in rules]
    if "S" in matched:
        fp += 1
        true_label = classes[y[idx]]
        m = meta[idx]
        fp_details.append((true_label, m.get("video_id"), m.get("frame_index")))

print(f"Total False Positives for S on entire non-S dataset ({len(non_s_indices)} samples): {fp}")
for d in fp_details:
    print(f"  FP on {d[0]} ({d[1]}, frame {d[2]})")

