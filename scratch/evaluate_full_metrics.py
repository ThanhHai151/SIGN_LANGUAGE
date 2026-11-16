import numpy as np
import json
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
classes = list(data['classes'])
y = data['y']
lms = data['landmarks']

all_tp = {}
all_fp = {}
all_unk = {}

for idx in range(len(y)):
    m = meta[idx]
    true_label = m['label']
    lm = lms[idx].tolist()
    
    res = clf.classify_detailed(lm)
    pred_label = res['letter']
    
    if pred_label == true_label:
        all_tp[true_label] = all_tp.get(true_label, 0) + 1
    elif pred_label in [None, "UNKNOWN"]:
        all_unk[true_label] = all_unk.get(true_label, 0) + 1
    else:
        all_fp[pred_label] = all_fp.get(pred_label, []) + [(true_label, m['frame_index'])]

print("=" * 80)
print(f"{'Class':6s} | {'Total':5s} | {'TP':4s} | {'FP':4s} | {'UNK':4s} | {'Precision':10s} | {'Recall':8s}")
print("-" * 80)

for c in sorted(list(set(classes))):
    c_idx = classes.index(c)
    total_c = int(np.sum(y == c_idx))
    c_tp = all_tp.get(c, 0)
    c_fp = len(all_fp.get(c, []))
    c_unk = all_unk.get(c, 0)
    
    prec = (c_tp / (c_tp + c_fp) * 100) if (c_tp + c_fp) > 0 else 0.0
    rec = (c_tp / total_c * 100) if total_c > 0 else 0.0
    print(f"{c:6s} | {total_c:5d} | {c_tp:4d} | {c_fp:4d} | {c_unk:4d} | {prec:9.1f}% | {rec:7.1f}%")

print("=" * 80)
print(f"Total False Positives across ALL classes: {sum(len(v) for v in all_fp.values())}")
print(f"Details of any FP: {all_fp}")
