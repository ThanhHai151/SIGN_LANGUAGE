import numpy as np
import json
import sys
import os

sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
classes = list(data['classes'])
y = data['y']
landmarks = data['landmarks']

total_samples = len(y)
class_stats = {}
confusion_matrix = {}

for c1 in classes:
    confusion_matrix[c1] = {c2: 0 for c2 in classes + ['UNKNOWN']}

for idx in range(total_samples):
    true_cls = classes[y[idx]]
    lm = landmarks[idx].tolist()
    pred_cls, conf = clf.classify(lm)
    confusion_matrix[true_cls][pred_cls] += 1

print("=" * 90)
print(f"{'Class':6s} | {'Total':5s} | {'TP':4s} | {'FP':4s} | {'FN/UNK':6s} | {'Precision':10s} | {'Recall':8s} | {'F1-Score':8s}")
print("-" * 90)

metrics_summary = {}

for c in classes:
    total_true = sum(confusion_matrix[c].values())
    tp = confusion_matrix[c][c]
    fp = sum(confusion_matrix[other][c] for other in classes if other != c)
    fn = total_true - tp
    unk = confusion_matrix[c]['UNKNOWN']
    
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    rec = (tp / total_true * 100.0) if total_true > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    
    metrics_summary[c] = {
        "total": total_true,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "unk": unk,
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2)
    }
    
    print(f"{c:6s} | {total_true:5d} | {tp:4d} | {fp:4d} | {unk:6d} | {prec:9.1f}% | {rec:7.1f}% | {f1:7.1f}%")

print("=" * 90)
total_fp_all = sum(m["fp"] for m in metrics_summary.values())
print(f"TOTAL FALSE POSITIVES ACROSS ENTIRE DATASET: {total_fp_all}")

# Specific focus on Q, R, S, X
print("\n" + "=" * 50)
print("CHI TIẾT 4 CHỮ Q, R, S, X:")
print("=" * 50)
for c in ["Q", "R", "S", "X"]:
    info = metrics_summary[c]
    print(f"Chữ {c}: TP={info['tp']}/{info['total']}, FP={info['fp']}, Precision={info['precision']}%, Recall={info['recall']}%, F1={info['f1']}%")

# Save detailed JSON report
with open('scratch/qrsx_final_metrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        "metrics_summary": metrics_summary,
        "confusion_matrix": confusion_matrix
    }, f, indent=2, ensure_ascii=False)

print("Saved scratch/qrsx_final_metrics.json")


