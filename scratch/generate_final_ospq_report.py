import os
import sys
import json
import numpy as np

sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = [str(c) for c in data['classes']]
y = data['y']
landmarks = data['landmarks']

clf = VSLAlphabetClassifier()

n_samples = len(y)
target_letters = ['O', 'S', 'P', 'Q']

# Compute confusion and predictions
conf_matrix = {let: {other: 0 for other in classes + ['UNKNOWN']} for let in target_letters}
overall_counts = {let: {"total": 0, "TP": 0, "FP": 0, "FN": 0, "TN": 0} for let in target_letters}

predictions = []
for i in range(n_samples):
    true_label = classes[y[i]]
    lm = landmarks[i].tolist()
    pred_letter, conf = clf.classify(lm)
    predictions.append((true_label, pred_letter, conf))
    
    if true_label in target_letters:
        conf_matrix[true_label][pred_letter] += 1
        overall_counts[true_label]["total"] += 1

for true_label, pred_letter, conf in predictions:
    for target in target_letters:
        if pred_letter == target:
            if true_label == target:
                overall_counts[target]["TP"] += 1
            else:
                overall_counts[target]["FP"] += 1
        else:
            if true_label == target:
                overall_counts[target]["FN"] += 1
            else:
                overall_counts[target]["TN"] += 1

report = {
    "total_samples": n_samples,
    "classes": classes,
    "metrics": {},
    "confusion": conf_matrix,
}

for let in target_letters:
    tp = overall_counts[let]["TP"]
    fp = overall_counts[let]["FP"]
    fn = overall_counts[let]["FN"]
    tn = overall_counts[let]["TN"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    report["metrics"][let] = {
        "total_true": overall_counts[let]["total"],
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
    }

with open('scratch/final_ospq_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report["metrics"], indent=2))

