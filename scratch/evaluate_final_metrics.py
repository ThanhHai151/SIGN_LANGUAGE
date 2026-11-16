import sys
sys.path.insert(0, '.')
import json
import numpy as np
from collections import defaultdict
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# 1. Evaluate on vsl_alphabet_static_v2.npz (551 samples, 22 classes)
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
classes = list(data["classes"])
y = data["y"]
landmarks = data["landmarks"]

print("=" * 70)
print(f"EVALUATION ON {npz_path} (Total samples: {len(y)}, Classes: {len(classes)})")
print("=" * 70)

target_letters = ["T", "U", "S"]
stats = {let: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_true": 0} for let in target_letters}
conf_matrix = defaultdict(lambda: defaultdict(int))
pred_counts = defaultdict(int)

for idx in range(len(y)):
    true_label = classes[y[idx]]
    lm = landmarks[idx].tolist()
    pred_letter, conf = clf.classify(lm)
    
    pred_counts[pred_letter] += 1
    conf_matrix[true_label][pred_letter] += 1
    
    for let in target_letters:
        if true_label == let:
            stats[let]["total_true"] += 1
            if pred_letter == let:
                stats[let]["TP"] += 1
            else:
                stats[let]["FN"] += 1
        else:
            if pred_letter == let:
                stats[let]["FP"] += 1
            else:
                stats[let]["TN"] += 1

print("\n--- PERFORMANCE METRICS FOR TARGET LETTERS (T, U, S) ---")
print(f"{'Letter':<8}{'True':<8}{'TP':<8}{'FP':<8}{'FN':<8}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
for let in target_letters:
    tp = stats[let]["TP"]
    fp = stats[let]["FP"]
    fn = stats[let]["FN"]
    total = stats[let]["total_true"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"{let:<8}{total:<8}{tp:<8}{fp:<8}{fn:<8}{prec*100:>8.2f}%   {rec*100:>8.2f}%   {f1*100:>8.2f}%")

# Confusion Matrix for T, U, S and related classes: V, A, E, O, B, D, L
relevant_classes = ["T", "U", "S", "V", "A", "E", "O", "B", "D", "L", "H", "K"]
print("\n--- CONFUSION MATRIX (Rows: Ground Truth, Columns: Prediction) ---")
header = f"{'True/Pred':<10}" + "".join(f"{c:>6}" for c in relevant_classes) + f"{'Other/Unk':>12}"
print(header)
print("-" * len(header))
for true_c in relevant_classes:
    row_str = f"{true_c:<10}"
    other_unk = 0
    for pred_c in relevant_classes:
        row_str += f"{conf_matrix[true_c][pred_c]:>6}"
    for k, v in conf_matrix[true_c].items():
        if k not in relevant_classes:
            other_unk += v
    row_str += f"{other_unk:>12}"
    print(row_str)

# 2. Detailed hold-frame verification on static_landmarks raw data
print("\n" + "=" * 70)
print("EVALUATION ON RAW STATIC SAMPLES (Hold Frames in static_landmarks/*.json)")
print("=" * 70)
for let, vid, fn in [("T", "D0513B", "backend/data/alphabet_v2/static_landmarks/T.json"),
                     ("U", "D0514", "backend/data/alphabet_v2/static_landmarks/U.json"),
                     ("S", "D0512", "backend/data/alphabet_v2/static_landmarks/S.json")]:
    with open(fn, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    samples = [s for s in raw_data["samples"] if s.get("phase") == "stable_pose" and s.get("landmarks")]
    tp_hold = sum(1 for s in samples if clf.classify(s["landmarks"])[0] == let)
    print(f"Letter {let} ({vid}) Hold Frames: {tp_hold}/{len(samples)} ({tp_hold/len(samples)*100:.1f}%)")

