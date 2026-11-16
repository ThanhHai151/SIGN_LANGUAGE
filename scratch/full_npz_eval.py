import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

c = VSLAlphabetClassifier()

print(f"{'Class':<8} {'Total':<8} {'TP':<8} {'Recall':<10} {'FP':<8}")
print("-" * 45)

total_tp = 0
total_fp = 0
total_samples = len(landmarks)

for t_idx, target in enumerate(classes):
    tp = fn = fp = 0
    for i, (lm, label) in enumerate(zip(landmarks, y)):
        pred, _ = c.classify(lm.tolist())
        if label == t_idx:
            if pred == target: tp += 1
            else: fn += 1
        elif pred == target:
            fp += 1
    total = tp + fn
    total_tp += tp
    total_fp += fp
    print(f"{target:<8} {total:<8} {tp:<8} {tp/total*100:<9.1f}% {fp:<8}")

print("-" * 45)
print(f"Overall Accuracy: {total_tp}/{total_samples} ({total_tp/total_samples*100:.1f}%), Total FP: {total_fp}")

