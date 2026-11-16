import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

c = VSLAlphabetClassifier()

# Evaluate current classifier on all 22 classes
print("================ CURRENT NPZ EVALUATION ================")
results_before = {}
for target in ["P", "Q", "R", "X"]:
    t_idx = classes.index(target)
    tp = fn = fp = 0
    for i, (lm, label) in enumerate(zip(landmarks, y)):
        pred, _ = c.classify(lm.tolist())
        if label == t_idx:
            if pred == target: tp += 1
            else: fn += 1
        elif pred == target:
            fp += 1
    total = tp + fn
    results_before[target] = (tp, total, fp)
    print(f"{target}: TP={tp}/{total} ({tp/total*100:.1f}%), FP={fp}")

