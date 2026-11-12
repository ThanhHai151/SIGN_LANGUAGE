import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from backend.vision.recognizer import VSLRecognizer
import backend.vision.alphabet_classifier as ac

rec = VSLRecognizer()
rec.mode = "alphabet"

# Load legacy 551 dataset
npz_path = 'backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz'
legacy_data = np.load(npz_path)
X_legacy = legacy_data['landmarks']  # (551, 21, 3)
y_legacy = legacy_data['y']          # (551,)
classes = legacy_data['classes']     # 22 classes

# Load canonical fixtures
fixtures = {}
for letter in ['H', 'D0500N', 'V', 'R', 'U', 'K', 'D', 'N', 'M']:
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            fixtures[letter] = json.load(f).get('samples', [])

candidates = [0.50, 0.45, 0.40, 0.35]

for cand in candidates:
    print(f"\n=======================================================")
    print(f"EVALUATING CANDIDATE: H_HORIZ_RATIO_MIN = {cand}")
    print(f"=======================================================")
    
    ac.H_HORIZ_RATIO_MIN = cand
    
    # 1. Canonical Fixtures
    print("--- Canonical Fixtures ---")
    for letter, samples in fixtures.items():
        correct = 0
        h_misclassified = 0
        other_preds = {}
        for s in samples:
            lms = s['landmarks']
            res = rec.predict_landmarks(lms, mode="letters")
            pred = res.get('sign', 'NEUTRAL')
            expected = 'H' if letter in ('D0500N', 'H') else letter
            if pred == expected:
                correct += 1
            else:
                other_preds[pred] = other_preds.get(pred, 0) + 1
            if expected != 'H' and pred == 'H':
                h_misclassified += 1
        
        acc = correct / len(samples) * 100 if samples else 0
        h_leak = f" [WARNING: {h_misclassified} misclassified as H!]" if h_misclassified > 0 else ""
        print(f"  {letter:7s} ({len(samples):2d} samples): {correct:2d}/{len(samples):2d} ({acc:5.1f}%) | others: {other_preds}{h_leak}")

    # 2. Legacy 551 Dataset
    total_correct = 0
    per_class = {}
    h_fp_letters = {}
    for i in range(len(X_legacy)):
        lms = X_legacy[i].tolist()
        expected = str(classes[y_legacy[i]])
        res = rec.predict_landmarks(lms, mode="letters")
        pred = res.get('sign', 'NEUTRAL')
        if expected not in per_class:
            per_class[expected] = {'correct': 0, 'total': 0, 'pred_as_h': 0}
        per_class[expected]['total'] += 1
        if pred == expected:
            total_correct += 1
            per_class[expected]['correct'] += 1
        elif pred == 'H':
            per_class[expected]['pred_as_h'] += 1
            h_fp_letters[expected] = h_fp_letters.get(expected, 0) + 1

    total_acc = total_correct / len(X_legacy) * 100
    print(f"\n--- Legacy 551 Dataset ---")
    print(f"  Total Legacy Accuracy: {total_correct}/{len(X_legacy)} ({total_acc:.2f}%)")
    for target in ['H', 'V', 'R', 'U', 'K', 'D', 'N', 'M']:
        if target in per_class:
            c = per_class[target]['correct']
            t = per_class[target]['total']
            fp = per_class[target]['pred_as_h']
            print(f"    Class {target:2s}: {c:2d}/{t:2d} ({c/t*100:5.1f}%) | false-as-H: {fp}")
    
    if h_fp_letters:
        print(f"  Total False Positives as H across ALL 22 classes: {h_fp_letters}")
    else:
        print(f"  Total False Positives as H across ALL 22 classes: ZERO (0)")

# Reset monkey-patch
ac.H_HORIZ_RATIO_MIN = 0.50
