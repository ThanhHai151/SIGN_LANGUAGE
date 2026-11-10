import json
import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

c = VSLAlphabetClassifier()

# 1. Inspect fixtures
with open("backend/tests/fixtures/real_alphabet_landmarks.json", "r", encoding="utf-8") as f:
    fixtures = json.load(f)

print("=== FIXTURES EVALUATION ===")
for letter in ["P", "Q", "R", "X"]:
    if letter in fixtures:
        lms = fixtures[letter]
        pred, conf = c.classify(lms)
        feat = c.extract_features(lms)
        print(f"Fixture {letter}: pred={pred} (conf={conf:.2f})")
        # Print diagnostic if pred != letter
        if pred != letter:
            diag_fn = getattr(c, f"evaluate_rule_{letter.lower()}_diagnostics")
            diag = diag_fn(feat)
            print(f"  FAILED conditions: {diag.get('failed_conditions') if diag else 'None'}")
    else:
        print(f"Fixture {letter} NOT in real_alphabet_landmarks.json")

# 2. Inspect NPZ dataset
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

print("\n=== NPZ DATASET EVALUATION ===")
for target in ["P", "Q", "R", "X"]:
    if target not in classes:
        print(f"Class {target} NOT in NPZ classes: {classes}")
        continue
    t_idx = classes.index(target)
    tp = fn = fp = 0
    failed_cond_counts = {}
    diag_fn = getattr(c, f"evaluate_rule_{target.lower()}_diagnostics")
    
    for i, (lm, label) in enumerate(zip(landmarks, y)):
        pred, _ = c.classify(lm.tolist())
        is_target = (label == t_idx)
        if is_target:
            if pred == target:
                tp += 1
            else:
                fn += 1
                feat = c.extract_features(lm.tolist())
                diag = diag_fn(feat)
                if diag:
                    for cond in diag.get("failed_conditions", []):
                        failed_cond_counts[cond] = failed_cond_counts.get(cond, 0) + 1
        else:
            if pred == target:
                fp += 1
                print(f"  FALSE POSITIVE: sample {i} (actual {classes[label]}) predicted as {target}!")
    
    total = tp + fn
    rec = tp / total if total > 0 else 0
    print(f"Target {target}: TP={tp}/{total} ({rec*100:.1f}%), FP={fp}")
    if failed_cond_counts:
        print(f"  Failed conditions breakdown for FN ({fn} samples):")
        for k, v in sorted(failed_cond_counts.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}/{fn} ({v/fn*100:.1f}%)")

