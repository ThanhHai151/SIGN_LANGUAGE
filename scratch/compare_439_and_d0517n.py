import numpy as np
import json
import sys
sys.path.insert(0, ".")

import backend.vision.alphabet_classifier as ac

d = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz")
f439 = ac.alphabet_classifier.classify_detailed(d["landmarks"][439].tolist())["features"]

with open("scratch/d0517n_extracted_raw.json", encoding="utf-8") as f:
    d0517n_samples = json.load(f)

f42 = ac.alphabet_classifier.classify_detailed(d0517n_samples[42]["landmarks"])["features"]

print(f"{'Feature':<25} | {'Sample 439 (A)':<15} | {'D0517N frame 42 (X)':<15} | {'Diff':<10}")
print("-" * 75)
for k, v1 in sorted(f439.items()):
    v2 = f42.get(k)
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        diff = abs(v1 - v2)
        if diff > 0.05:
            print(f"{k:<25} | {v1:<15.4f} | {v2:<15.4f} | {diff:<10.4f}")

