import sys
sys.path.insert(0, '.')
import json
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
classes = list(data["classes"])
landmarks = data["landmarks"]
y = data["y"]

print("Comparing letters across static_v2.npz:")
for check_letter in ["O", "S", "P", "Q", "K", "G", "C", "A", "E"]:
    if check_letter in classes:
        idx = classes.index(check_letter)
        samples = [landmarks[i].tolist() for i in range(len(y)) if y[i] == idx]
        feats = [clf.extract_features(lm) for lm in samples if clf.extract_features(lm)]
        print(f"\n--- Class {check_letter} ({len(feats)} samples) ---")
        for key in ["d_IW", "d_MW", "d_RW", "d_PW", "d_TI", "d_TM", "d_TW", "d_IM", "idx_tip_y", "mid_tip_y", "ang_IM_dir", "thumb_lat"]:
            vals = [f[key] for f in feats if key in f]
            if vals:
                print(f"  {key:<12}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")

