import os, numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

classifier = VSLAlphabetClassifier()

r_idx = classes.index("R")
r_samples = landmarks[np.where(y == r_idx)[0]]
print(f"Total R samples: {len(r_samples)}")

for i, lm in enumerate(r_samples):
    feat = classifier.extract_features(lm.tolist())
    print(f"Sample {i}: idx_tip_y={feat.get('idx_tip_y')}, mid_tip_y={feat.get('mid_tip_y')}, idx_horiz={feat.get('idx_horiz_ratio')}, idx_dy={feat.get('idx_dy')}, idx_dx={feat.get('idx_dx')}, cross_ind={feat.get('cross_indicator')}")
