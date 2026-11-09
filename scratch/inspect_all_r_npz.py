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

for i, lm in enumerate(r_samples):
    feat = classifier.extract_features(lm.tolist())
    diag_r = classifier.evaluate_rule_r_diagnostics(feat)
    res = classifier.classify_detailed(lm.tolist())
    print(f"Sample {i:2d}: d_IW={feat.get('d_IW'):.2f}, d_MW={feat.get('d_MW'):.2f}, d_RW={feat.get('d_RW'):.2f}, d_PW={feat.get('d_PW'):.2f}, idx_dy={feat.get('idx_dy'):.2f}, idx_horiz={feat.get('idx_horiz_ratio'):.2f}, cross={feat.get('cross_indicator'):.4f} -> {res.get('letter')} (matched={diag_r.get('matched')}, failed={diag_r.get('failed_conditions')})")

