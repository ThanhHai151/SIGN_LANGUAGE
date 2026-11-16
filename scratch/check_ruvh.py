import os, numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

classifier = VSLAlphabetClassifier()

for c_name in ["R", "U", "V", "H"]:
    c_idx = classes.index(c_name)
    samples = landmarks[np.where(y == c_idx)[0]]
    idx_dys = []
    idx_horizs = []
    cross_inds = []
    for lm in samples:
        feat = classifier.extract_features(lm.tolist())
        idx_dys.append(feat.get("idx_dy", 0.0))
        idx_horizs.append(feat.get("idx_horiz_ratio", 0.0))
        cross_inds.append(feat.get("cross_indicator", 0.0))
    print(f"Class {c_name} (n={len(samples)}):")
    print(f"  idx_dy min={min(idx_dys):.4f}, max={max(idx_dys):.4f}, mean={np.mean(idx_dys):.4f}")
    print(f"  idx_horiz min={min(idx_horizs):.4f}, max={max(idx_horizs):.4f}, mean={np.mean(idx_horizs):.4f}")
    print(f"  cross_ind min={min(cross_inds):.4f}, max={max(cross_inds):.4f}, mean={np.mean(cross_inds):.4f}")

