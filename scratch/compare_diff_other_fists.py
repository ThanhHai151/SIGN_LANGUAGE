"""
Kiểm tra diff(mid - ring) trên các chữ A, S, T, E.
"""
import numpy as np

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

for ch in ["A", "S", "T", "E"]:
    idx = classes.index(ch)
    diffs = []
    for lm, label in zip(landmarks, y):
        if label == idx:
            palm = np.linalg.norm(np.array(lm[9]) - np.array(lm[0])) + 1e-6
            diff = (lm[12][1] - lm[16][1]) / palm
            diffs.append(diff)
    print(f"Chữ {ch} ({len(diffs)} mẫu): diff(mid - ring) min={min(diffs):.3f}, max={max(diffs):.3f}, mean={np.mean(diffs):.3f}")

