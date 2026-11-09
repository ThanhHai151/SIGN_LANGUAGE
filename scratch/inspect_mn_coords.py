"""
Kiểm tra chi tiết tọa độ 21 điểm của M sample 5 và N sample 5.
"""
import numpy as np

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

m_idx = classes.index("M")
n_idx = classes.index("N")

m_lms = [lm for lm, label in zip(landmarks, y) if label == m_idx]
n_lms = [lm for lm, label in zip(landmarks, y) if label == n_idx]

print("=== M SAMPLE 5 ===")
lm_m = m_lms[5]
for idx in [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]:
    print(f"  Point {idx:2d}: x={lm_m[idx][0]:.3f}, y={lm_m[idx][1]:.3f}, z={lm_m[idx][2]:.3f}")

print("\n=== N SAMPLE 5 ===")
lm_n = n_lms[5]
for idx in [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]:
    print(f"  Point {idx:2d}: x={lm_n[idx][0]:.3f}, y={lm_n[idx][1]:.3f}, z={lm_n[idx][2]:.3f}")

