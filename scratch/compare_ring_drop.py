"""
Kiểm tra độ chênh lệch y của ngón áp út (16) và ngón giữa (12) trên toàn bộ mẫu M và N.
"""
import numpy as np

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

m_idx = classes.index("M")
n_idx = classes.index("N")

print("=== CHỮ M (19 mẫu): So sánh y_16 (áp út) và y_12 (giữa) ===")
m_diffs = []
for i, (lm, label) in enumerate(zip(landmarks, y)):
    if label == m_idx:
        wrist_y = lm[0][1]
        mid_mcp_y = lm[9][1]
        palm = np.linalg.norm(np.array(lm[9]) - np.array(lm[0]))
        # y_16 vs y_12
        y_mid = lm[12][1]
        y_ring = lm[16][1]
        diff = (y_mid - y_ring) / palm
        m_diffs.append(diff)
        print(f"  M {i:02d}: y_mid={y_mid:.3f}, y_ring={y_ring:.3f}, diff(mid-ring)={diff:.3f}")

print("\n=== CHỮ N (21 mẫu): So sánh y_16 (áp út) và y_12 (giữa) ===")
n_diffs = []
for i, (lm, label) in enumerate(zip(landmarks, y)):
    if label == n_idx:
        palm = np.linalg.norm(np.array(lm[9]) - np.array(lm[0]))
        y_mid = lm[12][1]
        y_ring = lm[16][1]
        diff = (y_mid - y_ring) / palm
        n_diffs.append(diff)
        print(f"  N {i:02d}: y_mid={y_mid:.3f}, y_ring={y_ring:.3f}, diff(mid-ring)={diff:.3f}")

print(f"\nM: diff(mid - ring) min={min(m_diffs):.3f}, max={max(m_diffs):.3f}, mean={np.mean(m_diffs):.3f}")
print(f"N: diff(mid - ring) min={min(n_diffs):.3f}, max={max(n_diffs):.3f}, mean={np.mean(n_diffs):.3f}")

