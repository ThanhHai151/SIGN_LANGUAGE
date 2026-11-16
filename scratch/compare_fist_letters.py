"""
So sánh đặc trưng hình học giữa M, N và các chữ nắm tay A, S, T, E trong NPZ.
"""
import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

c = VSLAlphabetClassifier()

target_classes = ["M", "N", "A", "S", "T", "E"]

stats = {ch: [] for ch in target_classes}

for lm, label in zip(landmarks, y):
    ch = classes[label]
    if ch in target_classes:
        f = c.extract_features(lm.tolist())
        stats[ch].append(f)

print(f"{'Chữ':<4} {'Số mẫu':<8} {'d_TI':<10} {'d_TM':<10} {'d_TW':<10} {'idx_curl':<10} {'mid_curl':<10} {'ring_curl':<10} {'pinky_curl':<10} {'ang_TI':<10}")
print("-" * 95)

for ch in target_classes:
    flist = stats[ch]
    n = len(flist)
    d_ti = np.mean([f['d_TI'] for f in flist])
    d_tm = np.mean([f['d_TM'] for f in flist])
    d_tw = np.mean([f['d_TW'] for f in flist])
    i_c = np.mean([f['index_curl'] for f in flist])
    m_c = np.mean([f['middle_curl'] for f in flist])
    r_c = np.mean([f['ring_curl'] for f in flist])
    p_c = np.mean([f['pinky_curl'] for f in flist])
    ang_ti = np.mean([f['ang_TI'] for f in flist])
    print(f"{ch:<4} {n:<8} {d_ti:<10.2f} {d_tm:<10.2f} {d_tw:<10.2f} {i_c:<10.2f} {m_c:<10.2f} {r_c:<10.2f} {p_c:<10.2f} {ang_ti:<10.1f}")

