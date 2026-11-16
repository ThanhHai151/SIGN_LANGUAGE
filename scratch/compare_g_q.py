import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

c = VSLAlphabetClassifier()

g_idx = classes.index("G")
q_idx = classes.index("Q")

print("=== Q SAMPLES ===")
for i, (lm, label) in enumerate(zip(landmarks, y)):
    if label == q_idx:
        feat = c.extract_features(lm.tolist())
        print(f"Q {i:02d}: d_IW={feat['d_IW']:.2f}, d_TW={feat['d_TW']:.2f}, d_TI={feat['d_TI']:.2f}, ang_TI={feat['ang_TI']:.1f}, idx_dy={feat['idx_dy']:.2f}, idx_y={feat['idx_tip_y']:.2f}, idx_h={feat['idx_horiz_ratio']:.2f}")

print("\n=== G SAMPLES (111-127) ===")
for i, (lm, label) in enumerate(zip(landmarks, y)):
    if label == g_idx and i >= 110:
        feat = c.extract_features(lm.tolist())
        print(f"G {i:02d}: d_IW={feat['d_IW']:.2f}, d_TW={feat['d_TW']:.2f}, d_TI={feat['d_TI']:.2f}, ang_TI={feat['ang_TI']:.1f}, idx_dy={feat['idx_dy']:.2f}, idx_y={feat['idx_tip_y']:.2f}, idx_h={feat['idx_horiz_ratio']:.2f}")

