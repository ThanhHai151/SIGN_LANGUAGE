import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

c = VSLAlphabetClassifier()

for idx in [112, 115]:
    lm = landmarks[idx]
    feat = c.extract_features(lm.tolist())
    print(f"Sample {idx} (G): idx_tip_y={feat['idx_tip_y']:.2f}, idx_dy={feat['idx_dy']:.2f}, idx_horiz={feat['idx_horiz_ratio']:.2f}, ang_im_dir={feat['ang_IM_dir']:.1f}")

# Also check all G samples:
g_idx = classes.index("G")
print("\nAll G samples:")
for i, (lm, label) in enumerate(zip(landmarks, y)):
    if label == g_idx:
        feat = c.extract_features(lm.tolist())
        print(f"G sample {i}: idx_tip_y={feat['idx_tip_y']:.2f}, idx_dy={feat['idx_dy']:.2f}, idx_horiz={feat['idx_horiz_ratio']:.2f}, d_IW={feat['d_IW']:.2f}")

