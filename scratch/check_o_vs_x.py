import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

c = VSLAlphabetClassifier()

lm = landmarks[254]
feat = c.extract_features(lm.tolist())
print(f"Sample 254 (O): d_MW={feat['d_MW']:.2f}, d_IW={feat['d_IW']:.2f}, d_TI={feat['d_TI']:.2f}, d_TM={feat['d_TM']:.2f}, ang_IM={feat['ang_IM_dir']:.1f}")

# Also check X forward samples d_MW:
x_idx = classes.index("X")
print("\nX forward samples d_MW:")
for i, (lm, label) in enumerate(zip(landmarks, y)):
    if label == x_idx and i >= 13:
        feat = c.extract_features(lm.tolist())
        print(f"X sample {i}: d_MW={feat['d_MW']:.2f}, d_TI={feat['d_TI']:.2f}, d_TM={feat['d_TM']:.2f}")

