import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

c = VSLAlphabetClassifier()
data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

for target in ["P", "Q", "R", "X"]:
    t_idx = classes.index(target)
    target_lms = [lm for lm, label in zip(landmarks, y) if label == t_idx]
    print(f"\n==================== {target} (Total: {len(target_lms)}) ====================")
    for idx, lm in enumerate(target_lms):
        feat = c.extract_features(lm.tolist())
        pred, conf = c.classify(lm.tolist())
        print(f"Sample {idx:02d}: pred={pred:<6} | d_IW={feat.get('d_IW',0):.2f}, d_MW={feat.get('d_MW',0):.2f}, d_TI={feat.get('d_TI',0):.2f}, idx_tip_y={feat.get('idx_tip_y',0):.2f}, idx_dy={feat.get('idx_dy',0):.2f}, idx_horiz={feat.get('idx_horiz_ratio',0):.2f}, idx_curl={feat.get('idx_curl',0):.2f}, idx_str={feat.get('idx_straight',0):.2f}, ang_IM={feat.get('ang_IM_dir',0):.1f}")

