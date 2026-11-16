import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

c = VSLAlphabetClassifier()

tp = fp = fn = 0
fps = []

for idx, (lm, label) in enumerate(zip(landmarks, y)):
    feat = c.extract_features(lm.tolist())
    actual = classes[label]
    
    d_ti = feat.get("d_TI", 0.0)
    d_tm = feat.get("d_TM", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_iw = feat.get("d_IW", 0.0)
    d_tw = feat.get("d_TW", 0.0)
    d_rw = feat.get("d_RW", 0.0)
    d_pw = feat.get("d_PW", 0.0)
    i_curl = feat.get("index_curl", 0.0)
    idx_str = feat.get("idx_straight", 0.0)
    idx_tip_y = feat.get("idx_tip_y", 0.0)
    idx_dy = feat.get("idx_dy", 0.0)
    
    # X upright (frame 36-48 & webcam upright hooked)
    is_x_upright = (
        d_ti <= 0.25
        and d_iw <= 1.10
        and i_curl >= 0.15
        and idx_str <= 0.85
        and d_rw <= 0.70
        and d_pw <= 0.60
        and 0.70 <= d_tw <= 0.95
        and idx_tip_y <= -0.20
    )
    
    # X tilted/forward (frame 108-136)
    is_x_forward = (
        0.35 <= d_ti <= 0.60
        and 1.40 <= d_iw <= 1.80
        and d_mw >= 0.90
        and d_tm >= 0.35
        and idx_dy >= 0.85
        and 0.80 <= idx_str <= 0.95
        and d_rw <= 0.90
        and d_pw <= 0.85
        and 1.15 <= d_tw <= 1.45
        and 0.35 <= idx_tip_y <= 0.85
    )
    
    is_x = is_x_upright or is_x_forward
    
    if is_x:
        if actual == "X": tp += 1
        else:
            fp += 1
            fps.append((idx, actual))
    else:
        if actual == "X": fn += 1

print(f"X Evaluation: TP={tp}/25 ({tp/25*100:.1f}%), FP={fp} {fps}")
