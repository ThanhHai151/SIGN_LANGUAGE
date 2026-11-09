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
    
    d_iw = feat.get("d_IW", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_ti = feat.get("d_TI", 0.0)
    d_tm = feat.get("d_TM", 0.0)
    d_tw = feat.get("d_TW", 0.0)
    idx_str = feat.get("idx_straight", 0.0)
    idx_tip_y = feat.get("idx_tip_y", 0.0)
    idx_horiz = feat.get("idx_horiz_ratio", 0.0)
    ang_im_dir = feat.get("ang_IM_dir", 0.0)
    idx_dy = feat.get("idx_dy", 0.0)
    ang_ti = feat.get("ang_TI", 0.0)
    mid_str = feat.get("mid_straight", 0.0)
    m_curl = feat.get("middle_curl", 0.0)
    
    # Q variant 1: Q chúc chéo (như hiện tại nhưng ngưỡng bao phủ)
    is_q_diag = (
        1.80 <= d_iw <= 2.25
        and idx_str >= 0.90
        and 0.30 <= idx_tip_y <= 1.70
        and idx_dy >= 0.45
        and 0.50 <= idx_horiz <= 0.90
        and 1.25 <= d_tw <= 1.65
        and 0.50 <= d_ti <= 1.05
        and ang_ti <= 25.0
        and mid_str <= 0.50
        and m_curl >= 0.35
        and ang_im_dir >= 30.0
    )
    
    # Q variant 2: Q chúc thẳng đứng xuống đất (như D0510 frame 42-64 và webcam)
    is_q_vert = (
        d_iw >= 2.20
        and idx_str >= 0.90
        and idx_tip_y >= 1.80
        and idx_dy >= 1.20
        and idx_horiz <= 0.35
        and d_tw >= 1.60
        and ang_ti <= 20.0
        and mid_str <= 0.60
        and ang_im_dir <= 20.0
    )
    
    is_q = is_q_diag or is_q_vert
    
    if is_q:
        if actual == "Q": tp += 1
        else:
            fp += 1
            fps.append((idx, actual))
    else:
        if actual == "Q": fn += 1

print(f"Q Dual Variant: TP={tp}/19 ({tp/19*100:.1f}%), FP={fp} {fps}")

