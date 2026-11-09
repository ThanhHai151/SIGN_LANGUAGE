import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load("backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz", allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

c = VSLAlphabetClassifier()

print(f"Total samples in NPZ: {len(landmarks)}, classes: {len(classes)}")

# Let's test a proposed rule function for P, Q, X
def test_proposed_rules(p_ang_min=55.0, p_d_ti_min=1.15, p_d_tm_max=0.75, p_horiz_max=0.45,
                        q_horiz_min=0.0, q_tip_y_max=2.85, q_d_iw_max=3.00, q_d_tw_max=3.50, q_ang_ti_max=26.0, q_m_curl_min=0.25):
    # Count TP, FP, FN
    stats = {k: {"tp": 0, "fn": 0, "fp": 0, "fps": []} for k in ["P", "Q", "R", "X"]}
    
    for idx, (lm, label) in enumerate(zip(landmarks, y)):
        feat = c.extract_features(lm.tolist())
        actual = classes[label]
        
        # Test original classify first
        pred_orig, _ = c.classify(lm.tolist())
        
        # Now evaluate with custom conditions for P, Q
        # Let's see what matches
        d_iw = feat.get("d_IW", 0.0)
        d_mw = feat.get("d_MW", 0.0)
        d_ti = feat.get("d_TI", 0.0)
        d_tm = feat.get("d_TM", 0.0)
        d_tw = feat.get("d_TW", 0.0)
        idx_str = feat.get("idx_straight", 0.0)
        idx_tip_y = feat.get("idx_tip_y", 0.0)
        idx_horiz = feat.get("idx_horiz_ratio", 0.0)
        ang_im_dir = feat.get("ang_IM_dir", 0.0)
        thumb_lat = feat.get("thumb_lat", 0.0)
        idx_dy = feat.get("idx_dy", 0.0)
        ang_ti = feat.get("ang_TI", 0.0)
        mid_str = feat.get("mid_straight", 0.0)
        m_curl = feat.get("middle_curl", 0.0)
        
        # P proposal
        is_p = (
            d_iw >= 1.65
            and idx_str >= 0.95
            and idx_tip_y <= -1.40
            and idx_horiz <= p_horiz_max
            and d_mw <= 1.25
            and d_ti >= p_d_ti_min
            and d_tm <= p_d_tm_max
            and d_tw <= 1.05
            and thumb_lat >= 0.65
            and ang_im_dir >= p_ang_min
        )
        
        if is_p:
            if actual == "P": stats["P"]["tp"] += 1
            else:
                stats["P"]["fp"] += 1
                stats["P"]["fps"].append((idx, actual))
        else:
            if actual == "P": stats["P"]["fn"] += 1
            
        # Q proposal
        is_q = (
            1.80 <= d_iw <= q_d_iw_max
            and idx_str >= 0.85
            and (0.30 <= idx_tip_y <= q_tip_y_max)
            and idx_dy >= 0.40
            and 1.20 <= d_tw <= q_d_tw_max
            and 0.25 <= d_ti <= 1.10
            and ang_ti <= q_ang_ti_max
            and mid_str <= 0.65
            and m_curl >= q_m_curl_min
            and ang_im_dir >= 25.0
        )
        if is_q:
            if actual == "Q": stats["Q"]["tp"] += 1
            else:
                stats["Q"]["fp"] += 1
                stats["Q"]["fps"].append((idx, actual))
        else:
            if actual == "Q": stats["Q"]["fn"] += 1
            
    print(f"P: TP={stats['P']['tp']}/20 ({stats['P']['tp']/20*100:.1f}%), FP={stats['P']['fp']} {stats['P']['fps']}")
    print(f"Q: TP={stats['Q']['tp']}/19 ({stats['Q']['tp']/19*100:.1f}%), FP={stats['Q']['fp']} {stats['Q']['fps']}")

test_proposed_rules()

