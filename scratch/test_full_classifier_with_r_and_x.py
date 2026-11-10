import numpy as np
import json
import sys
import copy
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

# Let's test the full evaluate_rules logic by subclassing or monkey patching
clf = VSLAlphabetClassifier()

# Define R and X thresholds
R_D_IW_MIN = 1.10
R_D_MW_MIN = 1.25
R_D_RW_MAX = 0.65
R_D_PW_MAX = 0.55
R_D_IM_MAX = 0.55
R_IDX_STRAIGHT_MIN = 0.85
R_MID_STRAIGHT_MIN = 0.85
R_IDX_TIP_Y_MAX = -0.80
R_MID_TIP_Y_MAX = -1.00
R_CROSS_IND_MAX = 0.0050
R_D_MW_IW_DIFF_MIN = -0.25

X_D_TI_MAX = 0.25
X_D_IW_MAX = 1.10
X_IDX_CURL_MIN = 0.15
X_IDX_STRAIGHT_MAX = 0.85
X_D_RW_MAX = 0.70
X_D_PW_MAX = 0.60
X_D_TW_MIN = 0.70
X_D_TW_MAX = 0.95
X_IDX_TIP_Y_MAX = -0.20

orig_extract = clf.extract_features

def new_extract_features(landmarks):
    feat = orig_extract(landmarks)
    if not feat:
        return None
    # Calculate cross_indicator
    p = landmarks
    sh0 = p[0]
    sh9 = p[9]
    scale = feat['palm_scale']
    base_dx = float((p[9][0] - p[5][0]) / scale)
    tip_dx = float((p[12][0] - p[8][0]) / scale)
    feat['cross_indicator'] = float(base_dx * tip_dx)
    return feat

clf.extract_features = new_extract_features

orig_evaluate_rules = clf.evaluate_rules

def new_evaluate_rules(feat):
    matches = orig_evaluate_rules(feat)
    
    d_iw = feat.get("d_IW", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_rw = feat.get("d_RW", 0.0)
    d_pw = feat.get("d_PW", 0.0)
    d_ti = feat.get("d_TI", 0.0)
    d_tw = feat.get("d_TW", 0.0)
    d_im = feat.get("d_IM", 0.0)
    idx_straight = feat.get("idx_straight", 0.0)
    mid_straight = feat.get("mid_straight", 0.0)
    idx_curl = feat.get("index_curl", 0.0)
    idx_tip_y = feat.get("idx_tip_y", 0.0)
    mid_tip_y = feat.get("mid_tip_y", 0.0)
    cross_ind = feat.get("cross_indicator", 1.0)
    
    # -------------------------------------------------------------------
    # 17. Chữ R (Ngón giữa bắt chéo qua ngón trỏ - QIPEDC D0511)
    # -------------------------------------------------------------------
    is_r_fingers = (
        d_iw >= R_D_IW_MIN
        and d_mw >= R_D_MW_MIN
        and d_rw <= R_D_RW_MAX
        and d_pw <= R_D_PW_MAX
        and idx_straight >= R_IDX_STRAIGHT_MIN
        and mid_straight >= R_MID_STRAIGHT_MIN
    )
    is_r_upright = (
        idx_tip_y <= R_IDX_TIP_Y_MAX
        and mid_tip_y <= R_MID_TIP_Y_MAX
    )
    is_r_cross = (
        d_im <= R_D_IM_MAX
        and cross_ind <= R_CROSS_IND_MAX
        and (d_mw - d_iw) >= R_D_MW_IW_DIFF_MIN
    )
    if is_r_fingers and is_r_upright and is_r_cross:
        matches[:] = [m for m in matches if m[0] not in ("U", "V", "K", "D")]
        matches.append((
            "R",
            f"Chữ R (trỏ & giữa bắt chéo): d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, cross_ind={cross_ind:.4f}<={R_CROSS_IND_MAX}",
            0.95
        ))

    # -------------------------------------------------------------------
    # 18. Chữ X (Ngón trỏ cong móc chạm ngón cái - QIPEDC D0517B)
    # -------------------------------------------------------------------
    is_x_hook = (
        d_ti <= X_D_TI_MAX
        and d_iw <= X_D_IW_MAX
        and idx_curl >= X_IDX_CURL_MIN
        and idx_straight <= X_IDX_STRAIGHT_MAX
    )
    is_x_curled = (
        d_rw <= X_D_RW_MAX
        and d_pw <= X_D_PW_MAX
    )
    is_x_thumb = (
        X_D_TW_MIN <= d_tw <= X_D_TW_MAX
    )
    is_x_upright = (
        idx_tip_y <= X_IDX_TIP_Y_MAX
    )
    if is_x_hook and is_x_curled and is_x_thumb and is_x_upright:
        matches[:] = [m for m in matches if m[0] not in ("A", "S", "O", "E", "C")]
        matches.append((
            "X",
            f"Chữ X (trỏ cong móc chạm cái): d_TI={d_ti:.2f}<={X_D_TI_MAX}, d_TW={d_tw:.2f}, idx_curl={idx_curl:.2f}>={X_IDX_CURL_MIN}",
            0.95
        ))
        
    return matches

clf.evaluate_rules = new_evaluate_rules

print("=" * 80)
print("TESTING FULL CLASSIFIER ON 551 SAMPLES IN vsl_alphabet_static_v2.npz")
print("=" * 80)

data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
classes = list(data['classes'])
y = data['y']
lms = data['landmarks']

tp = {}
fp = {}
unknown = {}

for idx in range(len(y)):
    m = meta[idx]
    true_label = m['label']
    lm = lms[idx].tolist()
    
    res = clf.classify_detailed(lm)
    pred_label = res['letter']
    
    if pred_label == true_label:
        tp[true_label] = tp.get(true_label, 0) + 1
    elif pred_label in [None, "UNKNOWN"]:
        unknown[true_label] = unknown.get(true_label, 0) + 1
    else:
        # False Positive / Misclassification
        fp[pred_label] = fp.get(pred_label, []) + [(true_label, m['frame_index'])]

print("PER-CLASS EVALUATION ON 551 SAMPLES:")
for c in ['Q', 'R', 'S', 'X']:
    total_c = np.sum(y == classes.index(c))
    c_tp = tp.get(c, 0)
    c_fp = len(fp.get(c, []))
    c_unk = unknown.get(c, 0)
    print(f"Class {c:2s}: Total={total_c:2d}, TP={c_tp:2d}, FP={c_fp:2d}, UNKNOWN={c_unk:2d}")
    if c_fp > 0:
        print(f"  False positives for {c}: {fp[c]}")

print("\nCHECKING ANY FALSE POSITIVES ASSIGNED AS R OR X ACROSS ALL 551 SAMPLES:")
print(f"  False Positives labeled as R: {fp.get('R', [])}")
print(f"  False Positives labeled as X: {fp.get('X', [])}")
print(f"  False Positives labeled as Q: {fp.get('Q', [])}")
print(f"  False Positives labeled as S: {fp.get('S', [])}")


