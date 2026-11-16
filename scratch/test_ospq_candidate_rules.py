import sys
sys.path.insert(0, '.')
import json
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(data['classes'])
landmarks = data['landmarks']
y = data['y']

# Candidate rules for O, S, P, Q

def eval_candidate_o(feat):
    d_ti = feat.get('d_TI', 99.0)
    d_tm = feat.get('d_TM', 99.0)
    d_iw = feat.get('d_IW', 0.0)
    d_mw = feat.get('d_MW', 0.0)
    i_ext = feat.get('index_ext', False)
    m_ext = feat.get('middle_ext', False)
    r_ext = feat.get('ring_ext', False)
    p_ext = feat.get('pinky_ext', False)
    
    # O: All fingers curled into a hollow circle meeting thumb tip
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if d_ti <= 0.22 and d_tm <= 0.22 and d_iw >= 0.95 and d_mw >= 0.90:
            return True
    return False

def eval_candidate_s(feat):
    d_tw = feat.get('d_TW', 99.0)
    d_ti = feat.get('d_TI', 99.0)
    d_tm = feat.get('d_TM', 99.0)
    d_iw = feat.get('d_IW', 99.0)
    d_mw = feat.get('d_MW', 99.0)
    thumb_lat = feat.get('thumb_lat', 0.0)
    i_ext = feat.get('index_ext', False)
    m_ext = feat.get('middle_ext', False)
    r_ext = feat.get('ring_ext', False)
    p_ext = feat.get('pinky_ext', False)
    
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if (d_tw <= 0.95 and 0.25 <= d_ti <= 0.45 and
            0.40 <= d_tm <= 0.65 and d_iw <= 0.90 and d_mw <= 0.75 and
            thumb_lat >= 0.50):
            return True
    return False

def eval_candidate_p(feat):
    d_iw = feat.get('d_IW', 0.0)
    d_mw = feat.get('d_MW', 99.0)
    d_rw = feat.get('d_RW', 99.0)
    d_pw = feat.get('d_PW', 99.0)
    d_ti = feat.get('d_TI', 0.0)
    d_tm = feat.get('d_TM', 99.0)
    d_tw = feat.get('d_TW', 99.0)
    thumb_lat = feat.get('thumb_lat', 0.0)
    ang_im = feat.get('ang_IM_dir', 0.0)
    idx_tip_y = feat.get('idx_tip_y', 0.0)
    idx_straight = feat.get('idx_straight', 0.0)
    idx_horiz = feat.get('idx_horiz_ratio', 1.0)
    
    # P: Index straight up, middle forward/angled, thumb touching middle, wide angle, high thumb_lat
    is_p_index = (d_iw >= 1.70 and idx_straight >= 0.95 and idx_tip_y <= -1.60 and idx_horiz <= 0.20)
    is_p_middle = (d_mw <= 1.25)
    is_p_thumb = (d_ti >= 1.35 and d_tm <= 0.55 and d_tw <= 1.05 and thumb_lat >= 0.65)
    is_p_angle = (ang_im >= 105.0)
    
    return is_p_index and is_p_middle and is_p_thumb and is_p_angle

def eval_candidate_q(feat):
    d_iw = feat.get('d_IW', 0.0)
    d_mw = feat.get('d_MW', 99.0)
    d_ti = feat.get('d_TI', 0.0)
    d_tw = feat.get('d_TW', 0.0)
    idx_tip_y = feat.get('idx_tip_y', -99.0)
    idx_straight = feat.get('idx_straight', 0.0)
    mid_straight = feat.get('mid_straight', 1.0)
    
    # Q: Index pointing downwards, thumb pointing downwards, pinch angle, middle curled
    is_q_index = (d_iw >= 1.80 and idx_straight >= 0.90 and idx_tip_y >= 0.30)
    is_q_thumb = (d_tw >= 1.25 and 0.20 <= d_ti <= 1.05)
    is_q_diff = (mid_straight <= 0.60 or (d_iw - d_mw) >= 0.40)
    
    return is_q_index and is_q_thumb and is_q_diff

print("=== TESTING CANDIDATE RULES ON 551 SAMPLES ===")
eval_fns = {"O": eval_candidate_o, "S": eval_candidate_s, "P": eval_candidate_p, "Q": eval_candidate_q}

for let, fn in eval_fns.items():
    tp = 0
    fp = 0
    fn_cnt = 0
    true_cnt = 0
    fp_classes = {}
    
    for i in range(len(y)):
        true_c = classes[y[i]]
        lm = landmarks[i].tolist()
        feat = clf.extract_features(lm)
        if not feat: continue
        
        pred = fn(feat)
        if true_c == let:
            true_cnt += 1
            if pred: tp += 1
            else: fn_cnt += 1
        else:
            if pred:
                fp += 1
                fp_classes[true_c] = fp_classes.get(true_c, 0) + 1
                
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn_cnt) if (tp + fn_cnt) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"\nRule {let}:")
    print(f"  True={true_cnt}, TP={tp}, FP={fp}, FN={fn_cnt}")
    print(f"  Precision={prec*100:.2f}%, Recall={rec*100:.2f}%, F1={f1*100:.2f}%")
    if fp > 0:
        print(f"  False Positives from: {fp_classes}")

