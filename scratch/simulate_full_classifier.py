import json
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle, compute_finger_states
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
ds = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(ds['classes'])
y = ds['y']
lms = ds['landmarks']
meta = json.loads(str(ds['metadata_json']))

# Patch VSLAlphabetClassifier in memory to simulate the new rules
# Constants
clf_T_D_TI_MAX = 0.20
clf_T_D_TM_MIN = 0.70
clf_T_D_IM_MIN = 0.75
clf_T_D_IW_MAX = 1.30
clf_T_D_MW_MIN = 1.65
clf_T_D_RW_MIN = 1.55

clf_U_D_IW_MIN = 1.65
clf_U_D_MW_MIN = 1.65
clf_U_D_RW_MAX = 1.05
clf_U_D_PW_MAX = 1.05
clf_U_D_IM_MAX = 0.22
clf_U_ANG_IM_MAX = 6.5
clf_U_DIFF_X_IM_MAX = 0.20
clf_U_IDX_HORIZ_MAX = 0.35
clf_U_IDX_TIP_Y_MAX = -0.80

clf_S_D_TW_MAX = 0.95
clf_S_D_TI_MIN = 0.25
clf_S_D_TI_MAX = 0.45
clf_S_D_TM_MIN = 0.40
clf_S_D_TM_MAX = 0.65
clf_S_D_IW_MAX = 0.90
clf_S_D_MW_MAX = 0.75
clf_S_THUMB_LAT_MIN = 0.50

original_evaluate_rules = clf.evaluate_rules

def patched_evaluate_rules(feat):
    matches = original_evaluate_rules(feat)

    t_ext = feat["thumb_ext"]
    i_ext = feat["index_ext"]
    m_ext = feat["middle_ext"]
    r_ext = feat["ring_ext"]
    p_ext = feat["pinky_ext"]

    d_ti = feat["d_TI"]
    d_tm = feat["d_TM"]
    d_tw = feat["d_TW"]
    d_im = feat["d_IM"]
    d_iw = feat["d_IW"]
    d_mw = feat["d_MW"]
    d_rw = feat["d_RW"]
    d_pw = feat["d_PW"]
    diff_x_im = feat["diff_x_IM"]
    thumb_lat = feat["thumb_lat"]
    idx_horiz_ratio = float(feat.get("idx_horiz_ratio", 0.0))
    idx_tip_y = float(feat.get("idx_tip_y", 0.0))
    ang_im_dir = float(feat.get("ang_IM_dir", 0.0))

    # Overwrite T
    matches = [m for m in matches if m[0] != "T"]
    if (d_ti <= clf_T_D_TI_MAX and d_tm >= clf_T_D_TM_MIN and d_im >= clf_T_D_IM_MIN 
        and d_iw <= clf_T_D_IW_MAX and d_mw >= clf_T_D_MW_MIN and d_rw >= clf_T_D_RW_MIN):
        matches.append(("T", f"Ngón cái chạm trỏ, 3 ngón duỗi: d_TI={d_ti:.2f}, d_IM={d_im:.2f}, d_MW={d_mw:.2f}", 0.95))

    # Add U
    if i_ext and m_ext and not r_ext and not p_ext:
        if (d_iw >= clf_U_D_IW_MIN and d_mw >= clf_U_D_MW_MIN
            and d_rw <= clf_U_D_RW_MAX and d_pw <= clf_U_D_PW_MAX
            and d_im <= clf_U_D_IM_MAX and ang_im_dir <= clf_U_ANG_IM_MAX
            and diff_x_im <= clf_U_DIFF_X_IM_MAX
            and idx_horiz_ratio <= clf_U_IDX_HORIZ_MAX
            and idx_tip_y <= clf_U_IDX_TIP_Y_MAX):
            matches[:] = [m for m in matches if m[0] not in ("V", "D", "L", "H", "K")]
            matches.append(("U", f"Trỏ & Giữa duỗi sát song song chữ U: d_IM={d_im:.2f}<={clf_U_D_IM_MAX}, ang_IM={ang_im_dir:.1f}°<={clf_U_ANG_IM_MAX}°", 0.95))

    # Overwrite S
    matches = [m for m in matches if m[0] != "S"]
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if (d_tw <= clf_S_D_TW_MAX and clf_S_D_TI_MIN <= d_ti <= clf_S_D_TI_MAX
            and clf_S_D_TM_MIN <= d_tm <= clf_S_D_TM_MAX and d_iw <= clf_S_D_IW_MAX and d_mw <= clf_S_D_MW_MAX
            and thumb_lat >= clf_S_THUMB_LAT_MIN):
            matches.append(("S", f"Nắm đấm ngón cái vắt ngang chữ S: d_TI={d_ti:.2f}, thumb_lat={thumb_lat:.2f}>={clf_S_THUMB_LAT_MIN}", 0.95))

    return matches

clf.evaluate_rules = patched_evaluate_rules

# Test on all 551 samples
cm = {c: {c2: 0 for c2 in classes + ['UNKNOWN']} for c in classes}
total_correct = 0

for i in range(len(y)):
    true_label = classes[y[i]]
    p = lms[i].tolist()
    res = clf.classify_detailed(p)
    pred_label = res.get('letter', 'UNKNOWN')
    if pred_label == true_label:
        total_correct += 1
    cm[true_label][pred_label] = cm[true_label].get(pred_label, 0) + 1

print(f"Overall Accuracy: {total_correct}/{len(y)} ({total_correct/len(y)*100:.2f}%)")
print("\nAccuracy & Confusion for T, U, S:")
for let in ['T', 'U', 'S']:
    preds = cm[let]
    non_zero = {k: v for k, v in preds.items() if v > 0}
    tot_let = np.sum(y == classes.index(let))
    rec = preds.get(let, 0) / tot_let * 100
    print(f"Letter {let}: Recall = {preds.get(let, 0)}/{tot_let} ({rec:.1f}%), Predictions: {non_zero}")

# Check precision: are there any other classes predicting T, U, or S?
for let in ['T', 'U', 'S']:
    fp_sources = {}
    for other_cls in classes:
        if other_cls != let and cm[other_cls].get(let, 0) > 0:
            fp_sources[other_cls] = cm[other_cls][let]
    tp = cm[let].get(let, 0)
    fp = sum(fp_sources.values())
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"Letter {let}: Precision = {tp}/{tp+fp} ({prec:.1f}%), F1 = {f1:.1f}%, False Positives from: {fp_sources}")

