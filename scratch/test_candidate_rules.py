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

# Proposed thresholds
# Chữ T
T_D_TI_MAX = 0.20
T_D_TM_MIN = 0.70
T_D_IM_MIN = 0.75
T_D_IW_MAX = 1.30
T_D_MW_MIN = 1.65
T_D_RW_MIN = 1.55

# Chữ U
U_D_IW_MIN = 1.65
U_D_MW_MIN = 1.65
U_D_RW_MAX = 1.05
U_D_PW_MAX = 1.05
U_D_IM_MAX = 0.22
U_ANG_IM_MAX = 6.5
U_DIFF_X_IM_MAX = 0.20
U_IDX_HORIZ_MAX = 0.35
U_IDX_TIP_Y_MAX = -0.80

# Chữ S
S_D_TW_MAX = 0.95
S_D_TI_MIN = 0.25
S_D_TI_MAX = 0.45
S_D_TM_MIN = 0.40
S_D_TM_MAX = 0.65
S_D_IW_MAX = 0.90
S_D_MW_MAX = 0.75
S_THUMB_LAT_MIN = 0.50

def eval_candidate_rules(feat):
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
    idx_horiz_ratio = feat.get("idx_horiz_ratio", 0.0)
    idx_tip_y = feat.get("idx_tip_y", 0.0)
    ang_im_dir = feat.get("ang_IM_dir", 0.0)

    matches = []

    # Rule T
    if (d_ti <= T_D_TI_MAX and d_tm >= T_D_TM_MIN and d_im >= T_D_IM_MIN 
        and d_iw <= T_D_IW_MAX and d_mw >= T_D_MW_MIN and d_rw >= T_D_RW_MIN):
        matches.append("T")

    # Rule U
    if i_ext and m_ext and not r_ext and not p_ext:
        if (d_iw >= U_D_IW_MIN and d_mw >= U_D_MW_MIN
            and d_rw <= U_D_RW_MAX and d_pw <= U_D_PW_MAX
            and d_im <= U_D_IM_MAX and ang_im_dir <= U_ANG_IM_MAX
            and diff_x_im <= U_DIFF_X_IM_MAX
            and idx_horiz_ratio <= U_IDX_HORIZ_MAX
            and idx_tip_y <= U_IDX_TIP_Y_MAX):
            matches.append("U")

    # Rule S
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if (d_tw <= S_D_TW_MAX and S_D_TI_MIN <= d_ti <= S_D_TI_MAX
            and S_D_TM_MIN <= d_tm <= S_D_TM_MAX and d_iw <= S_D_IW_MAX and d_mw <= S_D_MW_MAX
            and thumb_lat >= S_THUMB_LAT_MIN):
            matches.append("S")

    return matches

# Test across all 551 samples
results = {c: {"T": 0, "U": 0, "S": 0, "none": 0, "multi": 0} for c in classes}

for i in range(len(y)):
    true_label = classes[y[i]]
    p = lms[i].tolist()
    feat = clf.extract_features(p)
    if not feat:
        results[true_label]["none"] += 1
        continue
    cand = eval_candidate_rules(feat)
    if len(cand) == 1:
        results[true_label][cand[0]] += 1
    elif len(cand) == 0:
        results[true_label]["none"] += 1
    else:
        results[true_label]["multi"] += 1

print("\n=== Candidate Rules Evaluation on all 22 classes (551 samples) ===")
print(f"{'Class':<6} | {'T':<4} {'U':<4} {'S':<4} {'None':<5} {'Multi':<5} | Status")
print("-" * 50)
for c in classes:
    row = results[c]
    status = "OK"
    if c == "T" and row["T"] > 0: status = f"TP={row['T']}/{row['T']+row['none']}"
    elif c == "U" and row["U"] > 0: status = f"TP={row['U']}/{row['U']+row['none']}"
    elif c == "S" and row["S"] > 0: status = f"TP={row['S']}/{row['S']+row['none']}"
    elif c not in ["T", "U", "S"]:
        fps = row["T"] + row["U"] + row["S"]
        if fps > 0:
            status = f"FALSE POSITIVE! (T={row['T']}, U={row['U']}, S={row['S']})"
    print(f"{c:<6} | {row['T']:<4} {row['U']:<4} {row['S']:<4} {row['none']:<5} {row['multi']:<5} | {status}")

