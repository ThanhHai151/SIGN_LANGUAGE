import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle, finger_curl_ratio, compute_finger_states
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# Load real landmark datasets
ds_static = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(ds_static['classes'])
y = ds_static['y']
lms = ds_static['landmarks']
meta = json.loads(str(ds_static['metadata_json']))

def extract_all_metrics(p):
    if not p or len(p) != 21:
        return None
    wrist = p[0]
    sh, ps = _shift_and_scale(p)
    if ps < 1e-5:
        return None

    # Base features from classifier
    clf_feat = clf.extract_features(p)
    states = compute_finger_states(sh, ps)

    # 1. Distances to wrist
    d_tw = calculate_distance(sh[4], sh[0]) / ps
    d_iw = calculate_distance(sh[8], sh[0]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_rw = calculate_distance(sh[16], sh[0]) / ps
    d_pw = calculate_distance(sh[20], sh[0]) / ps

    # 2. Inter-fingertip distances
    d_ti = calculate_distance(sh[4], sh[8]) / ps
    d_tm = calculate_distance(sh[4], sh[12]) / ps
    d_tr = calculate_distance(sh[4], sh[16]) / ps
    d_tp = calculate_distance(sh[4], sh[20]) / ps
    d_im = calculate_distance(sh[8], sh[12]) / ps
    d_mr = calculate_distance(sh[12], sh[16]) / ps
    d_rp = calculate_distance(sh[16], sh[20]) / ps

    # 3. Thumb tip to other joints
    d_t4_i5 = calculate_distance(sh[4], sh[5]) / ps   # Thumb tip to Index MCP
    d_t4_i6 = calculate_distance(sh[4], sh[6]) / ps   # Thumb tip to Index PIP
    d_t4_m9 = calculate_distance(sh[4], sh[9]) / ps   # Thumb tip to Middle MCP
    d_t4_m10 = calculate_distance(sh[4], sh[10]) / ps # Thumb tip to Middle PIP

    # 4. Angles
    # Joint angles
    ang_t_ip = calculate_angle(sh[2], sh[3], sh[4])
    ang_t_mcp = calculate_angle(sh[1], sh[2], sh[3])
    ang_i_pip = calculate_angle(sh[5], sh[6], sh[7])
    ang_i_dip = calculate_angle(sh[6], sh[7], sh[8])
    ang_m_pip = calculate_angle(sh[9], sh[10], sh[11])
    ang_m_dip = calculate_angle(sh[10], sh[11], sh[12])
    ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
    ang_p_pip = calculate_angle(sh[17], sh[18], sh[19])

    # Inter-finger angles
    ang_ti = calculate_angle(sh[4], sh[0], sh[8])
    ang_tm = calculate_angle(sh[4], sh[0], sh[12])
    ang_im = calculate_angle(sh[8], sh[0], sh[12])
    ang_tp = calculate_angle(sh[4], sh[0], sh[20])

    # Direction vectors for index & middle
    vx_i, vy_i = sh[8][0] - sh[5][0], sh[8][1] - sh[5][1]
    len_i = math.hypot(vx_i, vy_i) + 1e-8
    vx_m, vy_m = sh[12][0] - sh[9][0], sh[12][1] - sh[9][1]
    len_m = math.hypot(vx_m, vy_m) + 1e-8
    cos_im = max(-1.0, min(1.0, (vx_i * vx_m + vy_i * vy_m) / (len_i * len_m)))
    ang_im_dir = math.degrees(math.acos(cos_im))

    # Curls
    t_curl = states.get("thumb_curl", 0.0)
    i_curl = states.get("index_curl", 0.0)
    m_curl = states.get("middle_curl", 0.0)
    r_curl = states.get("ring_curl", 0.0)
    p_curl = states.get("pinky_curl", 0.0)

    # Relative coordinates of thumb tip to index/middle
    rel_tx_i6 = (sh[4][0] - sh[6][0]) / ps
    rel_ty_i6 = (sh[4][1] - sh[6][1]) / ps
    rel_tz_i6 = (sh[4][2] - sh[6][2]) / ps

    rel_tx_m10 = (sh[4][0] - sh[10][0]) / ps
    rel_ty_m10 = (sh[4][1] - sh[10][1]) / ps

    # Thumb lateral
    thumb_lat = abs(sh[4][0] - sh[0][0]) / ps

    # Finger straightness
    idx_straight = clf_feat.get("idx_straight", 0.0) if clf_feat else 0.0
    mid_straight = clf_feat.get("mid_straight", 0.0) if clf_feat else 0.0

    return {
        "thumb_ext": states["thumb_extended"],
        "index_ext": states["index_extended"],
        "middle_ext": states["middle_extended"],
        "ring_ext": states["ring_extended"],
        "pinky_ext": states["pinky_extended"],
        "t_curl": t_curl, "i_curl": i_curl, "m_curl": m_curl, "r_curl": r_curl, "p_curl": p_curl,
        "d_tw": d_tw, "d_iw": d_iw, "d_mw": d_mw, "d_rw": d_rw, "d_pw": d_pw,
        "d_ti": d_ti, "d_tm": d_tm, "d_tr": d_tr, "d_tp": d_tp,
        "d_im": d_im, "d_mr": d_mr, "d_rp": d_rp,
        "d_t4_i5": d_t4_i5, "d_t4_i6": d_t4_i6, "d_t4_m9": d_t4_m9, "d_t4_m10": d_t4_m10,
        "ang_t_ip": ang_t_ip, "ang_t_mcp": ang_t_mcp,
        "ang_i_pip": ang_i_pip, "ang_i_dip": ang_i_dip,
        "ang_m_pip": ang_m_pip, "ang_m_dip": ang_m_dip,
        "ang_r_pip": ang_r_pip, "ang_p_pip": ang_p_pip,
        "ang_ti": ang_ti, "ang_tm": ang_tm, "ang_im": ang_im, "ang_tp": ang_tp,
        "ang_im_dir": ang_im_dir,
        "thumb_lat": thumb_lat,
        "idx_straight": idx_straight, "mid_straight": mid_straight,
        "rel_tx_i6": rel_tx_i6, "rel_ty_i6": rel_ty_i6, "rel_tz_i6": rel_tz_i6,
        "rel_tx_m10": rel_tx_m10, "rel_ty_m10": rel_ty_m10,
    }

# Collect metrics for T, U, S from static_v2
data_by_letter = {}
for letter in ['T', 'U', 'S']:
    idx = classes.index(letter)
    indices = np.where(y == idx)[0]
    rows = []
    for i in indices:
        m = meta[i]
        lm = lms[i].tolist()
        met = extract_all_metrics(lm)
        if met:
            met['frame_index'] = m.get('frame_index')
            met['video_id'] = m.get('video_id')
            rows.append(met)
    data_by_letter[letter] = rows
    print(f"Extracted {len(rows)} samples for {letter}")

# Also check pure U (D0514) vs U-from-Ư (D0515N)
u_pure = [r for r in data_by_letter['U'] if r.get('video_id') == 'D0514']
u_u_hook = [r for r in data_by_letter['U'] if r.get('video_id') == 'D0515N']
print(f"Letter U: {len(u_pure)} from D0514 (pure U), {len(u_u_hook)} from D0515N (Ư base)")

# Compute statistics: mean, std, min, max
stat_keys = [
    "t_curl", "i_curl", "m_curl", "r_curl", "p_curl",
    "d_tw", "d_iw", "d_mw", "d_rw", "d_pw",
    "d_ti", "d_tm", "d_im", "d_mr", "d_rp",
    "d_t4_i5", "d_t4_i6", "d_t4_m9", "d_t4_m10",
    "ang_t_ip", "ang_i_pip", "ang_m_pip", "ang_r_pip", "ang_p_pip",
    "ang_ti", "ang_tm", "ang_im", "ang_im_dir",
    "thumb_lat", "idx_straight", "mid_straight",
    "rel_tx_i6", "rel_ty_i6", "rel_tz_i6"
]

print("\n" + "="*80)
print(f"{'Feature':<16} | {'T (mean +/- std, [min, max])':<30} | {'U (mean +/- std, [min, max])':<30} | {'S (mean +/- std, [min, max])':<30}")
print("="*80)

stats = {}
for k in stat_keys:
    stats[k] = {}
    row_strs = []
    for letter in ['T', 'U', 'S']:
        vals = [r[k] for r in data_by_letter[letter] if k in r and r[k] is not None]
        arr = np.array(vals)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        stats[k][letter] = {"mean": mean, "std": std, "min": vmin, "max": vmax}
        row_strs.append(f"{mean:.2f}+/-{std:.2f} [{vmin:.2f},{vmax:.2f}]")
    print(f"{k:<16} | {row_strs[0]:<30} | {row_strs[1]:<30} | {row_strs[2]:<30}")

