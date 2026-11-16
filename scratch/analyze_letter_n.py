import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_samples = json.load(f)['samples']

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_samples = json.load(f)['samples']

def extract_detailed_hand_metrics(p):
    if not p or len(p) != 21:
        return None
    wrist = p[0]
    sh, ps = _shift_and_scale(p)
    if ps < 1e-5:
        return None
    
    # 1. Distances to wrist normalized by palm_scale
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

    # 3. Downward direction vector ratios (vy / len in 2D)
    vx_i, vy_i = sh[8][0] - sh[5][0], sh[8][1] - sh[5][1]
    len_i = math.hypot(vx_i, vy_i) + 1e-8
    v_down_i = vy_i / len_i

    vx_m, vy_m = sh[12][0] - sh[9][0], sh[12][1] - sh[9][1]
    len_m = math.hypot(vx_m, vy_m) + 1e-8
    v_down_m = vy_m / len_m

    vx_r, vy_r = sh[16][0] - sh[13][0], sh[16][1] - sh[13][1]
    len_r = math.hypot(vx_r, vy_r) + 1e-8
    v_down_r = vy_r / len_r

    # 4. Joint angles
    ang_i_pip = calculate_angle(sh[5], sh[6], sh[7])
    ang_i_dip = calculate_angle(sh[6], sh[7], sh[8])
    ang_m_pip = calculate_angle(sh[9], sh[10], sh[11])
    ang_m_dip = calculate_angle(sh[10], sh[11], sh[12])
    ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
    ang_r_dip = calculate_angle(sh[14], sh[15], sh[16])
    ang_p_pip = calculate_angle(sh[17], sh[18], sh[19])

    # 5. Key dimensionless ratios
    ratio_iw_mw = d_iw / d_mw if d_mw > 1e-5 else 0.0
    ratio_rw_mw = d_rw / d_mw if d_mw > 1e-5 else 0.0
    ratio_pw_mw = d_pw / d_mw if d_mw > 1e-5 else 0.0
    ratio_mr_mw = d_mr / d_mw if d_mw > 1e-5 else 0.0
    ratio_im_mw = d_im / d_mw if d_mw > 1e-5 else 0.0
    ratio_pw_rw = d_pw / d_rw if d_rw > 1e-5 else 0.0

    diff_mw_rw = d_mw - d_rw
    diff_rw_pw = d_rw - d_pw

    return {
        "wrist_y": wrist[1],
        "palm_scale": ps,
        "d_TW": d_tw,
        "d_IW": d_iw,
        "d_MW": d_mw,
        "d_RW": d_rw,
        "d_PW": d_pw,
        "d_TI": d_ti,
        "d_TM": d_tm,
        "d_IM": d_im,
        "d_MR": d_mr,
        "d_RP": d_rp,
        "v_down_i": v_down_i,
        "v_down_m": v_down_m,
        "v_down_r": v_down_r,
        "ang_i_pip": ang_i_pip,
        "ang_m_pip": ang_m_pip,
        "ang_r_pip": ang_r_pip,
        "ang_p_pip": ang_p_pip,
        "ratio_iw_mw": ratio_iw_mw,
        "ratio_rw_mw": ratio_rw_mw,
        "ratio_pw_mw": ratio_pw_mw,
        "ratio_mr_mw": ratio_mr_mw,
        "ratio_im_mw": ratio_im_mw,
        "ratio_pw_rw": ratio_pw_rw,
        "diff_mw_rw": diff_mw_rw,
        "diff_rw_pw": diff_rw_pw,
    }

# Collect stable frames
n_metrics = []
for s in n_samples:
    if s.get("phase") == "stable_pose" and s.get("quality_status") == "ok":
        m = extract_detailed_hand_metrics(s.get("landmarks"))
        if m:
            m["frame_index"] = s.get("frame_index")
            n_metrics.append(m)

m_metrics = []
for s in m_samples:
    if s.get("phase") == "stable_pose" and s.get("quality_status") == "ok":
        m = extract_detailed_hand_metrics(s.get("landmarks"))
        if m:
            m["frame_index"] = s.get("frame_index")
            m_metrics.append(m)

print(f"Collected {len(n_metrics)} N stable frames, {len(m_metrics)} M stable frames.")

# Print individual N frames
print("\n=== N STABLE FRAMES BREAKDOWN ===")
header = f"{'Frame':>5} | {'WristY':>6} | {'d_IW':>5} | {'d_MW':>5} | {'d_RW':>5} | {'d_PW':>5} | {'RW/MW':>6} | {'d_MR':>5} | {'MR/MW':>6} | {'R_PIP':>5} | {'MW-RW':>6}"
print(header)
print("-" * len(header))
for m in n_metrics:
    print(f"{m['frame_index']:5d} | {m['wrist_y']:6.2f} | {m['d_IW']:5.2f} | {m['d_MW']:5.2f} | {m['d_RW']:5.2f} | {m['d_PW']:5.2f} | {m['ratio_rw_mw']:6.2f} | {m['d_MR']:5.2f} | {m['ratio_mr_mw']:6.2f} | {m['ang_r_pip']:5.1f} | {m['diff_mw_rw']:6.2f}")

