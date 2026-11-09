import json
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle
from backend.vision.alphabet_classifier import alphabet_classifier

# Load O.json
with open('backend/data/alphabet_v2/static_landmarks/O.json', 'r', encoding='utf-8') as f:
    o_json = json.load(f)

# Load dataset v2
npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes'].tolist()
y = npz['y']
lms = npz['landmarks']

def extract_metrics(lm):
    if not lm or len(lm) != 21:
        return None
    sh, ps = _shift_and_scale(lm)
    if ps < 1e-5:
        return None
    
    # Distances from wrist
    d_tw = calculate_distance(sh[4], sh[0]) / ps
    d_iw = calculate_distance(sh[8], sh[0]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_rw = calculate_distance(sh[16], sh[0]) / ps
    d_pw = calculate_distance(sh[20], sh[0]) / ps

    # Distances to thumb tip (landmark 4)
    d_ti = calculate_distance(sh[4], sh[8]) / ps
    d_tm = calculate_distance(sh[4], sh[12]) / ps
    d_tr = calculate_distance(sh[4], sh[16]) / ps
    d_tp = calculate_distance(sh[4], sh[20]) / ps

    # Pairwise fingertip distances
    d_im = calculate_distance(sh[8], sh[12]) / ps
    d_mr = calculate_distance(sh[12], sh[16]) / ps
    d_rp = calculate_distance(sh[16], sh[20]) / ps

    # Finger curl metrics (dist from tip to MCP normalized by MCP-to-wrist or finger segment)
    # Angles at MCP (0-x-x), PIP (x-x-x), DIP (x-x-x)
    ang_t_cmc = calculate_angle(sh[0], sh[1], sh[2])
    ang_t_mcp = calculate_angle(sh[1], sh[2], sh[3])
    ang_t_ip = calculate_angle(sh[2], sh[3], sh[4])

    ang_i_mcp = calculate_angle(sh[0], sh[5], sh[6])
    ang_i_pip = calculate_angle(sh[5], sh[6], sh[7])
    ang_i_dip = calculate_angle(sh[6], sh[7], sh[8])

    ang_m_mcp = calculate_angle(sh[0], sh[9], sh[10])
    ang_m_pip = calculate_angle(sh[9], sh[10], sh[11])
    ang_m_dip = calculate_angle(sh[10], sh[11], sh[12])

    ang_r_mcp = calculate_angle(sh[0], sh[13], sh[14])
    ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
    ang_r_dip = calculate_angle(sh[14], sh[15], sh[16])

    ang_p_mcp = calculate_angle(sh[0], sh[17], sh[18])
    ang_p_pip = calculate_angle(sh[17], sh[18], sh[19])
    ang_p_dip = calculate_angle(sh[18], sh[19], sh[20])

    wrist_y = lm[0][1]

    return {
        'wrist_y': wrist_y,
        'd_TW': d_tw,
        'd_IW': d_iw,
        'd_MW': d_mw,
        'd_RW': d_rw,
        'd_PW': d_pw,
        'd_TI': d_ti,
        'd_TM': d_tm,
        'd_TR': d_tr,
        'd_TP': d_tp,
        'd_IM': d_im,
        'd_MR': d_mr,
        'd_RP': d_rp,
        'ang_i_pip': ang_i_pip,
        'ang_m_pip': ang_m_pip,
        'ang_r_pip': ang_r_pip,
        'ang_p_pip': ang_p_pip,
        'ang_i_mcp': ang_i_mcp,
        'ang_m_mcp': ang_m_mcp,
    }

# Collect O metrics from stable_pose in O.json (quality ok)
o_metrics = []
for s in o_json['samples']:
    if s.get('phase') == 'stable_pose' and s.get('landmarks'):
        m = extract_metrics(s['landmarks'])
        if m:
            o_metrics.append(m)

# Collect O metrics from dataset v2
o_idx = classes.index('O')
for i in range(len(y)):
    if y[i] == o_idx:
        m = extract_metrics(lms[i].tolist())
        if m:
            o_metrics.append(m)

print(f"Total O samples collected for stats: {len(o_metrics)}")

# Compare with confusable letters: C, A, E, S, T, B
def get_letter_metrics(letter):
    res = []
    idx = classes.index(letter)
    for i in range(len(y)):
        if y[i] == idx:
            m = extract_metrics(lms[i].tolist())
            if m: res.append(m)
    return res

c_metrics = get_letter_metrics('C')
a_metrics = get_letter_metrics('A')
e_metrics = get_letter_metrics('E')
s_metrics = get_letter_metrics('S')
t_metrics = get_letter_metrics('T')
b_metrics = get_letter_metrics('B')
m_metrics = get_letter_metrics('M')
n_metrics = get_letter_metrics('N')

features_to_report = [
    'd_TI', 'd_TM', 'd_TR', 'd_TP',
    'd_IW', 'd_MW', 'd_RW', 'd_PW', 'd_TW',
    'd_IM', 'd_MR',
    'ang_i_pip', 'ang_m_pip', 'ang_r_pip', 'ang_p_pip'
]

print(f"{'Feature':12s} | {'O Mean':7s} {'O Std':6s} {'O [Min, Max]':16s} | {'C Mean':7s} {'A Mean':7s} {'E Mean':7s} {'S Mean':7s} {'T Mean':7s} {'B Mean':7s}")
print("-" * 105)

for feat in features_to_report:
    o_vals = [m[feat] for m in o_metrics]
    c_vals = [m[feat] for m in c_metrics]
    a_vals = [m[feat] for m in a_metrics]
    e_vals = [m[feat] for m in e_metrics]
    s_vals = [m[feat] for m in s_metrics]
    t_vals = [m[feat] for m in t_metrics]
    b_vals = [m[feat] for m in b_metrics]

    o_mean, o_std = np.mean(o_vals), np.std(o_vals)
    o_min, o_max = np.min(o_vals), np.max(o_vals)
    c_mean = np.mean(c_vals)
    a_mean = np.mean(a_vals)
    e_mean = np.mean(e_vals)
    s_mean = np.mean(s_vals)
    t_mean = np.mean(t_vals)
    b_mean = np.mean(b_vals)

    print(f"{feat:12s} | {o_mean:7.2f} {o_std:6.2f} [{o_min:6.2f}, {o_max:6.2f}] | {c_mean:7.2f} {a_mean:7.2f} {e_mean:7.2f} {s_mean:7.2f} {t_mean:7.2f} {b_mean:7.2f}")

