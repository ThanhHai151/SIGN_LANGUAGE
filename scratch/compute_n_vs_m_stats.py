import numpy as np
import json
import math
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_samples = [s for s in json.load(f)['samples'] if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok']

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_samples = [s for s in json.load(f)['samples'] if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok']

def compute_all_metrics(samples):
    metrics = {
        'wrist_y': [],
        'd_TW': [], 'd_IW': [], 'd_MW': [], 'd_RW': [], 'd_PW': [],
        'd_TI': [], 'd_TM': [], 'd_TR': [], 'd_TP': [],
        'd_IM': [], 'd_MR': [], 'd_RP': [],
        'v_down_i': [], 'v_down_m': [], 'v_down_r': [],
        'ang_i_pip': [], 'ang_m_pip': [], 'ang_r_pip': [], 'ang_p_pip': [],
        'ratio_iw_mw': [], 'ratio_rw_mw': [], 'ratio_pw_mw': [],
        'ratio_mr_mw': [], 'ratio_im_mw': [], 'ratio_pw_rw': [],
        'diff_mw_rw': [], 'diff_rw_pw': []
    }
    for s in samples:
        p = s.get('landmarks')
        if not p or len(p) != 21: continue
        wrist = p[0]
        sh, ps = _shift_and_scale(p)
        if ps < 1e-5: continue

        d_tw = calculate_distance(sh[4], sh[0]) / ps
        d_iw = calculate_distance(sh[8], sh[0]) / ps
        d_mw = calculate_distance(sh[12], sh[0]) / ps
        d_rw = calculate_distance(sh[16], sh[0]) / ps
        d_pw = calculate_distance(sh[20], sh[0]) / ps

        d_ti = calculate_distance(sh[4], sh[8]) / ps
        d_tm = calculate_distance(sh[4], sh[12]) / ps
        d_tr = calculate_distance(sh[4], sh[16]) / ps
        d_tp = calculate_distance(sh[4], sh[20]) / ps
        d_im = calculate_distance(sh[8], sh[12]) / ps
        d_mr = calculate_distance(sh[12], sh[16]) / ps
        d_rp = calculate_distance(sh[16], sh[20]) / ps

        vx_i, vy_i = sh[8][0] - sh[5][0], sh[8][1] - sh[5][1]
        v_down_i = vy_i / (math.hypot(vx_i, vy_i) + 1e-8)

        vx_m, vy_m = sh[12][0] - sh[9][0], sh[12][1] - sh[9][1]
        v_down_m = vy_m / (math.hypot(vx_m, vy_m) + 1e-8)

        vx_r, vy_r = sh[16][0] - sh[13][0], sh[16][1] - sh[13][1]
        v_down_r = vy_r / (math.hypot(vx_r, vy_r) + 1e-8)

        ang_i_pip = calculate_angle(sh[5], sh[6], sh[7])
        ang_m_pip = calculate_angle(sh[9], sh[10], sh[11])
        ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
        ang_p_pip = calculate_angle(sh[17], sh[18], sh[19])

        ratio_iw_mw = d_iw / d_mw if d_mw > 1e-5 else 0.0
        ratio_rw_mw = d_rw / d_mw if d_mw > 1e-5 else 0.0
        ratio_pw_mw = d_pw / d_mw if d_mw > 1e-5 else 0.0
        ratio_mr_mw = d_mr / d_mw if d_mw > 1e-5 else 0.0
        ratio_im_mw = d_im / d_mw if d_mw > 1e-5 else 0.0
        ratio_pw_rw = d_pw / d_rw if d_rw > 1e-5 else 0.0

        diff_mw_rw = d_mw - d_rw
        diff_rw_pw = d_rw - d_pw

        metrics['wrist_y'].append(wrist[1])
        metrics['d_TW'].append(d_tw)
        metrics['d_IW'].append(d_iw)
        metrics['d_MW'].append(d_mw)
        metrics['d_RW'].append(d_rw)
        metrics['d_PW'].append(d_pw)
        metrics['d_TI'].append(d_ti)
        metrics['d_TM'].append(d_tm)
        metrics['d_TR'].append(d_tr)
        metrics['d_TP'].append(d_tp)
        metrics['d_IM'].append(d_im)
        metrics['d_MR'].append(d_mr)
        metrics['d_RP'].append(d_rp)
        metrics['v_down_i'].append(v_down_i)
        metrics['v_down_m'].append(v_down_m)
        metrics['v_down_r'].append(v_down_r)
        metrics['ang_i_pip'].append(ang_i_pip)
        metrics['ang_m_pip'].append(ang_m_pip)
        metrics['ang_r_pip'].append(ang_r_pip)
        metrics['ang_p_pip'].append(ang_p_pip)
        metrics['ratio_iw_mw'].append(ratio_iw_mw)
        metrics['ratio_rw_mw'].append(ratio_rw_mw)
        metrics['ratio_pw_mw'].append(ratio_pw_mw)
        metrics['ratio_mr_mw'].append(ratio_mr_mw)
        metrics['ratio_im_mw'].append(ratio_im_mw)
        metrics['ratio_pw_rw'].append(ratio_pw_rw)
        metrics['diff_mw_rw'].append(diff_mw_rw)
        metrics['diff_rw_pw'].append(diff_rw_pw)

    return metrics

n_stats = compute_all_metrics(n_samples)
m_stats = compute_all_metrics(m_samples)

print(f"{'Feature':<14} | {'N Mean':>7} | {'N Std':>6} | {'N [Min, Max]':>16} | {'M Mean':>7} | {'M Std':>6} | {'M [Min, Max]':>16} | {'Separation'}")
print("-" * 105)

for k in n_stats:
    n_arr = np.array(n_stats[k])
    m_arr = np.array(m_stats[k])
    n_mean, n_std = np.mean(n_arr), np.std(n_arr)
    n_min, n_max = np.min(n_arr), np.max(n_arr)
    m_mean, m_std = np.mean(m_arr), np.std(m_arr)
    m_min, m_max = np.min(m_arr), np.max(m_arr)

    overlap = max(0, min(n_max, m_max) - max(n_min, m_min))
    sep = 'STRONG' if overlap == 0 else ('MODERATE' if abs(n_mean - m_mean) > (n_std + m_std) else 'SHARED')
    
    print(f'{k:<14} | {n_mean:7.2f} | {n_std:6.2f} | [{n_min:6.2f}, {n_max:6.2f}] | {m_mean:7.2f} | {m_std:6.2f} | [{m_min:6.2f}, {m_max:6.2f}] | {sep}')

