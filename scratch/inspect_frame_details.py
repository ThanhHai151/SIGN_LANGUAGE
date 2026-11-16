import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle, compute_finger_states
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

for letter in ['T', 'U', 'S']:
    print(f"\n==================== LETTER {letter} ====================")
    with open(f'backend/data/alphabet_v2/static_landmarks/{letter}.json', 'r', encoding='utf-8') as f:
        js = json.load(f)
    samples = [s for s in js['samples'] if s.get('landmarks')]
    print(f"Total valid frames in {letter}.json: {len(samples)}")
    print(f"{'Frm':<4} | {'Phase':<11} | {'i_ext':<5} {'m_ext':<5} {'r_ext':<5} {'p_ext':<5} | {'d_TI':<6} {'d_TM':<6} {'d_IM':<6} {'d_IW':<6} {'d_MW':<6} {'d_RW':<6} {'d_PW':<6} | {'ang_IM':<6} {'ang_dir':<7} {'d_t4_i6':<7}")
    for s in samples:
        p = s['landmarks']
        sh, ps = _shift_and_scale(p)
        states = compute_finger_states(sh, ps)
        d_ti = calculate_distance(sh[4], sh[8]) / ps
        d_tm = calculate_distance(sh[4], sh[12]) / ps
        d_im = calculate_distance(sh[8], sh[12]) / ps
        d_iw = calculate_distance(sh[8], sh[0]) / ps
        d_mw = calculate_distance(sh[12], sh[0]) / ps
        d_rw = calculate_distance(sh[16], sh[0]) / ps
        d_pw = calculate_distance(sh[20], sh[0]) / ps
        ang_im = calculate_angle(sh[8], sh[0], sh[12])

        vx_i, vy_i = sh[8][0] - sh[5][0], sh[8][1] - sh[5][1]
        len_i = math.hypot(vx_i, vy_i) + 1e-8
        vx_m, vy_m = sh[12][0] - sh[9][0], sh[12][1] - sh[9][1]
        len_m = math.hypot(vx_m, vy_m) + 1e-8
        cos_im = max(-1.0, min(1.0, (vx_i * vx_m + vy_i * vy_m) / (len_i * len_m)))
        ang_dir = math.degrees(math.acos(cos_im))
        d_t4_i6 = calculate_distance(sh[4], sh[6]) / ps

        print(f"{s['frame_index']:<4} | {s['phase']:<11} | {str(states['index_extended']):<5} {str(states['middle_extended']):<5} {str(states['ring_extended']):<5} {str(states['pinky_extended']):<5} | {d_ti:<6.2f} {d_tm:<6.2f} {d_im:<6.2f} {d_iw:<6.2f} {d_mw:<6.2f} {d_rw:<6.2f} {d_pw:<6.2f} | {ang_im:<6.1f} {ang_dir:<7.1f} {d_t4_i6:<7.2f}")

