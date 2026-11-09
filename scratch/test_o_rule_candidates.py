import numpy as np
import json
import math
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance
from backend.vision.alphabet_classifier import alphabet_classifier

# Load datasets
with open('backend/data/alphabet_v2/static_landmarks/O.json', 'r', encoding='utf-8') as f:
    o_json = json.load(f)

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    fix_data = json.load(f)

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes'].tolist()
y = npz['y']
lms = npz['landmarks']

def test_rule_o(lm, d_ti_max=0.30, d_tm_max=0.28, d_im_max=0.30, d_iw_min=0.90, d_iw_max=1.45, d_mw_min=0.80, d_mw_max=1.45, d_pw_min=0.60):
    if not lm or len(lm) != 21:
        return False, 'bad len'
    sh, ps = _shift_and_scale(lm)
    if ps < 1e-5:
        return False, 'ps'

    d_ti = calculate_distance(sh[4], sh[8]) / ps
    d_tm = calculate_distance(sh[4], sh[12]) / ps
    d_im = calculate_distance(sh[8], sh[12]) / ps
    d_iw = calculate_distance(sh[8], sh[0]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_rw = calculate_distance(sh[16], sh[0]) / ps
    d_pw = calculate_distance(sh[20], sh[0]) / ps

    # Finger extension checks: none of index, middle, ring, pinky should be extended upright
    # In O, fingertips are curved towards thumb
    i_ext = (sh[8][1] < sh[6][1]) and (d_iw > 1.5)
    m_ext = (sh[12][1] < sh[10][1]) and (d_mw > 1.5)

    if d_ti > d_ti_max:
        return False, f'd_ti={d_ti:.3f}>{d_ti_max}'
    if d_tm > d_tm_max:
        return False, f'd_tm={d_tm:.3f}>{d_tm_max}'
    if d_im > d_im_max:
        return False, f'd_im={d_im:.3f}>{d_im_max}'
    if not (d_iw_min <= d_iw <= d_iw_max):
        return False, f'd_iw={d_iw:.3f} not in [{d_iw_min}, {d_iw_max}]'
    if not (d_mw_min <= d_mw <= d_mw_max):
        return False, f'd_mw={d_mw:.3f} not in [{d_mw_min}, {d_mw_max}]'
    if d_pw < d_pw_min:
        return False, f'd_pw={d_pw:.3f}<{d_pw_min} (fist like A)'

    return True, 'MATCH O'

# Grid search / evaluation
for d_ti_m in [0.26, 0.28, 0.30, 0.32]:
    for d_tm_m in [0.24, 0.26, 0.28, 0.30]:
        for d_pw_m in [0.55, 0.60, 0.65]:
            # Evaluate on O.json stable
            o_stable = [s for s in o_json['samples'] if s.get('phase') == 'stable_pose' and s.get('landmarks')]
            o_pass = sum(1 for s in o_stable if test_rule_o(s['landmarks'], d_ti_max=d_ti_m, d_tm_max=d_tm_m, d_pw_min=d_pw_m)[0])

            # Evaluate on NPZ O
            o_idx = classes.index('O')
            npz_o_pass = sum(1 for i in range(len(y)) if y[i] == o_idx and test_rule_o(lms[i].tolist(), d_ti_max=d_ti_m, d_tm_max=d_tm_m, d_pw_min=d_pw_m)[0])

            # False positives on NPZ non-O
            npz_fp = [(classes[y[i]], i) for i in range(len(y)) if y[i] != o_idx and test_rule_o(lms[i].tolist(), d_ti_max=d_ti_m, d_tm_max=d_tm_m, d_pw_min=d_pw_m)[0]]

            # False positives on fixture non-O
            fix_fp = [(s.get('letter_upper'), s.get('frame_idx')) for s in fix_data['samples'] if s.get('letter_upper') != 'O' and test_rule_o(s.get('landmarks'), d_ti_max=d_ti_m, d_tm_max=d_tm_m, d_pw_min=d_pw_m)[0]]

            if len(npz_fp) == 0 and len(fix_fp) == 0:
                print(f"d_ti_max={d_ti_m:.2f}, d_tm_max={d_tm_m:.2f}, d_pw_min={d_pw_m:.2f} -> O.json stable={o_pass}/21, NPZ O={npz_o_pass}/48, NPZ FP={len(npz_fp)}, Fix FP={len(fix_fp)}")

