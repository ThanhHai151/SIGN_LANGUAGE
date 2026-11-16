import numpy as np
import json
import math
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
all_labels = [npz['classes'][i] for i in npz['y']]
all_lms = npz['landmarks']

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    fix_data = json.load(f)

def test_rule_n(p, diff_mw_rw_min=0.60, ratio_rw_mw_max=0.78, d_mr_min=0.65, ratio_mr_mw_min=0.20):
    if not p or len(p) != 21: return False, 'bad len'
    wrist = p[0]
    if wrist[1] > 0.68: return False, 'wrist too low'
    sh, ps = _shift_and_scale(p)
    if ps < 1e-5: return False, 'ps too small'

    # Downward check for Index and Middle
    vx_i, vy_i = sh[8][0] - sh[5][0], sh[8][1] - sh[5][1]
    v_down_i = vy_i / (math.hypot(vx_i, vy_i) + 1e-8)
    vx_m, vy_m = sh[12][0] - sh[9][0], sh[12][1] - sh[9][1]
    v_down_m = vy_m / (math.hypot(vx_m, vy_m) + 1e-8)

    if v_down_i < 0.60 or v_down_m < 0.60:
        return False, 'not pointing down'

    d_iw = calculate_distance(sh[8], sh[0]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_rw = calculate_distance(sh[16], sh[0]) / ps
    d_pw = calculate_distance(sh[20], sh[0]) / ps
    d_im = calculate_distance(sh[8], sh[12]) / ps
    d_mr = calculate_distance(sh[12], sh[16]) / ps

    if d_iw < 2.00 or d_mw < 2.00:
        return False, 'index/mid too short'

    ratio_iw_mw = d_iw / d_mw if d_mw > 0 else 0
    if not (0.75 <= ratio_iw_mw <= 1.25):
        return False, 'ratio_iw_mw bad'

    if d_im > 0.65:
        return False, 'index and mid too far'

    ratio_rw_mw = d_rw / d_mw if d_mw > 0 else 0
    ratio_mr_mw = d_mr / d_mw if d_mw > 0 else 0
    diff_mw_rw = d_mw - d_rw
    diff_mw_pw = d_mw - d_pw

    # Pinky must be curled
    if diff_mw_pw < 1.00 or d_pw > 2.60:
        return False, 'pinky not curled'

    # Ring finger must be curled (NOT extended like M)
    # At least ONE of the strong ring-curled discriminants must hold, OR a combination:
    ring_curled = (
        (diff_mw_rw >= diff_mw_rw_min) and
        (ratio_rw_mw <= ratio_rw_mw_max or d_mr >= d_mr_min or ratio_mr_mw >= ratio_mr_mw_min)
    )
    if not ring_curled:
        return False, 'ring not curled (like M)'

    return True, 'match N'

for d_min in [0.55, 0.60, 0.65, 0.70]:
    for r_max in [0.75, 0.78, 0.80]:
        for mr_min in [0.60, 0.65, 0.70]:
            n_pass = sum(1 for s in n_data['samples'] if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok' and test_rule_n(s['landmarks'], d_min, r_max, mr_min)[0])
            m_fp = sum(1 for s in m_data['samples'] if test_rule_n(s.get('landmarks'), d_min, r_max, mr_min)[0])
            other_fp = sum(1 for lbl, lm in zip(all_labels, all_lms) if lbl not in ('N') and test_rule_n(lm.tolist(), d_min, r_max, mr_min)[0])
            fix_n_pass = sum(1 for s in fix_data['samples'] if s.get('letter_upper') == 'N' and s.get('phase') == 'stable_pose' and test_rule_n(s.get('landmarks'), d_min, r_max, mr_min)[0])
            fix_m_fp = sum(1 for s in fix_data['samples'] if s.get('letter_upper') == 'M' and test_rule_n(s.get('landmarks'), d_min, r_max, mr_min)[0])
            fix_other_fp = sum(1 for s in fix_data['samples'] if s.get('letter_upper') not in ('M', 'N') and test_rule_n(s.get('landmarks'), d_min, r_max, mr_min)[0])

            if m_fp == 0 and other_fp == 0 and fix_m_fp == 0 and fix_other_fp == 0 and n_pass >= 10:
                print(f"diff_mw_rw_min={d_min}, ratio_rw_mw_max={r_max}, d_mr_min={mr_min} -> N_stable={n_pass}/21, M_FP={m_fp}, Other_FP={other_fp}, Fix_N={fix_n_pass}/3, Fix_M_FP={fix_m_fp}")

