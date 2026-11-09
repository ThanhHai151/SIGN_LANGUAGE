import sys, os
sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle
import json
import numpy as np

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

npz_data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
all_labels = [npz_data['classes'][i] for i in npz_data['y']]
all_lms = npz_data['landmarks']

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    fixture_data = json.load(f)

def eval_m_tuned_v2(landmarks):
    if not landmarks or len(landmarks) < 21: return False, 'no lms'
    p = landmarks
    wrist = p[0]
    if wrist[1] > 0.75: return False, 'wrist too low'
    sh, ps = _shift_and_scale(landmarks)
    if ps < 1e-5: return False, 'ps too small'
    
    d_iw = calculate_distance(sh[8], sh[0]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_rw = calculate_distance(sh[16], sh[0]) / ps
    d_pw = calculate_distance(sh[20], sh[0]) / ps
    d_tw = calculate_distance(sh[4], sh[0]) / ps
    
    dy_i = p[8][1] - p[5][1]
    dy_m = p[12][1] - p[9][1]
    dy_r = p[16][1] - p[13][1]
    
    len_i = calculate_distance(p[8], p[5])
    len_m = calculate_distance(p[12], p[9])
    len_r = calculate_distance(p[16], p[13])
    if len_i < 1e-5 or len_m < 1e-5 or len_r < 1e-5: return False, 'len too small'
    
    v_i = dy_i / len_i
    v_m = dy_m / len_m
    v_r = dy_r / len_r
    if v_i < 0.60 or v_m < 0.60 or v_r < 0.60: return False, 'not down'
    
    if d_iw < 2.0 or d_mw < 2.0 or d_rw < 1.8: return False, 'too short'
    
    ratio_rw_mw = d_rw / d_mw
    ratio_iw_mw = d_iw / d_mw
    if not (0.70 <= ratio_rw_mw <= 1.25): return False, 'rw_mw bad'
    if not (0.75 <= ratio_iw_mw <= 1.25): return False, 'iw_mw bad'
    
    raw_d_mw = calculate_distance(p[12], wrist)
    raw_d_mr = calculate_distance(p[12], p[16])
    raw_d_im = calculate_distance(p[8], p[12])
    
    ratio_mr = raw_d_mr / raw_d_mw if raw_d_mw > 0 else 99.0
    ratio_im = raw_d_im / raw_d_mw if raw_d_mw > 0 else 99.0
    
    if ratio_mr > 0.35: return False, f'ratio_mr too big ({ratio_mr:.2f})'
    if ratio_im > 0.35: return False, f'ratio_im too big ({ratio_im:.2f})'
    
    ratio_pw_rw = d_pw / d_rw
    diff_rw_pw = d_rw - d_pw
    if ratio_pw_rw > 0.90 or diff_rw_pw < 0.30: return False, 'pinky not curled'
    
    ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
    if ang_r_pip < 150.0: return False, 'r_pip too curled'
    
    return True, 'match M'

m_pass = sum(1 for s in m_data['samples'] if eval_m_tuned_v2(s.get('landmarks'))[0])
m_stable_pass = sum(1 for s in m_data['samples'] if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok' and eval_m_tuned_v2(s.get('landmarks'))[0])
m_stable_tot = sum(1 for s in m_data['samples'] if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok')

n_pass = sum(1 for s in n_data['samples'] if eval_m_tuned_v2(s.get('landmarks'))[0])
print(f"M pass: {m_pass}/{len(m_data['samples'])}, Stable: {m_stable_pass}/{m_stable_tot} ({m_stable_pass/m_stable_tot*100:.1f}%)")
print(f"N FP: {n_pass}/{len(n_data['samples'])}")

# check all other 21 letters in dataset v2
fps = {}
for lbl, lm in zip(all_labels, all_lms):
    if lbl == 'M': continue
    ok, _ = eval_m_tuned_v2(lm.tolist())
    if ok: fps[lbl] = fps.get(lbl, 0) + 1
print("Dataset v2 other FPs:", fps)

# check fixture
fix_m = sum(1 for s in fixture_data['samples'] if s.get('letter_upper') == 'M' and eval_m_tuned_v2(s.get('landmarks'))[0])
fix_n = sum(1 for s in fixture_data['samples'] if s.get('letter_upper') == 'N' and eval_m_tuned_v2(s.get('landmarks'))[0])
fix_other = sum(1 for s in fixture_data['samples'] if s.get('letter_upper') not in ('M', 'N') and eval_m_tuned_v2(s.get('landmarks'))[0])
print(f"Fixture M: {fix_m}/7, Fixture N FP: {fix_n}/7, Fixture other FP: {fix_other}")
