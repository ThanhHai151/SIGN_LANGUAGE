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

def evaluate_params(wrist_y_max, d_mr_norm_max, d_mr_raw_ratio_max, ratio_rw_mw_min, diff_rw_pw_min, ang_r_pip_min):
    def check_one(landmarks):
        if not landmarks or len(landmarks) < 21: return False
        p = landmarks
        wrist = p[0]
        if wrist[1] > wrist_y_max: return False
        sh, ps = _shift_and_scale(landmarks)
        if ps < 1e-5: return False
        
        d_iw = calculate_distance(sh[8], sh[0]) / ps
        d_mw = calculate_distance(sh[12], sh[0]) / ps
        d_rw = calculate_distance(sh[16], sh[0]) / ps
        d_pw = calculate_distance(sh[20], sh[0]) / ps
        
        dy_i = p[8][1] - p[5][1]
        dy_m = p[12][1] - p[9][1]
        dy_r = p[16][1] - p[13][1]
        len_i = calculate_distance(p[8], p[5])
        len_m = calculate_distance(p[12], p[9])
        len_r = calculate_distance(p[16], p[13])
        if len_i < 1e-5 or len_m < 1e-5 or len_r < 1e-5: return False
        if (dy_i / len_i < 0.60) or (dy_m / len_m < 0.60) or (dy_r / len_r < 0.60): return False
        
        if d_iw < 2.0 or d_mw < 2.0 or d_rw < 1.8: return False
        
        ratio_rw_mw = d_rw / d_mw
        ratio_iw_mw = d_iw / d_mw
        if not (ratio_rw_mw_min <= ratio_rw_mw <= 1.25): return False
        if not (0.75 <= ratio_iw_mw <= 1.25): return False
        
        # d_mr checks (both normalized by palm_scale and ratio to d_mw)
        d_mr = calculate_distance(sh[12], sh[16]) / ps
        raw_d_mw = calculate_distance(p[12], wrist)
        raw_d_mr = calculate_distance(p[12], p[16])
        ratio_mr = raw_d_mr / raw_d_mw if raw_d_mw > 0 else 99.0
        
        if d_mr > d_mr_norm_max: return False
        if ratio_mr > d_mr_raw_ratio_max: return False
        
        d_im = calculate_distance(sh[8], sh[12]) / ps
        if d_im > 0.65: return False
        
        ratio_pw_rw = d_pw / d_rw
        diff_rw_pw = d_rw - d_pw
        if ratio_pw_rw > 0.90 or diff_rw_pw < diff_rw_pw_min: return False
        
        ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
        if ang_r_pip < ang_r_pip_min: return False
        
        return True

    m_stable_tot = sum(1 for s in m_data['samples'] if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok')
    m_stable_pass = sum(1 for s in m_data['samples'] if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok' and check_one(s.get('landmarks')))
    n_fps = sum(1 for s in n_data['samples'] if check_one(s.get('landmarks')))
    
    other_fps = sum(1 for lbl, lm in zip(all_labels, all_lms) if lbl != 'M' and check_one(lm.tolist()))
    
    fix_m_pass = sum(1 for s in fixture_data['samples'] if s.get('letter_upper') == 'M' and s.get('phase') == 'stable_pose' and check_one(s.get('landmarks')))
    fix_n_fps = sum(1 for s in fixture_data['samples'] if s.get('letter_upper') == 'N' and check_one(s.get('landmarks')))
    
    return {
        'm_stable': f"{m_stable_pass}/{m_stable_tot}",
        'm_rate': m_stable_pass / m_stable_tot,
        'n_fps': n_fps,
        'other_fps': other_fps,
        'fix_m': fix_m_pass,
        'fix_n_fps': fix_n_fps
    }

print("Running parameter grid search...")
for w_y in [0.68, 0.70]:
    for mr_norm in [0.55, 0.60, 0.65]:
        for mr_raw in [0.20, 0.25]:
            for rw_mw_min in [0.75, 0.80]:
                for diff_pw in [0.35, 0.45]:
                    res = evaluate_params(w_y, mr_norm, mr_raw, rw_mw_min, diff_pw, 150.0)
                    if res['other_fps'] <= 1 and res['n_fps'] <= 1 and res['fix_n_fps'] == 0:
                        print(f"w_y={w_y}, mr_norm={mr_norm}, mr_raw={mr_raw}, rw_mw={rw_mw_min}, diff_pw={diff_pw} -> M={res['m_stable']} ({res['m_rate']*100:.1f}%), N_fps={res['n_fps']}, Other_fps={res['other_fps']}, Fix_M={res['fix_m']}/3, Fix_N_fps={res['fix_n_fps']}")

