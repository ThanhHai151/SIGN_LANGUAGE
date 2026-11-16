import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from backend.vision.feature_extractor import calculate_angle, calculate_distance, _shift_and_scale, compute_finger_states

# Define Constants for Letter M
M_D_IW_MIN = 2.00
M_D_MW_MIN = 2.00
M_D_RW_MIN = 1.90
M_RATIO_RW_MW_MIN = 0.80
M_RATIO_RW_MW_MAX = 1.20
M_RATIO_IW_MW_MIN = 0.80
M_RATIO_IW_MW_MAX = 1.25
M_D_MR_MAX = 0.65
M_D_IM_MAX = 0.65
M_RATIO_PW_RW_MAX = 0.88
M_DIFF_RW_PW_MIN = 0.50
M_DOWN_RATIO_MIN = 0.60
M_WRIST_Y_MAX = 0.68
M_ANG_R_PIP_MIN = 150.0

def evaluate_m_rule(landmarks):
    if not landmarks or len(landmarks) < 21:
        return False, "Thiếu landmarks"
    
    p = landmarks
    wrist = p[0]
    
    # 1. Bàn tay phải ở vị trí ký hiệu (không hạ thấp dưới đùi)
    if wrist[1] > M_WRIST_Y_MAX:
        return False, f"Cổ tay quá thấp: wrist_y={wrist[1]:.2f}>{M_WRIST_Y_MAX}"
    
    sh, palm_scale = _shift_and_scale(landmarks)
    if palm_scale < 1e-5:
        return False, "palm_scale quá nhỏ"
    
    d_iw = calculate_distance(sh[8], sh[0]) / palm_scale
    d_mw = calculate_distance(sh[12], sh[0]) / palm_scale
    d_rw = calculate_distance(sh[16], sh[0]) / palm_scale
    d_pw = calculate_distance(sh[20], sh[0]) / palm_scale
    d_tw = calculate_distance(sh[4], sh[0]) / palm_scale
    
    # 2. Hướng chúc xuống: vector MCP -> TIP của Trỏ, Giữa, Áp út
    dy_i = p[8][1] - p[5][1]
    dy_m = p[12][1] - p[9][1]
    dy_r = p[16][1] - p[13][1]
    
    len_i = calculate_distance(p[8], p[5])
    len_m = calculate_distance(p[12], p[9])
    len_r = calculate_distance(p[16], p[13])
    
    if len_i < 1e-5 or len_m < 1e-5 or len_r < 1e-5:
        return False, "Độ dài ngón quá nhỏ"
    
    v_ratio_i = dy_i / len_i
    v_ratio_m = dy_m / len_m
    v_ratio_r = dy_r / len_r
    
    if v_ratio_i < M_DOWN_RATIO_MIN or v_ratio_m < M_DOWN_RATIO_MIN or v_ratio_r < M_DOWN_RATIO_MIN:
        return False, f"Ngón không chúc xuống: v_i={v_ratio_i:.2f}, v_m={v_ratio_m:.2f}, v_r={v_ratio_r:.2f}"
    
    # 3. Ba ngón Trỏ, Giữa, Áp út phải vươn dài chúc xuống
    if d_iw < M_D_IW_MIN or d_mw < M_D_MW_MIN or d_rw < M_D_RW_MIN:
        return False, f"Độ dài chưa đạt: d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, d_RW={d_rw:.2f}"
    
    # 4. Trỏ, Giữa, Áp út đều nhau
    ratio_rw_mw = d_rw / d_mw
    ratio_iw_mw = d_iw / d_mw
    
    if not (M_RATIO_RW_MW_MIN <= ratio_rw_mw <= M_RATIO_RW_MW_MAX):
        return False, f"Áp út không đều giữa: ratio_rw_mw={ratio_rw_mw:.2f}"
    
    if not (M_RATIO_IW_MW_MIN <= ratio_iw_mw <= M_RATIO_IW_MW_MAX):
        return False, f"Trỏ không đều giữa: ratio_iw_mw={ratio_iw_mw:.2f}"
    
    # 5. Phân biệt với chữ N: Khoảng cách giữa đầu ngón Giữa và Áp út phải nhỏ (chụm)
    d_mr = calculate_distance(sh[12], sh[16]) / palm_scale
    d_im = calculate_distance(sh[8], sh[12]) / palm_scale
    
    if d_mr > M_D_MR_MAX:
        return False, f"Đầu ngón Giữa và Áp út tách xa (chữ N): d_MR={d_mr:.2f}>{M_D_MR_MAX}"
    
    if d_im > M_D_IM_MAX:
        return False, f"Đầu ngón Trỏ và Giữa tách xa: d_IM={d_im:.2f}>{M_D_IM_MAX}"
    
    # 6. Ngón út gập vào trong lòng bàn tay
    ratio_pw_rw = d_pw / d_rw
    diff_rw_pw = d_rw - d_pw
    if ratio_pw_rw > M_RATIO_PW_RW_MAX or diff_rw_pw < M_DIFF_RW_PW_MIN:
        return False, f"Ngón út không gập: ratio_pw_rw={ratio_pw_rw:.2f}, diff_rw_pw={diff_rw_pw:.2f}"
    
    # 7. Góc đốt ngón áp út
    ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
    if ang_r_pip < M_ANG_R_PIP_MIN:
        return False, f"Đốt ngón áp út bị gập (chữ N): ang_R_PIP={ang_r_pip:.1f}<{M_ANG_R_PIP_MIN}"
    
    return True, f"Khớp chữ M: ratio_rw_mw={ratio_rw_mw:.2f}, d_MR={d_mr:.2f}, ang_R_PIP={ang_r_pip:.1f}"

# 1. Test on M.json
with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

m_pass = 0
m_stable_pass = 0
m_stable_tot = 0
for s in m_data['samples']:
    is_stable = (s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok')
    if is_stable: m_stable_tot += 1
    ok, _ = evaluate_m_rule(s.get('landmarks'))
    if ok:
        m_pass += 1
        if is_stable: m_stable_pass += 1

print(f"=== M.json ===\nTotal: {m_pass}/{len(m_data['samples'])}, Stable: {m_stable_pass}/{m_stable_tot} ({m_stable_pass/m_stable_tot*100:.1f}%)")

# 2. Test on N.json
with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

n_fps = 0
for s in n_data['samples']:
    ok, reason = evaluate_m_rule(s.get('landmarks'))
    if ok:
        n_fps += 1
        print(f"  N f{s['frame_index']} FP: {reason}")

print(f"=== N.json ===\nFalse Positives: {n_fps}/{len(n_data['samples'])} (Target: 0)")

# 3. Test on ALL 551 samples in static dataset v2
npz_data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
all_labels = [npz_data['classes'][i] for i in npz_data['y']]
all_lms = npz_data['landmarks']

fps = {}
for lbl, lm in zip(all_labels, all_lms):
    if lbl == 'M':
        continue
    ok, reason = evaluate_m_rule(lm.tolist())
    if ok:
        fps[lbl] = fps.get(lbl, 0) + 1

print(f"=== Dataset v2 (551 samples, 21 other letters) ===\nFalse Positives: {fps} (Total: {sum(fps.values())})")

# 4. Test on real_alphabet_landmarks.json
with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    fixture_data = json.load(f)

m_fix_pass = 0
m_fix_tot = 0
n_fix_fps = 0
other_fix_fps = {}

for s in fixture_data['samples']:
    lbl = s.get('letter_upper')
    lm = s.get('landmarks')
    phase = s.get('phase')
    ok, reason = evaluate_m_rule(lm)
    if lbl == 'M':
        m_fix_tot += 1
        if ok: m_fix_pass += 1
        print(f"Fixture M ({phase}): match={ok} ({reason})")
    elif lbl == 'N':
        if ok:
            n_fix_fps += 1
            print(f"Fixture N ({phase}) FP: {reason}")
    else:
        if ok:
            other_fix_fps[lbl] = other_fix_fps.get(lbl, 0) + 1

print(f"=== Fixture real_alphabet_landmarks ===\nM matches: {m_fix_pass}/{m_fix_tot}, N FPs: {n_fix_fps}, Other FPs: {other_fix_fps}")

