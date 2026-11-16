import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from backend.vision.feature_extractor import calculate_angle, calculate_distance, _shift_and_scale

def evaluate_rule_m_clean(landmarks):
    if not landmarks or len(landmarks) < 21:
        return False, "Thiếu landmarks"
    
    p = landmarks
    wrist = p[0]
    
    # 1. Chống nhiễu tay hạ dưới đùi (entry/exit): bàn tay phải nâng lên trước ngực/camera
    if wrist[1] > 0.68:
        return False, f"Cổ tay hạ quá thấp: wrist_y={wrist[1]:.2f}>0.68"
    
    sh, palm_scale = _shift_and_scale(landmarks)
    if palm_scale < 1e-5:
        return False, "palm_scale quá nhỏ"
    
    # Khoảng cách chuẩn hóa tới cổ tay
    d_iw = calculate_distance(sh[8], sh[0]) / palm_scale
    d_mw = calculate_distance(sh[12], sh[0]) / palm_scale
    d_rw = calculate_distance(sh[16], sh[0]) / palm_scale
    d_pw = calculate_distance(sh[20], sh[0]) / palm_scale
    d_tw = calculate_distance(sh[4], sh[0]) / palm_scale
    
    # 2. Hướng chúc xuống (úp xuống):
    # Vector từ MCP đến TIP của Trỏ (5->8), Giữa (9->12), Áp út (13->16) phải hướng xuống dưới (dy > 0)
    dy_i = p[8][1] - p[5][1]
    dy_m = p[12][1] - p[9][1]
    dy_r = p[16][1] - p[13][1]
    
    len_i = calculate_distance(p[8], p[5])
    len_m = calculate_distance(p[12], p[9])
    len_r = calculate_distance(p[16], p[13])
    
    if len_i < 1e-5 or len_m < 1e-5 or len_r < 1e-5:
        return False, "Độ dài ngón quá nhỏ"
    
    # Tỷ lệ hướng thẳng đứng chúc xuống
    v_ratio_i = dy_i / len_i
    v_ratio_m = dy_m / len_m
    v_ratio_r = dy_r / len_r
    
    # Phải chúc xuống (ít nhất 0.60, tức góc nghiêng so với phương thẳng đứng <= 53 độ)
    if v_ratio_i < 0.60 or v_ratio_m < 0.60 or v_ratio_r < 0.60:
        return False, f"Ngón không chúc xuống: v_i={v_ratio_i:.2f}, v_m={v_ratio_m:.2f}, v_r={v_ratio_r:.2f}"
    
    # 3. Ba ngón Trỏ, Giữa, Áp út phải vươn chúc xuống dài so với palm_scale
    if d_iw < 2.0 or d_mw < 2.0 or d_rw < 1.9:
        return False, f"Độ vươn chưa đạt: d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, d_RW={d_rw:.2f}"
    
    # 4. Trỏ, Giữa, Áp út phải có chiều dài tương đương nhau (chữ M có 3 vòm đều nhau)
    ratio_rw_mw = d_rw / d_mw
    ratio_iw_mw = d_iw / d_mw
    
    if not (0.80 <= ratio_rw_mw <= 1.20):
        return False, f"Áp út không đều giữa: ratio_rw_mw={ratio_rw_mw:.2f}"
    
    if not (0.80 <= ratio_iw_mw <= 1.25):
        return False, f"Trỏ không đều giữa: ratio_iw_mw={ratio_iw_mw:.2f}"
    
    # 5. Phân biệt với chữ N:
    # - Trong M: Đầu ngón Giữa (12) và Áp út (16) chụm kề sát nhau (d_MR <= 0.70)
    # - Trong N: Áp út gập vào lòng bàn tay nên đầu ngón Giữa và Áp út tách xa (d_MR >= 1.30)
    d_mr = calculate_distance(sh[12], sh[16]) / palm_scale
    d_im = calculate_distance(sh[8], sh[12]) / palm_scale
    
    if d_mr > 0.70:
        return False, f"Đầu ngón Giữa và Áp út tách xa (dạng N): d_MR={d_mr:.2f}>0.70"
    
    if d_im > 0.75:
        return False, f"Đầu ngón Trỏ và Giữa tách xa: d_im={d_im:.2f}>0.75"
    
    # 6. Ngón út gập vào trong lòng bàn tay:
    # Khoảng cách từ ngón út đến cổ tay phải ngắn hơn đáng kể so với ngón áp út
    ratio_pw_rw = d_pw / d_rw
    if ratio_pw_rw > 0.88:
        return False, f"Ngón út không gập: ratio_pw_rw={ratio_pw_rw:.2f}>0.88"
    
    # 7. Độ cong đốt ngón áp út: phải thẳng (gập PIP <= 25 độ, tức góc duỗi >= 155 độ)
    ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
    if ang_r_pip < 155.0:
        return False, f"Đốt ngón áp út bị cong (dạng N): ang_R_PIP={ang_r_pip:.1f}<155"
    
    return True, f"Khớp chữ M: ratio_rw_mw={ratio_rw_mw:.2f}, d_MR={d_mr:.2f}, ang_R_PIP={ang_r_pip:.1f}"

# Test on M.json
with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

m_pass = 0
m_stable_pass = 0
m_stable_total = 0
for s in m_data['samples']:
    is_stable = (s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok')
    if is_stable: m_stable_total += 1
    ok, reason = evaluate_rule_m_clean(s.get('landmarks'))
    if ok:
        m_pass += 1
        if is_stable: m_stable_pass += 1
    else:
        if is_stable:
            print(f"M stable f{s['frame_index']} rejected: {reason}")

print(f"\nResult on M.json: {m_pass}/{len(m_data['samples'])} total, {m_stable_pass}/{m_stable_total} stable (Rate: {m_stable_pass/m_stable_total*100:.1f}%)")

# Test on N.json
with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

n_pass = 0
for s in n_data['samples']:
    ok, reason = evaluate_rule_m_clean(s.get('landmarks'))
    if ok:
        n_pass += 1
        print(f"N f{s['frame_index']} FALSE POSITIVE: {reason}")

print(f"Result on N.json: {n_pass}/{len(n_data['samples'])} FALSE POSITIVES (Expected: 0)")

# Test on ALL 551 samples of static dataset v2
npz_data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
all_labels = [npz_data['classes'][i] for i in npz_data['y']]
all_lms = npz_data['landmarks']

fps = {}
for lbl, lm in zip(all_labels, all_lms):
    if lbl == 'M':
        continue
    ok, reason = evaluate_rule_m_clean(lm.tolist())
    if ok:
        fps[lbl] = fps.get(lbl, 0) + 1

print(f"False Positives on all other 21 letters in dataset v2: {fps} (Total: {sum(fps.values())})")

