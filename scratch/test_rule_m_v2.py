import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

classifier = VSLAlphabetClassifier()

def dist_3d(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5

def evaluate_rule_m_v2(landmarks):
    if not landmarks or len(landmarks) < 21:
        return False, "Thiếu landmarks"
    
    p = landmarks
    wrist = p[0]
    
    # 0. Không nhận diện khi bàn tay đang hạ thấp dưới đáy màn hình (entry/exit)
    if wrist[1] > 0.68:
        return False, f"Cổ tay quá thấp: wrist_y={wrist[1]:.2f}>0.68"
    
    sh = [[p[i][0] - wrist[0], p[i][1] - wrist[1], p[i][2] - wrist[2]] for i in range(21)]
    palm_scale = dist_3d(sh[0], sh[9])
    if palm_scale < 1e-5:
        return False, "palm_scale quá nhỏ"
    
    # Khoảng cách chuẩn hóa tới cổ tay
    d_iw = dist_3d(p[8], wrist) / palm_scale
    d_mw = dist_3d(p[12], wrist) / palm_scale
    d_rw = dist_3d(p[16], wrist) / palm_scale
    d_pw = dist_3d(p[20], wrist) / palm_scale
    d_tw = dist_3d(p[4], wrist) / palm_scale
    
    # Chữ M có 3 ngón duỗi chúc xuống dài vượt trội so với palm_scale (thường >= 2.5)
    if d_iw < 2.0 or d_mw < 2.0 or d_rw < 1.9:
        return False, f"Các ngón chưa đủ độ vươn chúc xuống: d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, d_RW={d_rw:.2f}"
    
    # 1. Downward pointing check (úp xuống)
    dy_i = p[8][1] - p[5][1]
    dy_m = p[12][1] - p[9][1]
    dy_r = p[16][1] - p[13][1]
    
    len_i = dist_3d(p[8], p[5])
    len_m = dist_3d(p[12], p[9])
    len_r = dist_3d(p[16], p[13])
    
    if len_i < 1e-5 or len_m < 1e-5 or len_r < 1e-5:
        return False, "Độ dài ngón quá nhỏ"
    
    v_ratio_i = dy_i / len_i
    v_ratio_m = dy_m / len_m
    v_ratio_r = dy_r / len_r
    
    if v_ratio_i < 0.65 or v_ratio_m < 0.65 or v_ratio_r < 0.65:
        return False, f"Ngón không chúc xuống: v_i={v_ratio_i:.2f}, v_m={v_ratio_m:.2f}, v_r={v_ratio_r:.2f}"
    
    # 2. Tỷ lệ chiều dài tương đối giữa Trỏ, Giữa và Áp út (chữ M có 3 ngón cùng duỗi)
    ratio_rw_mw = d_rw / d_mw
    ratio_iw_mw = d_iw / d_mw
    
    if not (0.80 <= ratio_rw_mw <= 1.20):
        return False, f"Áp út không duỗi đều với giữa: ratio_rw_mw={ratio_rw_mw:.2f}"
    
    if not (0.80 <= ratio_iw_mw <= 1.25):
        return False, f"Trỏ không duỗi đều với giữa: ratio_iw_mw={ratio_iw_mw:.2f}"
    
    # 3. Phân biệt với chữ N: Trong chữ M, khoảng cách giữa đầu ngón Giữa (12) và Áp út (16) phải nhỏ
    d_mr = dist_3d(p[12], p[16]) / palm_scale
    d_im = dist_3d(p[8], p[12]) / palm_scale
    
    if d_mr > 0.90:  # Trong M là 0.10 - 0.55
        return False, f"Đầu ngón giữa và áp út tách quá xa (có thể là N): d_mr={d_mr:.2f}"
    
    if d_im > 0.95:  # Trong M là 0.15 - 0.46
        return False, f"Đầu ngón trỏ và giữa tách quá xa: d_im={d_im:.2f}"
    
    # 4. Ngón út phải gập ngắn hơn rõ rệt so với ngón áp út
    ratio_pw_rw = d_pw / d_rw
    if ratio_pw_rw > 0.88:
        return False, f"Ngón út không gập lại: ratio_pw_rw={ratio_pw_rw:.2f}"
    
    # 5. Phân biệt N vs M bằng vị trí ngón cái:
    # Trong M: ngón cái gập về phía ngón út (d_TP <= d_TR)
    # Trong N: ngón cái tì vào ngón áp út (d_TR < d_TP)
    d_tr = dist_3d(p[4], p[16])
    d_tp = dist_3d(p[4], p[20])
    # Nếu d_TR < 0.85 * d_TP thì ngón cái đang kẹp ngón áp út -> Chữ N!
    if d_tr < 0.80 * d_tp:
        return False, f"Ngón cái kẹp áp út (chữ N): d_TR={d_tr:.3f} < 0.8*d_TP={0.8*d_tp:.3f}"
    
    return True, f"Khớp chữ M: ratio_rw_mw={ratio_rw_mw:.2f}, d_MR={d_mr:.2f}, ratio_pw_rw={ratio_pw_rw:.2f}"

# Test on M.json
with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

m_pass = 0
m_stable_pass = 0
m_stable_total = 0
for s in m_data['samples']:
    is_stable = (s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok')
    if is_stable: m_stable_total += 1
    ok, reason = evaluate_rule_m_v2(s.get('landmarks'))
    if ok:
        m_pass += 1
        if is_stable: m_stable_pass += 1
    else:
        if is_stable:
            print(f"M stable frame {s['frame_index']} rejected: {reason}")

print(f"\nResult on M.json: {m_pass}/{len(m_data['samples'])} total, {m_stable_pass}/{m_stable_total} stable")

# Test on N.json
with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

n_pass = 0
for s in n_data['samples']:
    ok, reason = evaluate_rule_m_v2(s.get('landmarks'))
    if ok:
        n_pass += 1
        print(f"N frame {s['frame_index']} falsely matched as M: {reason}")

print(f"Result on N.json: {n_pass}/{len(n_data['samples'])} falsely matched as M")

# Test on ALL other 21 letters in static dataset v2
npz_data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
all_labels = [npz_data['classes'][i] for i in npz_data['y']]
all_lms = npz_data['landmarks']

other_fps = {}
for lbl, lm in zip(all_labels, all_lms):
    if lbl == 'M':
        continue
    ok, reason = evaluate_rule_m_v2(lm.tolist())
    if ok:
        other_fps[lbl] = other_fps.get(lbl, 0) + 1

print(f"False Positives on all other 21 letters in dataset v2: {other_fps} (Total: {sum(other_fps.values())})")

