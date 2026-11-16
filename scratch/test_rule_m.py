import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

classifier = VSLAlphabetClassifier()

def dist_3d(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5

def evaluate_rule_m(landmarks):
    if not landmarks or len(landmarks) < 21:
        return False, "Thiếu landmarks"
    
    p = landmarks
    wrist = p[0]
    
    # 1. Downward pointing check (úp xuống)
    # MediaPipe Y increases downwards.
    # Check Index (5->8), Middle (9->12), Ring (13->16)
    dy_i = p[8][1] - p[5][1]
    dy_m = p[12][1] - p[9][1]
    dy_r = p[16][1] - p[13][1]
    
    len_i = dist_3d(p[8], p[5])
    len_m = dist_3d(p[12], p[9])
    len_r = dist_3d(p[16], p[13])
    
    if len_i < 1e-5 or len_m < 1e-5 or len_r < 1e-5:
        return False, "Độ dài ngón quá nhỏ"
    
    # Vertical downward ratio (1.0 = thẳng đứng chúc xuống)
    v_ratio_i = dy_i / len_i
    v_ratio_m = dy_m / len_m
    v_ratio_r = dy_r / len_r
    
    # Phải chúc xuống (ít nhất 0.60, tức góc nghiêng so với phương thẳng đứng <= 53 độ)
    if v_ratio_i < 0.60 or v_ratio_m < 0.60 or v_ratio_r < 0.60:
        return False, f"Ngón không chúc xuống: v_i={v_ratio_i:.2f}, v_m={v_ratio_m:.2f}, v_r={v_ratio_r:.2f}"
    
    # 2. Cả 3 ngón phải duỗi thẳng hơn cổ tay so với ngón út
    d_iw = dist_3d(p[8], wrist)
    d_mw = dist_3d(p[12], wrist)
    d_rw = dist_3d(p[16], wrist)
    d_pw = dist_3d(p[20], wrist)
    
    if d_mw < 1e-5 or d_rw < 1e-5:
        return False, "Khoảng cách ngón đến cổ tay quá nhỏ"
    
    # 3. Tỷ lệ chiều dài tương đối giữa Trỏ, Giữa và Áp út (chữ M có 3 ngón cùng duỗi)
    ratio_rw_mw = d_rw / d_mw
    ratio_iw_mw = d_iw / d_mw
    
    # Áp út phải gần tương đương Giữa (0.80 <= ratio <= 1.20)
    if not (0.80 <= ratio_rw_mw <= 1.20):
        return False, f"Áp út không duỗi đều với giữa: ratio_rw_mw={ratio_rw_mw:.2f}"
    
    # Trỏ phải gần tương đương Giữa (0.80 <= ratio <= 1.20)
    if not (0.80 <= ratio_iw_mw <= 1.25):
        return False, f"Trỏ không duỗi đều với giữa: ratio_iw_mw={ratio_iw_mw:.2f}"
    
    # 4. Phân biệt với chữ N: Trong chữ M, khoảng cách giữa đầu ngón Giữa (12) và Áp út (16) phải nhỏ
    # Chuẩn hóa khoảng cách giữa Giữa và Áp út theo chiều dài ngón Giữa
    d_mr = dist_3d(p[12], p[16])
    d_im = dist_3d(p[8], p[12])
    
    ratio_mr = d_mr / d_mw
    ratio_im = d_im / d_mw
    
    if ratio_mr > 0.28:
        return False, f"Đầu ngón giữa và áp út tách quá xa (có thể là N): ratio_mr={ratio_mr:.2f}"
    
    if ratio_im > 0.30:
        return False, f"Đầu ngón trỏ và giữa tách quá xa: ratio_im={ratio_im:.2f}"
    
    # 5. Ngón út phải gập ngắn hơn rõ rệt so với ngón áp út (để phân biệt với bàn tay xòe 4 ngón chúc xuống)
    ratio_pw_rw = d_pw / d_rw
    if ratio_pw_rw > 0.88:
        return False, f"Ngón út không gập lại: ratio_pw_rw={ratio_pw_rw:.2f}"
    
    return True, f"Khớp chữ M: ratio_rw_mw={ratio_rw_mw:.2f}, ratio_mr={ratio_mr:.2f}, ratio_pw_rw={ratio_pw_rw:.2f}"

# Test on M.json
with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

m_pass = 0
m_total = 0
m_stable_pass = 0
m_stable_total = 0

for s in m_data['samples']:
    m_total += 1
    is_stable = (s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok')
    if is_stable: m_stable_total += 1
    
    ok, reason = evaluate_rule_m(s.get('landmarks'))
    if ok:
        m_pass += 1
        if is_stable: m_stable_pass += 1
    else:
        if is_stable:
            print(f"M stable frame {s['frame_index']} rejected: {reason}")

print(f"\nResult on M.json: {m_pass}/{m_total} total, {m_stable_pass}/{m_stable_total} stable")

# Test on N.json
with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

n_pass = 0
n_total = 0
for s in n_data['samples']:
    n_total += 1
    ok, reason = evaluate_rule_m(s.get('landmarks'))
    if ok:
        n_pass += 1
        print(f"N frame {s['frame_index']} falsely matched as M: {reason}")

print(f"Result on N.json: {n_pass}/{n_total} falsely matched as M (Should be 0!)")

# Test on ALL other 20 letters in static dataset v2
npz_data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
all_labels = [npz_data['classes'][i] for i in npz_data['y']]
all_lms = npz_data['landmarks']

other_fps = {}
for lbl, lm in zip(all_labels, all_lms):
    if lbl == 'M':
        continue
    ok, reason = evaluate_rule_m(lm.tolist())
    if ok:
        other_fps[lbl] = other_fps.get(lbl, 0) + 1

print(f"False Positives on all other 21 letters in dataset v2: {other_fps} (Total: {sum(other_fps.values())})")

