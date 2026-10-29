import json
import math
import sys
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')
from backend.vision.feature_extractor import calculate_distance

# 1. Đọc dữ liệu số 10
with open('backend/tests/fixtures/real_digit_10_landmarks.json', 'r', encoding='utf-8') as f:
    d10_data = json.load(f)
d10_samples = [s for s in d10_data['samples'] if s['quality_status'] == 'ok']

# 2. Đọc dữ liệu 1-9
with open('backend/tests/fixtures/real_digit_landmarks.json', 'r', encoding='utf-8') as f:
    digits_data = json.load(f)

def extract_gesture_features(lms):
    wrist = lms[0]
    index_mcp = lms[5]
    index_tip = lms[8]
    middle_tip = lms[12]
    ring_tip = lms[16]
    pinky_tip = lms[20]
    
    scale = calculate_distance(wrist, index_mcp)
    if scale <= 1e-5:
        return None
    
    # Khoảng cách từ các đầu ngón tới cổ tay
    norm_iw = calculate_distance(index_tip, wrist) / scale
    norm_mw = calculate_distance(middle_tip, wrist) / scale
    norm_rw = calculate_distance(ring_tip, wrist) / scale
    norm_pw = calculate_distance(pinky_tip, wrist) / scale
    
    # 3 ngón giữa, áp út, út gập:
    tucked = (norm_mw < 0.85 and norm_rw < 0.85 and norm_pw < 0.85)
    
    # Góc hướng ngón trỏ (mcp -> tip) so với trục thẳng đứng Y (độ)
    v_idx = [index_tip[0] - index_mcp[0], index_tip[1] - index_mcp[1]]
    idx_angle_deg = math.degrees(math.atan2(v_idx[0], -v_idx[1]))
    
    return {
        "norm_iw": norm_iw,
        "norm_mw": norm_mw,
        "tucked": tucked,
        "idx_angle": idx_angle_deg
    }

print("=== MẪU SỐ 10 (D0539) THEO THỜI GIAN ===")
d10_angles = []
for s in d10_samples:
    feat = extract_gesture_features(s['landmarks'])
    d10_angles.append(feat['idx_angle'])
    print(f"Frm {s['frame_idx']:02d} (t={s['timestamp_sec']:.2f}s, {s['phase']:<12}): tucked={feat['tucked']}, norm_iw={feat['norm_iw']:.2f}, angle={feat['idx_angle']:6.1f}°")

angle_range = max(d10_angles) - min(d10_angles)
print(f"-> Biên độ góc ngón trỏ số 10: {angle_range:.1f}°")

print("\n=== MẪU SỐ 1 (DEV & EVAL) ===")
s1_samples = [s for s in digits_data['samples'] if s['digit'] == '1']
s1_angles = []
for s in s1_samples:
    feat = extract_gesture_features(s['landmarks'])
    s1_angles.append(feat['idx_angle'])
    print(f"Số 1 {s['split']:<4} (Frm {s['frame_idx']}): tucked={feat['tucked']}, norm_iw={feat['norm_iw']:.2f}, angle={feat['idx_angle']:6.1f}°")

angle_range_1 = max(s1_angles) - min(s1_angles)
print(f"-> Biên độ góc ngón trỏ số 1: {angle_range_1:.1f}°")

