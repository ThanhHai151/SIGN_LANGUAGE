import json
import math
import sys
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')
from backend.vision.feature_extractor import calculate_angle, calculate_distance, normalize_hand_landmarks
from backend.vision.digit_classifier import digit_classifier

with open('backend/tests/fixtures/real_digit_10_landmarks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

samples = data['samples']
print(f"Tổng số mẫu: {len(samples)}, Thành công: {data['successful_samples']}")
print("-" * 90)

for s in samples:
    fidx = s['frame_idx']
    ts = s['timestamp_sec']
    status = s['quality_status']
    phase = s['phase']
    if status != 'ok':
        print(f"Frame {fidx:02d} (t={ts:.2f}s, phase={phase}): NO_HAND_DETECTED")
        continue
    
    lms = s['landmarks']
    # Đánh giá qua VSLDigitClassifier hiện tại
    clf_res = digit_classifier.classify_detailed(lms)
    
    # Tính các góc và hướng
    wrist = lms[0]
    index_mcp = lms[5]
    index_pip = lms[6]
    index_tip = lms[8]
    middle_tip = lms[12]
    thumb_tip = lms[4]
    
    # Vector hướng ngón trỏ (mcp -> tip)
    v_idx = [index_tip[0] - index_mcp[0], index_tip[1] - index_mcp[1]]
    # Góc so với trục thẳng đứng Y (độ)
    angle_deg = math.degrees(math.atan2(v_idx[0], -v_idx[1]))
    
    # Vector cổ tay -> mcp (hướng bàn tay)
    v_hand = [index_mcp[0] - wrist[0], index_mcp[1] - wrist[1]]
    hand_angle_deg = math.degrees(math.atan2(v_hand[0], -v_hand[1]))
    
    # Độ gập ngón trỏ (khoảng cách tip -> wrist so với pip -> wrist)
    d_iw = calculate_distance(index_tip, wrist)
    d_mw = calculate_distance(middle_tip, wrist)
    d_tw = calculate_distance(thumb_tip, wrist)
    
    # Palm scale
    scale = calculate_distance(wrist, index_mcp)
    norm_iw = d_iw / scale if scale > 0 else 0
    norm_mw = d_mw / scale if scale > 0 else 0
    
    print(f"Frame {fidx:02d} (t={ts:.2f}s, {phase:<12}): clf={clf_res['digit']}({clf_res['status']}) | idx_angle={angle_deg:6.1f}° | hand_angle={hand_angle_deg:6.1f}° | norm_iw={norm_iw:.2f} | norm_mw={norm_mw:.2f}")

