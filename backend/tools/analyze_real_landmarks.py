"""
Script phân tích dữ liệu Landmark Bảng chữ cái VSL phục vụ Task 3B.0.
======================================================================
Tính toán các đặc trưng hình học từ landmark thật của 122 mẫu 'ok':
- finger_curl_ratio cho 5 ngón tay (Thumb, Index, Middle, Ring, Pinky)
- Khoảng cách giữa các đầu ngón tay (d_TI, d_IM, d_MR, d_RP, d_TW...) chuẩn hóa theo palm_scale
- Góc khớp ngón tay (PIP, MCP)
- Phân tích chi tiết các cặp dễ nhầm: R vs U, M vs N, S vs T, C vs O, U vs V, D vs L, A vs E
- Thống kê phân bố (min, max, mean, std) và mức độ chồng lấn đặc trưng
"""

import os
import sys
import json
import math
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, WORKSPACE)

from backend.vision.feature_extractor import (
    _shift_and_scale,
    finger_curl_ratio,
    calculate_distance,
    calculate_angle
)

FIXTURE_PATH = os.path.join(WORKSPACE, "backend", "tests", "fixtures", "real_alphabet_landmarks.json")

with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
    fixture = json.load(f)

samples = fixture["samples"]
ok_samples = [s for s in samples if s["quality_status"] == "ok"]
print(f"Tổng số mẫu trong fixture: {len(samples)}")
print(f"Số mẫu 'ok': {len(ok_samples)}")
print(f"Số mẫu 'no_hand_detected': {len(samples) - len(ok_samples)}")

# Trích xuất đặc trưng cho từng mẫu 'ok'
# Landmark indices:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

features_by_letter = {}

for s in ok_samples:
    l = s["letter"]
    lms = s["landmarks"]
    shifted, palm_scale = _shift_and_scale(lms)
    
    # 1. Độ co ngón (0 = duỗi thẳng, 1 = gập hoàn toàn)
    curl_thumb = finger_curl_ratio(shifted, 4, 3, 2, palm_scale)
    curl_index = finger_curl_ratio(shifted, 8, 6, 5, palm_scale)
    curl_middle = finger_curl_ratio(shifted, 12, 10, 9, palm_scale)
    curl_ring = finger_curl_ratio(shifted, 16, 14, 13, palm_scale)
    curl_pinky = finger_curl_ratio(shifted, 20, 18, 17, palm_scale)
    
    # 2. Khoảng cách đầu ngón chuẩn hóa theo palm_scale
    d_TI = calculate_distance(shifted[4], shifted[8]) / palm_scale   # Thumb tip - Index tip
    d_TM = calculate_distance(shifted[4], shifted[12]) / palm_scale  # Thumb tip - Middle tip
    d_TR = calculate_distance(shifted[4], shifted[16]) / palm_scale  # Thumb tip - Ring tip
    d_TP = calculate_distance(shifted[4], shifted[20]) / palm_scale  # Thumb tip - Pinky tip
    d_TW = calculate_distance(shifted[4], shifted[0]) / palm_scale   # Thumb tip - Wrist
    d_IM = calculate_distance(shifted[8], shifted[12]) / palm_scale  # Index tip - Middle tip
    d_MR = calculate_distance(shifted[12], shifted[16]) / palm_scale # Middle tip - Ring tip
    d_RP = calculate_distance(shifted[16], shifted[20]) / palm_scale # Ring tip - Pinky tip
    
    # Khoảng cách đầu ngón đến Wrist
    d_IW = calculate_distance(shifted[8], shifted[0]) / palm_scale   # Index tip - Wrist
    d_MW = calculate_distance(shifted[12], shifted[0]) / palm_scale  # Middle tip - Wrist
    d_PW = calculate_distance(shifted[20], shifted[0]) / palm_scale  # Pinky tip - Wrist

    # Độ chéo nhau giữa Index và Middle (R vs U)
    # Tọa độ x tương đối giữa Index tip và Middle tip
    diff_x_IM = (shifted[8][0] - shifted[12][0]) / palm_scale
    diff_y_IM = (shifted[8][1] - shifted[12][1]) / palm_scale
    diff_z_IM = (shifted[8][2] - shifted[12][2]) / palm_scale

    # Vị trí ngón cái so với ngón trỏ/giữa (S vs T, M vs N)
    # y của ngón cái so với y của MCP ngón trỏ (điểm 5) và ngón giữa (điểm 9)
    rel_y_thumb_index_mcp = (shifted[4][1] - shifted[5][1]) / palm_scale
    rel_x_thumb_index_tip = (shifted[4][0] - shifted[8][0]) / palm_scale

    feat = {
        "letter": l,
        "video_id": s["video_id"],
        "frame_idx": s["frame_idx"],
        "timestamp_sec": s["timestamp_sec"],
        "split": s["split"],
        "phase": s["phase"],
        "confidence": s["confidence"],
        "curl_thumb": round(curl_thumb, 4),
        "curl_index": round(curl_index, 4),
        "curl_middle": round(curl_middle, 4),
        "curl_ring": round(curl_ring, 4),
        "curl_pinky": round(curl_pinky, 4),
        "d_TI": round(d_TI, 4),
        "d_TM": round(d_TM, 4),
        "d_IM": round(d_IM, 4),
        "d_TW": round(d_TW, 4),
        "d_IW": round(d_IW, 4),
        "d_MW": round(d_MW, 4),
        "d_PW": round(d_PW, 4),
        "diff_x_IM": round(diff_x_IM, 4),
        "diff_z_IM": round(diff_z_IM, 4),
        "rel_y_thumb_index_mcp": round(rel_y_thumb_index_mcp, 4),
        "rel_x_thumb_index_tip": round(rel_x_thumb_index_tip, 4)
    }
    
    if l not in features_by_letter:
        features_by_letter[l] = []
    features_by_letter[l].append(feat)

print(f"\nĐã trích xuất đặc trưng cho {len(features_by_letter)} chữ cái.")

# Hàm tính thống kê
def get_stats(vals):
    if not vals:
        return {"min": 0, "max": 0, "mean": 0, "std": 0}
    return {
        "min": round(float(np.min(vals)), 3),
        "max": round(float(np.max(vals)), 3),
        "mean": round(float(np.mean(vals)), 3),
        "std": round(float(np.std(vals)), 3)
    }

# Phân tích từng cặp dễ nhầm
pairs_to_analyze = [
    ("r", "u", "R vs U (Bắt chéo ngón vs Khép song song)"),
    ("m", "n", "M vs N (3 ngón úp vs 2 ngón úp, vị trí ngón cái)"),
    ("s", "t", "S vs T (Nắm đấm ngón cái chéo vs Ngón cái chen kẹp)"),
    ("c", "o", "C vs O (Khum mở rỗng vs Khum chụm đầu ngón)"),
    ("u", "v", "U vs V (2 ngón khép sát vs 2 ngón xòe chữ V)"),
    ("d", "l", "D vs L (1 ngón trỏ duỗi ngón cái nắm vs Trỏ và cái vuông góc)"),
    ("a", "e", "A vs E (Ngón cái duỗi đứng cạnh nắm đấm vs 4 ngón cong đặt lên đầu ngón cái)")
]

pair_results = {}

for l1, l2, desc in pairs_to_analyze:
    f1 = features_by_letter.get(l1, [])
    f2 = features_by_letter.get(l2, [])
    
    print("\n" + "=" * 70)
    print(f"PHÂN TÍCH CẶP: {desc}")
    print(f"Số mẫu: '{l1}' = {len(f1)} mẫu | '{l2}' = {len(f2)} mẫu")
    print("-" * 70)
    
    keys_to_compare = [
        "curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky",
        "d_TI", "d_TM", "d_IM", "diff_x_IM", "diff_z_IM", "d_TW", "d_IW"
    ]
    
    comp_table = []
    for k in keys_to_compare:
        v1 = [x[k] for x in f1]
        v2 = [x[k] for x in f2]
        st1 = get_stats(v1)
        st2 = get_stats(v2)
        
        # Kiểm tra độ chồng lấn khoảng [min, max]
        overlap_min = max(st1["min"], st2["min"])
        overlap_max = min(st1["max"], st2["max"])
        is_overlap = overlap_max >= overlap_min
        overlap_range = f"[{overlap_min:.2f}, {overlap_max:.2f}]" if is_overlap else "Tách biệt hoàn toàn"
        
        comp_table.append({
            "feature": k,
            f"{l1}_mean_std": f"{st1['mean']:.3f} ± {st1['std']:.3f} [{st1['min']:.2f}, {st1['max']:.2f}]",
            f"{l2}_mean_std": f"{st2['mean']:.3f} ± {st2['std']:.3f} [{st2['min']:.2f}, {st2['max']:.2f}]",
            "overlap": is_overlap,
            "overlap_detail": overlap_range
        })
        print(f"{k:<15} | {l1}: {comp_table[-1][f'{l1}_mean_std']:<26} | {l2}: {comp_table[-1][f'{l2}_mean_std']:<26} | Chồng lấn: {overlap_range}")
    
    pair_results[f"{l1}_{l2}"] = {
        "description": desc,
        "sample_count": {l1: len(f1), l2: len(f2)},
        "table": comp_table
    }

output_analysis_json = os.path.join(WORKSPACE, "backend", "reports", "task_3b0_feature_analysis.json")
with open(output_analysis_json, "w", encoding="utf-8") as f:
    json.dump({
        "features_by_letter": features_by_letter,
        "pair_results": pair_results
    }, f, ensure_ascii=False, indent=2)

print(f"\n-> Đã lưu dữ liệu phân tích chi tiết vào: {output_analysis_json}")

