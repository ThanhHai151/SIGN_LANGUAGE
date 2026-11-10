"""
Phân tích chi tiết tọa độ landmark 21 điểm và đặc trưng hình học của M và N.
So sánh với các chữ nắm đấm: A, E, S, T.
"""
import json
import numpy as np
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

c = VSLAlphabetClassifier()

# 1. Load M and N JSON
with open("backend/data/alphabet_v2/static_landmarks/M.json", "r", encoding="utf-8") as f:
    data_m = json.load(f)
with open("backend/data/alphabet_v2/static_landmarks/N.json", "r", encoding="utf-8") as f:
    data_n = json.load(f)

stable_m = [s for s in data_m["samples"] if s.get("phase") == "stable_pose" and s.get("quality_status") == "ok"]
stable_n = [s for s in data_n["samples"] if s.get("phase") == "stable_pose" and s.get("quality_status") == "ok"]

print(f"M stable samples: {len(stable_m)}")
print(f"N stable samples: {len(stable_n)}")

def analyze_landmarks_set(name, samples):
    print(f"\n==================== CHI TIẾT ĐẶC TRƯNG CHỮ {name} ({len(samples)} samples) ====================")
    feats = []
    t_tips_x = []
    t_tips_y = []
    
    # Khoảng cách từ thumb_tip (4) đến các ngón
    d_thumb_index_pip = []  # 4 -> 6
    d_thumb_mid_pip = []    # 4 -> 10
    d_thumb_ring_pip = []   # 4 -> 14
    d_thumb_pinky_pip = []  # 4 -> 18
    
    d_thumb_mid_tip = []    # 4 -> 12
    d_thumb_ring_tip = []   # 4 -> 16
    d_thumb_pinky_tip = []  # 4 -> 20
    
    thumb_x_rel_mid = []    # x_4 - x_10 (độ lệch ngang so với ngón giữa)
    thumb_x_rel_ring = []   # x_4 - x_14 (độ lệch ngang so với ngón áp út)
    
    for s in samples:
        lms = s["landmarks"]
        f = c.extract_features(lms)
        feats.append(f)
        
        # normalize palm scale
        wrist = np.array(lms[0])
        mid_mcp = np.array(lms[9])
        palm = np.linalg.norm(mid_mcp - wrist) + 1e-6
        
        p = np.array(lms)
        p_rel = (p - wrist) / palm
        
        t_tip = p_rel[4]
        
        d_thumb_mid_pip.append(np.linalg.norm(p_rel[4] - p_rel[10]))
        d_thumb_ring_pip.append(np.linalg.norm(p_rel[4] - p_rel[14]))
        d_thumb_pinky_pip.append(np.linalg.norm(p_rel[4] - p_rel[18]))
        
        d_thumb_mid_tip.append(np.linalg.norm(p_rel[4] - p_rel[12]))
        d_thumb_ring_tip.append(np.linalg.norm(p_rel[4] - p_rel[16]))
        d_thumb_pinky_tip.append(np.linalg.norm(p_rel[4] - p_rel[20]))
        
        thumb_x_rel_mid.append(p_rel[4][0] - p_rel[10][0])
        thumb_x_rel_ring.append(p_rel[4][0] - p_rel[14][0])
        
    print(f"d(Thumb_Tip, Middle_PIP):  min={min(d_thumb_mid_pip):.3f}, max={max(d_thumb_mid_pip):.3f}, mean={np.mean(d_thumb_mid_pip):.3f}")
    print(f"d(Thumb_Tip, Ring_PIP):    min={min(d_thumb_ring_pip):.3f}, max={max(d_thumb_ring_pip):.3f}, mean={np.mean(d_thumb_ring_pip):.3f}")
    print(f"d(Thumb_Tip, Pinky_PIP):   min={min(d_thumb_pinky_pip):.3f}, max={max(d_thumb_pinky_pip):.3f}, mean={np.mean(d_thumb_pinky_pip):.3f}")
    print(f"d(Thumb_Tip, Middle_TIP):  min={min(d_thumb_mid_tip):.3f}, max={max(d_thumb_mid_tip):.3f}, mean={np.mean(d_thumb_mid_tip):.3f}")
    print(f"d(Thumb_Tip, Ring_TIP):    min={min(d_thumb_ring_tip):.3f}, max={max(d_thumb_ring_tip):.3f}, mean={np.mean(d_thumb_ring_tip):.3f}")
    print(f"d(Thumb_Tip, Pinky_TIP):   min={min(d_thumb_pinky_tip):.3f}, max={max(d_thumb_pinky_tip):.3f}, mean={np.mean(d_thumb_pinky_tip):.3f}")
    print(f"x(Thumb) - x(Middle_PIP):  min={min(thumb_x_rel_mid):.3f}, max={max(thumb_x_rel_mid):.3f}, mean={np.mean(thumb_x_rel_mid):.3f}")
    print(f"x(Thumb) - x(Ring_PIP):    min={min(thumb_x_rel_ring):.3f}, max={max(thumb_x_rel_ring):.3f}, mean={np.mean(thumb_x_rel_ring):.3f}")

analyze_landmarks_set("M", stable_m)
analyze_landmarks_set("N", stable_n)

