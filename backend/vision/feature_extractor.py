"""
Bộ trích xuất và chuẩn hóa đặc trưng bàn tay (Hand Feature Normalization)
Đảm bảo tính bất biến với khoảng cách camera, kích thước bàn tay và vị trí trong khung hình.
"""

import math
import numpy as np

def calculate_distance(p1, p2):
    """Khoảng cách Euclid 3D giữa 2 điểm mốc"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def calculate_angle(a, b, c):
    """Tính góc giữa 3 điểm a-b-c (khớp tại b) theo độ"""
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def normalize_hand_landmarks(landmarks_list):
    """
    Chuẩn hóa 21 điểm mốc bàn tay:
    - Gốc tọa độ dịch về Cổ tay (Landmark 0)
    - Chia cho khoảng cách từ Cổ tay (0) đến khớp gốc ngón giữa (9) -> scale invariant
    """
    if not landmarks_list or len(landmarks_list) < 21:
        return np.zeros(63, dtype=np.float32), {}

    # Dịch gốc về cổ tay
    wrist = landmarks_list[0]
    shifted = []
    for lm in landmarks_list:
        shifted.append([lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]])

    # Tính hệ số tỷ lệ lòng bàn tay
    palm_scale = calculate_distance(shifted[0], shifted[9])
    if palm_scale < 1e-5:
        palm_scale = 1.0

    normalized_coords = []
    for pt in shifted:
        normalized_coords.extend([pt[0] / palm_scale, pt[1] / palm_scale, pt[2] / palm_scale])

    # Trích xuất thêm các đặc trưng góc ngón tay & trạng thái duỗi/gập
    # Ngón cái: [1, 2, 4], Ngón trỏ: [0, 5, 8], Ngón giữa: [0, 9, 12], Áp út: [0, 13, 16], Út: [0, 17, 20]
    # Điểm mút ngón tay xa hơn khớp đốt ngón tay khi duỗi
    finger_states = {
        "thumb_extended": shifted[4][0] > shifted[3][0] if shifted[5][0] > 0 else shifted[4][0] < shifted[3][0],
        "index_extended": shifted[8][1] < shifted[6][1],      # Trong tọa độ ảnh, y hướng xuống
        "middle_extended": shifted[12][1] < shifted[10][1],
        "ring_extended": shifted[16][1] < shifted[14][1],
        "pinky_extended": shifted[20][1] < shifted[18][1],
    }

    # Khoảng cách giữa các đầu ngón tay
    extra_features = [
        calculate_distance(shifted[4], shifted[8]) / palm_scale,   # Ngón cái - Ngón trỏ
        calculate_distance(shifted[8], shifted[12]) / palm_scale,  # Ngón trỏ - Ngón giữa
        calculate_distance(shifted[4], shifted[12]) / palm_scale,  # Ngón cái - Ngón giữa
        calculate_distance(shifted[4], shifted[20]) / palm_scale,  # Ngón cái - Ngón út
        float(finger_states["thumb_extended"]),
        float(finger_states["index_extended"]),
        float(finger_states["middle_extended"]),
        float(finger_states["ring_extended"]),
        float(finger_states["pinky_extended"])
    ]

    feature_vector = np.array(normalized_coords + extra_features, dtype=np.float32)
    return feature_vector, finger_states

def extract_dual_hand_features(landmarks_primary, landmarks_secondary=None):
    """
    Trích xuất đặc trưng hai bàn tay đồng thời (Dual-Hand Feature Extraction):
    - Đơn tay: Nếu chỉ có 1 tay, trích xuất 72 chiều và padding đặc trưng tương quan.
    - Hai tay: Trích xuất đặc trưng cả 2 tay + tương quan không gian giữa 2 tay
      (khoảng cách 2 cổ tay, chênh lệch độ cao, tỷ lệ đối xứng).
    """
    feat_prim, states_prim = normalize_hand_landmarks(landmarks_primary)
    
    if not landmarks_secondary or len(landmarks_secondary) < 21:
        dual_meta = {
            "is_dual_hand": False,
            "wrist_distance": 0.0,
            "height_delta": 0.0,
            "symmetry_score": 0.0,
            "primary_states": states_prim,
            "secondary_states": {}
        }
        return feat_prim, dual_meta

    feat_sec, states_sec = normalize_hand_landmarks(landmarks_secondary)
    
    # Khoảng cách không gian giữa 2 cổ tay
    p_wrist = landmarks_primary[0]
    s_wrist = landmarks_secondary[0]
    wrist_dist = calculate_distance(p_wrist, s_wrist)
    height_delta = p_wrist[1] - s_wrist[1] # < 0: primary cao hơn; > 0: secondary cao hơn

    # Khoảng cách giữa 2 ngón trỏ
    p_index = landmarks_primary[8]
    s_index = landmarks_secondary[8]
    index_dist = calculate_distance(p_index, s_index)

    # Điểm tương đồng đối xứng (cả 2 tay cùng duỗi hay cùng gập)
    match_count = sum(
        1 for k in states_prim if states_prim.get(k) == states_sec.get(k)
    )
    symmetry_score = match_count / 5.0

    dual_meta = {
        "is_dual_hand": True,
        "wrist_distance": float(wrist_dist),
        "index_distance": float(index_dist),
        "height_delta": float(height_delta),
        "symmetry_score": float(symmetry_score),
        "primary_states": states_prim,
        "secondary_states": states_sec
    }

    return feat_prim, dual_meta

