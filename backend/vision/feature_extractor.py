"""
Bo trich xuat va chuan hoa dac trung ban tay (Hand Feature Normalization)
Dam bao tinh bat bien voi khoang cach camera, kich thuoc ban tay va vi tri trong khung hinh.

Quy uoc chi so 21 Landmark cua MediaPipe (Hand):
  0  = Wrist (Co tay)
  1-4  = Thumb  (CMC, MCP, IP, TIP)
  5-8  = Index  (MCP, PIP, DIP, TIP)
  9-12 = Middle (MCP, PIP, DIP, TIP)
  13-16= Ring   (MCP, PIP, DIP, TIP)
  17-20= Pinky  (MCP, PIP, DIP, TIP)

He toa do MediaPipe (toa do chuan hoa [0,1]):
  x: tang sang phai
  y: tang xuong duoi (goc o goc tren-trai)
  z: tang ra gan camera (do sau, tuong doi so voi wrist)
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Ham tien ich co ban
# ---------------------------------------------------------------------------

def calculate_distance(p1, p2) -> float:
    """Khoang cach Euclid 3D giua 2 diem moc."""
    if p1 is None or p2 is None or len(p1) < 3 or len(p2) < 3:
        return 0.0
    try:
        dx = float(p1[0]) - float(p2[0])
        dy = float(p1[1]) - float(p2[1])
        dz = float(p1[2]) - float(p2[2])
    except (ValueError, TypeError):
        return 0.0
    d_sq = dx * dx + dy * dy + dz * dz
    if math.isnan(d_sq) or math.isinf(d_sq) or d_sq < 0.0:
        return 0.0
    return math.sqrt(d_sq)


def calculate_angle(a, b, c) -> float:
    """Tinh goc tai dinh b trong tam giac a-b-c, theo do (0-180)."""
    if a is None or b is None or c is None or len(a) < 3 or len(b) < 3 or len(c) < 3:
        return 0.0
    try:
        ba = np.array([float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])], dtype=float)
        bc = np.array([float(c[0]) - float(b[0]), float(c[1]) - float(b[1]), float(c[2]) - float(b[2])], dtype=float)
    except (ValueError, TypeError):
        return 0.0
    if np.isnan(ba).any() or np.isnan(bc).any() or np.isinf(ba).any() or np.isinf(bc).any():
        return 0.0
    norm_prod = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if norm_prod < 1e-7 or math.isnan(norm_prod) or math.isinf(norm_prod):
        return 0.0
    cosine_angle = np.clip(np.dot(ba, bc) / norm_prod, -1.0, 1.0)
    if math.isnan(cosine_angle):
        return 0.0
    return float(np.degrees(np.arccos(cosine_angle)))


def _shift_and_scale(landmarks_list):
    """
    Noi bo: Dich goc ve co tay va tinh palm_scale.
    Tra ve (shifted_list, palm_scale).
    shifted_list: list 21 diem [x,y,z] sau khi tru wrist.
    palm_scale: khoang cach wrist -> Middle MCP (landmark 9).
    """
    if not landmarks_list or len(landmarks_list) < 21:
        return [[0.0, 0.0, 0.0] for _ in range(21)], 1.0
    wrist = landmarks_list[0]
    if wrist is None or len(wrist) < 3:
        wrist = [0.0, 0.0, 0.0]
    shifted = []
    for lm in landmarks_list[:21]:
        if lm is None or len(lm) < 3:
            shifted.append([0.0, 0.0, 0.0])
        else:
            try:
                x = float(lm[0]) - float(wrist[0])
                y = float(lm[1]) - float(wrist[1])
                z = float(lm[2]) - float(wrist[2])
                if math.isnan(x) or math.isinf(x): x = 0.0
                if math.isnan(y) or math.isinf(y): y = 0.0
                if math.isnan(z) or math.isinf(z): z = 0.0
            except (ValueError, TypeError):
                x, y, z = 0.0, 0.0, 0.0
            shifted.append([x, y, z])
    palm_scale = calculate_distance(shifted[0], shifted[9])
    if palm_scale < 1e-5 or math.isnan(palm_scale) or math.isinf(palm_scale):
        palm_scale = 1.0
    return shifted, palm_scale


def finger_curl_ratio(shifted, tip_idx, pip_idx, mcp_idx, palm_scale) -> float:
    """
    Ti le co ngon tay: 0.0 = duoi thang, 1.0 = gap hoan toan.
    Do bang cach so sanh khoang cach TIP->MCP voi TIP->PIP + PIP->MCP.
    """
    if not shifted or len(shifted) <= max(tip_idx, pip_idx, mcp_idx):
        return 0.0
    d_tip_mcp = calculate_distance(shifted[tip_idx], shifted[mcp_idx])
    d_tip_pip = calculate_distance(shifted[tip_idx], shifted[pip_idx])
    d_pip_mcp = calculate_distance(shifted[pip_idx], shifted[mcp_idx])
    straight_len = d_tip_pip + d_pip_mcp
    if straight_len < 1e-7 or math.isnan(straight_len) or math.isinf(straight_len):
        return 0.0
    curl = 1.0 - (d_tip_mcp / straight_len)
    if math.isnan(curl) or math.isinf(curl):
        return 0.0
    return float(np.clip(curl, 0.0, 1.0))


def compute_finger_states(shifted, palm_scale) -> dict:
    """
    Tinh trang thai duoi/gap cho tung ngon dua tren toa do da shift.

    Nguyen tac:
    - Cac ngon tro/giua/ap ut/ut: so sanh y cua TIP voi PIP (toa do shifted).
      Vi y tang xuong duoi, TIP.y < PIP.y => ngon duoi len phia tren.
      Day la toa do tuong doi so voi wrist, khong phu thuoc vi tri tuyet doi.
    - Ngon cai: kiem tra khoang cach TIP->ngon tro MCP (p[5]) so voi
      IP->ngon tro MCP de xac dinh ngon cai co vuon ra ngoai khong.

    Tra ve dict voi cac khoa:
      thumb_extended, index_extended, middle_extended,
      ring_extended, pinky_extended,
      thumb_curl, index_curl, middle_curl, ring_curl, pinky_curl
    """
    default_states = {
        "thumb_extended": False,
        "index_extended": False,
        "middle_extended": False,
        "ring_extended": False,
        "pinky_extended": False,
        "thumb_curl": 0.0,
        "index_curl": 0.0,
        "middle_curl": 0.0,
        "ring_curl": 0.0,
        "pinky_curl": 0.0,
    }
    if not shifted or len(shifted) < 21:
        return default_states

    try:
        # --- Ngon tro, giua, ap ut, ut: so sanh TIP.y < PIP.y (shifted) ---
        index_ext  = shifted[8][1]  < shifted[6][1]
        middle_ext = shifted[12][1] < shifted[10][1]
        ring_ext   = shifted[16][1] < shifted[14][1]
        pinky_ext  = shifted[20][1] < shifted[18][1]

        # --- Ngon cai: dung khoang cach TIP vs IP so voi ngon tro MCP (p[5]) ---
        # Neu d(TIP, p[5]) > d(IP, p[5]) => ngon cai dang vuon ra xa => duoi
        d_tip_ref = calculate_distance(shifted[4], shifted[5])
        d_ip_ref  = calculate_distance(shifted[3], shifted[5])
        thumb_ext = d_tip_ref > d_ip_ref

        # --- Gia tri curl (0=duoi, 1=gap) ---
        thumb_curl  = finger_curl_ratio(shifted, 4, 3, 2, palm_scale)
        index_curl  = finger_curl_ratio(shifted, 8, 6, 5, palm_scale)
        middle_curl = finger_curl_ratio(shifted, 12, 10, 9, palm_scale)
        ring_curl   = finger_curl_ratio(shifted, 16, 14, 13, palm_scale)
        pinky_curl  = finger_curl_ratio(shifted, 20, 18, 17, palm_scale)

        return {
            "thumb_extended":  bool(thumb_ext),
            "index_extended":  bool(index_ext),
            "middle_extended": bool(middle_ext),
            "ring_extended":   bool(ring_ext),
            "pinky_extended":  bool(pinky_ext),
            "thumb_curl":   thumb_curl,
            "index_curl":   index_curl,
            "middle_curl":  middle_curl,
            "ring_curl":    ring_curl,
            "pinky_curl":   pinky_curl,
        }
    except (IndexError, TypeError, ValueError):
        return default_states


# ---------------------------------------------------------------------------
# Ham chuan hoa chinh
# ---------------------------------------------------------------------------

def normalize_hand_landmarks(landmarks_list):
    """
    Chuan hoa 21 diem moc ban tay:
    - Goc toa do dich ve Co tay (Landmark 0)
    - Chia cho khoang cach tu Co tay (0) den khop goc ngon giua (9) -> scale invariant

    Tra ve:
        feature_vector (np.ndarray, shape=(72,)): 63 toa do + 9 dac trung bo sung
        finger_states (dict): trang thai cac ngon tay
    """
    if not landmarks_list or len(landmarks_list) < 21:
        return np.zeros(72, dtype=np.float32), {}

    shifted, palm_scale = _shift_and_scale(landmarks_list)

    normalized_coords = []
    for pt in shifted:
        normalized_coords.extend([pt[0] / palm_scale, pt[1] / palm_scale, pt[2] / palm_scale])

    finger_states = compute_finger_states(shifted, palm_scale)

    # Khoang cach giua cac dau ngon tay (chuan hoa theo palm_scale)
    extra_features = [
        calculate_distance(shifted[4], shifted[8])  / palm_scale,  # Ngon cai - Ngon tro
        calculate_distance(shifted[8], shifted[12]) / palm_scale,  # Ngon tro - Ngon giua
        calculate_distance(shifted[4], shifted[12]) / palm_scale,  # Ngon cai - Ngon giua
        calculate_distance(shifted[4], shifted[20]) / palm_scale,  # Ngon cai - Ngon ut
        float(finger_states["thumb_extended"]),
        float(finger_states["index_extended"]),
        float(finger_states["middle_extended"]),
        float(finger_states["ring_extended"]),
        float(finger_states["pinky_extended"]),
    ]

    feature_vector = np.array(normalized_coords + extra_features, dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_vector, finger_states


# ---------------------------------------------------------------------------
# Trich xuat hai tay
# ---------------------------------------------------------------------------

def extract_dual_hand_features(landmarks_primary, landmarks_secondary=None):
    """
    Trich xuat dac trung hai ban tay dong thoi (Dual-Hand Feature Extraction):
    - Don tay: Neu chi co 1 tay, trich xuat 72 chieu va padding dac trung tuong quan.
    - Hai tay: Trich xuat dac trung ca 2 tay + tuong quan khong gian giua 2 tay
      (khoang cach 2 co tay, chenh lech do cao, ti le doi xung).
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

    # Khoang cach khong gian giua 2 co tay
    p_wrist = landmarks_primary[0]
    s_wrist = landmarks_secondary[0]
    wrist_dist = calculate_distance(p_wrist, s_wrist)
    height_delta = p_wrist[1] - s_wrist[1]  # < 0: primary cao hon; > 0: secondary cao hon

    # Khoang cach giua 2 ngon tro
    p_index = landmarks_primary[8]
    s_index = landmarks_secondary[8]
    index_dist = calculate_distance(p_index, s_index)

    # Diem tuong dong doi xung (ca 2 tay cung duoi hay cung gap)
    # Chi dung 5 khoa boolean de tinh symmetry
    bool_keys = ["thumb_extended", "index_extended", "middle_extended", "ring_extended", "pinky_extended"]
    match_count = sum(1 for k in bool_keys if states_prim.get(k) == states_sec.get(k))
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
