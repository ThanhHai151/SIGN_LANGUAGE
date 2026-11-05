"""
Bộ phân loại hình học Bảng chữ cái VSL độc lập (VSL Alphabet Classifier).
========================================================================
Phân loại hình dạng bàn tay cho các chữ cái tĩnh VSL dựa trên 21 mốc landmark MediaPipe.

Đặc điểm kỹ thuật:
- Độc lập: Module kiểm thử/nghiên cứu độc lập, không thay đổi hay can thiệp vào luồng production (recognizer.py).
- An toàn: Kiểm tra nghiêm ngặt đầu vào (None, rỗng, sai số điểm, tọa độ phi số, NaN/Inf).
- Minh bạch: Đánh giá độc lập các điều kiện hình học cho từng chữ cái, không ép nhãn mù quáng bằng thác lệnh if...elif.
  Phát hiện rõ các trường hợp không khớp (NO_MATCH), xung đột (CONFLICT) hoặc đầu vào lỗi (INVALID_INPUT).
- Dữ liệu thực nghiệm: Ngưỡng hình học được xây dựng dựa trên fixture landmark thật trích xuất
  từ video QIPEDC (Task 3A, Task 3B.0).
- Phạm vi chữ tĩnh: Tập trung vào các chữ có căn cứ phân định rõ ràng (S/T, C/O, D/L, A/E, B, I, Y, V).
- Các chữ chưa đủ căn cứ phân tách (R/U, M/N) hoặc cử chỉ động (Ă, Â, Ê, Ô, Ơ, Ư, Đ) được bảo lưu an toàn ở trạng thái UNKNOWN.
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    from backend.vision.feature_extractor import (
        calculate_distance,
        calculate_angle,
        extract_dual_hand_features
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from backend.vision.feature_extractor import (
        calculate_distance,
        calculate_angle,
        extract_dual_hand_features
    )


# ---------------------------------------------------------------------------
# Các hằng số ngưỡng hình học chuẩn hóa từ dữ liệu video QIPEDC thực tế
# ---------------------------------------------------------------------------

# 1. Chữ B (4 ngón duỗi thẳng khép sát, ngón cái khép)
B_D_IW_MIN = 1.65
B_D_MW_MIN = 1.70
B_THUMB_LAT_MAX = 0.35

# 2. Chữ I & Y (Duỗi ngón út)
I_D_PW_MIN = 1.35
I_D_IW_MAX = 0.90
I_D_MW_MAX = 0.80
I_D_RW_MAX = 0.80
I_DIFF_PW_FOLDED_MIN = 0.55
I_D_TI_MAX = 0.42
I_D_TM_MAX = 0.52
I_D_TW_MAX = 1.15
I_THUMB_LAT_MAX = 0.38
I_ANG_TP_MAX = 33.0
I_PINKY_CURL_MAX = 0.20

Y_D_TW_MIN = 1.05
Y_D_TI_MIN = 0.40
Y_THUMB_LAT_MIN = 0.36
Y_ANG_TP_MIN = 33.0

# 3. Chữ V (Ngón trỏ & giữa xòe chữ V)
V_D_IW_MIN = 1.70
V_D_MW_MIN = 1.70
V_D_IM_MIN = 0.28
V_DIFF_X_IM_MIN = 0.18
V_ANG_IM_MIN = 9.0
V_IDX_HORIZ_MAX = 0.50
V_IDX_TIP_Y_MAX = -0.80
V_MCP_WIDTH_MIN = 0.48
V_THUMB_RING_MIN = 0.21

# 4. Cặp D vs L (Trỏ duỗi đơn)
D_D_IW_MIN = 1.70
D_D_TI_MAX = 1.28
D_D_TM_MAX = 0.30

L_D_IW_MIN = 1.70
L_D_TI_MIN = 1.40
L_D_TM_MIN = 0.45
L_THUMB_LAT_MIN = 0.30
L_ANG_TI_MIN = 35.0

# 5. Chữ T (Ngón cái kẹp giữa trỏ và giữa, trỏ quặp trên ngón cái)
T_D_TI_MAX = 0.16
T_D_TM_MIN = 0.80
T_D_IM_MIN = 0.80
T_D_IW_MAX = 1.30

# 6. Chữ O (Vòng tròn kín chụm đầu các ngón vào ngón cái)
O_D_TI_MAX = 0.22
O_D_TM_MAX = 0.20
O_D_IW_MIN = 0.95
O_D_MW_MIN = 0.90

# 7. Chữ C (Vòm cong mở chữ C)
C_D_TI_MIN = 0.55
C_D_TI_MAX = 0.85
C_D_TM_MIN = 0.50
C_D_TM_MAX = 0.85
C_D_IW_MIN = 1.35
C_D_IW_MAX = 1.65
C_D_MW_MIN = 1.20
C_D_MW_MAX = 1.55
C_D_PW_MIN = 1.05
C_D_PW_MAX = 1.50

# 8. Chữ A (Nắm đấm với ngón cái dựng đứng thẳng cạnh bàn tay)
A_D_TW_MIN = 1.32
A_D_TM_MIN = 0.75
A_D_IW_MAX = 0.90
A_D_MW_MAX = 0.75

# 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón)
S_D_TW_MAX = 0.90
S_D_TI_MIN = 0.25
S_D_TI_MAX = 0.45
S_D_TM_MIN = 0.40
S_D_TM_MAX = 0.65
S_D_IW_MAX = 0.90
S_D_MW_MAX = 0.75

# 10. Chữ E (4 ngón tì gập lên ngón cái)
E_D_TI_MAX = 0.50
E_D_TM_MIN = 0.26
E_D_TM_MAX = 0.62
E_D_TW_MAX = 0.96
E_D_PW_MIN = 0.65
E_D_PW_MAX = 1.35

# 11. Chữ G (Ngón trỏ duỗi ngang chỉ sang bên, 3 ngón giữa/áp út/út gập gọn)
G_D_IW_MIN = 1.50
G_IDX_STRAIGHT_MIN = 0.90
G_IDX_CURL_MAX = 0.15
G_D_IW_MW_DIFF_MIN = 0.55
G_D_MW_MAX = 1.25
G_D_RW_MAX = 1.15
G_D_PW_MAX = 1.05
G_M_CURL_MIN = 0.35
G_R_CURL_MIN = 0.35
G_P_CURL_MIN = 0.30
G_IDX_HORIZ_MIN = 0.65
G_IDX_TIP_Y_MAX = 0.25
G_D_TI_MAX = 1.35
G_D_TM_MIN = 0.22
G_ANG_TI_MAX = 42.0

# 12. Chữ H (Hỗ trợ cả 3 biến thể thực tế:
#     - H-1 chuẩn từ điển: Ngón trỏ và ngón giữa duỗi thẳng song song hướng ngang, áp út & út gập.
#     - H-2 phương ngữ Miền Bắc QIPEDC D0500B: Trỏ, áp út, út duỗi thẳng; ngón giữa gập chạm ngón cái.
#     - H-3 phương ngữ Miền Nam QIPEDC D0500N: Hông/má bàn tay hướng camera (MCP_width_2D <= 0.40),
#       ngón cái đè ngón áp út (Thumb-Ring <= 0.16), trỏ & giữa duỗi hướng đứng (có thể xòe hoặc khép).)
H_HORIZ_D_IW_MIN = 1.40
H_HORIZ_D_MW_MIN = 1.38
H_HORIZ_STRAIGHT_MIN = 0.85
H_HORIZ_STRAIGHT_EXT_MIN = 0.80
H_HORIZ_CURL_MAX = 0.26
H_HORIZ_DIFF_RP_MIN = 0.32
H_HORIZ_D_RW_MAX = 1.48
H_HORIZ_D_PW_MAX = 1.48
H_HORIZ_R_CURL_MIN = 0.20
H_HORIZ_P_CURL_MIN = 0.18
H_HORIZ_RATIO_MIN = 0.50
H_HORIZ_ANG_IM_MAX = 28.0
H_HORIZ_D_IM_MAX = 0.72
H_HORIZ_TIP_Y_MAX = 0.85

H_NORTH_D_IW_MIN = 1.65
H_NORTH_D_RW_MIN = 1.80
H_NORTH_D_PW_MIN = 1.45
H_NORTH_D_MW_MAX = 1.00
H_NORTH_DIFF_IW_MW_MIN = 0.65
H_NORTH_DIFF_RW_MW_MIN = 0.80
H_NORTH_DIFF_PW_MW_MIN = 0.50
H_NORTH_IRP_CURL_MAX = 0.10
H_NORTH_M_CURL_MIN = 0.60
H_NORTH_D_TM_MAX = 0.20

H_SOUTH_D_IW_MIN = 1.60
H_SOUTH_D_MW_MIN = 1.60
H_SOUTH_MCP_WIDTH_MAX = 0.40
H_SOUTH_THUMB_RING_MAX = 0.16
H_SOUTH_THUMB_MID_MAX = 0.40
H_SOUTH_IDX_HORIZ_MAX = 0.50
H_SOUTH_IDX_TIP_Y_MAX = -0.80

# 13. Chữ K (Ngón trỏ duỗi đứng, ngón giữa duỗi ngang, ngón cái kẹp giữa, áp út & út gập)
K_D_IW_MIN = 1.65
K_IDX_CURL_MAX = 0.15
K_IDX_HORIZ_MAX = 0.35
K_IDX_STRAIGHT_MIN = 0.90

K_D_MW_MIN = 1.00
K_D_MW_MAX = 1.55
K_MID_CURL_MAX = 0.15
K_MID_STRAIGHT_MIN = 0.85
K_MID_HORIZ_MIN = 0.55

K_ANG_IM_MIN = 60.0
K_ANG_IM_MAX = 115.0
K_D_IM_MIN = 0.75

K_D_RW_MAX = 1.00
K_D_PW_MAX = 0.90
K_DIFF_IW_RP_MIN = 0.80

K_D_TI_MAX = 1.10
K_D_TM_MAX = 0.85
K_THUMB_MID_MAX = 0.50



def validate_input_landmarks(landmarks: Any) -> Tuple[bool, str]:
    """
    Kiểm tra tính hợp lệ cơ bản của danh sách điểm mốc bàn tay.
    Trả về: (is_valid: bool, reason: str)
    """
    if landmarks is None:
        return False, "Landmarks là None"
    if not isinstance(landmarks, (list, tuple)):
        return False, f"Kiểu dữ liệu không hợp lệ: mong muốn list/tuple, nhận {type(landmarks).__name__}"
    if len(landmarks) != 21:
        return False, f"Số lượng điểm không đúng: mong muốn 21 điểm, nhận {len(landmarks)} điểm"

    for idx, pt in enumerate(landmarks):
        if not isinstance(pt, (list, tuple)) or len(pt) < 3:
            return False, f"Mốc {idx} không đủ 3 tọa độ (x, y, z)"
        try:
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
        except (ValueError, TypeError):
            return False, f"Mốc {idx} chứa giá trị không phải số"
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            return False, f"Mốc {idx} chứa giá trị NaN"
        if math.isinf(x) or math.isinf(y) or math.isinf(z):
            return False, f"Mốc {idx} chứa giá trị vô cực (Inf)"

    return True, "Hợp lệ"


class VSLAlphabetClassifier:
    """Bộ phân loại hình học độc lập cho các chữ cái tĩnh VSL."""

    def __init__(self):
        pass

    def extract_features(self, landmarks: List[List[float]]) -> Optional[Dict[str, Any]]:
        """
        Trích xuất các đặc trưng hình học chuẩn hóa từ 21 điểm mốc.
        Trả về None nếu palm_scale bằng 0 (dữ liệu suy biến).
        """
        _, meta = extract_dual_hand_features(landmarks, None)
        states = meta.get("primary_states", {})

        p = landmarks
        wrist = p[0]
        sh = [[p[i][0] - wrist[0], p[i][1] - wrist[1], p[i][2] - wrist[2]] for i in range(21)]
        palm_scale = calculate_distance(sh[0], sh[9])
        if palm_scale < 1e-5:
            return None

        # Trạng thái duỗi ngón (True/False)
        t_ext = bool(states.get("thumb_extended", False))
        i_ext = bool(states.get("index_extended", False))
        m_ext = bool(states.get("middle_extended", False))
        r_ext = bool(states.get("ring_extended", False))
        p_ext = bool(states.get("pinky_extended", False))

        # Tỷ lệ co gập ngón (0.0: thẳng -> 1.0: gập sát)
        i_curl = float(states.get("index_curl", 0.5))
        m_curl = float(states.get("middle_curl", 0.5))
        r_curl = float(states.get("ring_curl", 0.5))
        p_curl = float(states.get("pinky_curl", 0.5))

        # Các góc và khoảng cách chuẩn hóa theo kích thước lòng bàn tay (palm_scale)
        ang_ti = float(calculate_angle(sh[4], sh[0], sh[8]))
        ang_tp = float(calculate_angle(sh[4], sh[0], sh[20]))
        thumb_lat = float(abs(sh[4][0] - sh[0][0]) / palm_scale)

        d_ti = float(calculate_distance(sh[4], sh[8]) / palm_scale)
        d_tm = float(calculate_distance(sh[4], sh[12]) / palm_scale)
        d_tw = float(calculate_distance(sh[4], sh[0]) / palm_scale)
        d_im = float(calculate_distance(sh[8], sh[12]) / palm_scale)
        d_iw = float(calculate_distance(sh[8], sh[0]) / palm_scale)
        d_mw = float(calculate_distance(sh[12], sh[0]) / palm_scale)
        d_rw = float(calculate_distance(sh[16], sh[0]) / palm_scale)
        d_pw = float(calculate_distance(sh[20], sh[0]) / palm_scale)

        diff_x_im = float(abs(sh[8][0] - sh[12][0]) / palm_scale)

        # Đặc trưng hướng ngang và độ thẳng 2D của ngón trỏ & ngón giữa (bất biến hướng trái/phải cho chữ G, H)
        vx_58 = float((sh[8][0] - sh[5][0]) / palm_scale)
        vy_58 = float((sh[8][1] - sh[5][1]) / palm_scale)
        len_58 = math.hypot(vx_58, vy_58) + 1e-8
        idx_horiz_ratio = float(abs(vx_58) / len_58)
        seg_58 = (
            math.hypot(sh[6][0] - sh[5][0], sh[6][1] - sh[5][1])
            + math.hypot(sh[7][0] - sh[6][0], sh[7][1] - sh[6][1])
            + math.hypot(sh[8][0] - sh[7][0], sh[8][1] - sh[7][1])
        ) / palm_scale + 1e-8
        idx_straight = float(len_58 / seg_58)
        idx_tip_y = float(sh[8][1] / palm_scale)

        vx_9_12 = float((sh[12][0] - sh[9][0]) / palm_scale)
        vy_9_12 = float((sh[12][1] - sh[9][1]) / palm_scale)
        len_9_12 = math.hypot(vx_9_12, vy_9_12) + 1e-8
        mid_horiz_ratio = float(abs(vx_9_12) / len_9_12)
        seg_9_12 = (
            math.hypot(sh[10][0] - sh[9][0], sh[10][1] - sh[9][1])
            + math.hypot(sh[11][0] - sh[10][0], sh[11][1] - sh[10][1])
            + math.hypot(sh[12][0] - sh[11][0], sh[12][1] - sh[11][1])
        ) / palm_scale + 1e-8
        mid_straight = float(len_9_12 / seg_9_12)
        mid_tip_y = float(sh[12][1] / palm_scale)

        cos_im = max(-1.0, min(1.0, (vx_58 * vx_9_12 + vy_58 * vy_9_12) / (len_58 * len_9_12)))
        ang_im_dir = float(math.degrees(math.acos(cos_im)))

        # Đặc trưng định hướng bàn tay và quan hệ ngón cái để phân biệt H (Miền Nam) vs V
        mcp_width_2d = float(math.hypot(sh[17][0] - sh[5][0], sh[17][1] - sh[5][1]) / palm_scale)
        thumb_ring_dist = float(calculate_distance(sh[4], sh[14]) / palm_scale)
        thumb_mid_dist = float(calculate_distance(sh[4], sh[10]) / palm_scale)

        return {
            "palm_scale": round(palm_scale, 4),
            "thumb_ext": t_ext,
            "index_ext": i_ext,
            "middle_ext": m_ext,
            "ring_ext": r_ext,
            "pinky_ext": p_ext,
            "index_curl": round(i_curl, 4),
            "middle_curl": round(m_curl, 4),
            "ring_curl": round(r_curl, 4),
            "pinky_curl": round(p_curl, 4),
            "ang_TI": round(ang_ti, 2),
            "ang_TP": round(ang_tp, 2),
            "thumb_lat": round(thumb_lat, 4),
            "d_TI": round(d_ti, 4),
            "d_TM": round(d_tm, 4),
            "d_TW": round(d_tw, 4),
            "d_IM": round(d_im, 4),
            "d_IW": round(d_iw, 4),
            "d_MW": round(d_mw, 4),
            "d_RW": round(d_rw, 4),
            "d_PW": round(d_pw, 4),
            "diff_x_IM": round(diff_x_im, 4),
            "idx_horiz_ratio": round(idx_horiz_ratio, 4),
            "idx_straight": round(idx_straight, 4),
            "idx_tip_y": round(idx_tip_y, 4),
            "idx_dx": round(vx_58, 4),
            "mid_horiz_ratio": round(mid_horiz_ratio, 4),
            "mid_straight": round(mid_straight, 4),
            "mid_tip_y": round(mid_tip_y, 4),
            "ang_IM_dir": round(ang_im_dir, 2),
            "mcp_width_2d": round(mcp_width_2d, 4),
            "thumb_ring_dist": round(thumb_ring_dist, 4),
            "thumb_mid_dist": round(thumb_mid_dist, 4),
        }

    def evaluate_rules(self, feat: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """
        Đánh giá độc lập các quy tắc hình học cho các chữ cái tĩnh đã có đủ căn cứ dữ liệu.
        Trả về danh sách các bộ (letter, reason, confidence) thỏa mãn điều kiện.
        """
        matches = []

        t_ext = feat["thumb_ext"]
        i_ext = feat["index_ext"]
        m_ext = feat["middle_ext"]
        r_ext = feat["ring_ext"]
        p_ext = feat["pinky_ext"]

        d_ti = feat["d_TI"]
        d_tm = feat["d_TM"]
        d_tw = feat["d_TW"]
        d_im = feat["d_IM"]
        d_iw = feat["d_IW"]
        d_mw = feat["d_MW"]
        d_pw = feat["d_PW"]
        diff_x_im = feat["diff_x_IM"]
        ang_ti = feat["ang_TI"]
        thumb_lat = feat["thumb_lat"]

        # -------------------------------------------------------------------
        # 1. Chữ B (4 ngón duỗi thẳng khép sát, ngón cái khép)
        # -------------------------------------------------------------------
        if i_ext and m_ext and r_ext and p_ext:
            if d_iw >= B_D_IW_MIN and d_mw >= B_D_MW_MIN and thumb_lat <= B_THUMB_LAT_MAX:
                matches.append(("B", f"4 ngón duỗi thẳng khép sát: d_IW={d_iw:.2f}>={B_D_IW_MIN}, d_MW={d_mw:.2f}>={B_D_MW_MIN}", 0.95))

        # -------------------------------------------------------------------
        # 2. Chữ I & Chữ Y (Duỗi ngón út)
        # -------------------------------------------------------------------
        d_rw = float(feat.get("d_RW", 0.0))
        ang_tp = float(feat.get("ang_TP", 0.0))
        p_curl = float(feat.get("pinky_curl", 1.0))

        # Chữ I: Duy nhất ngón út duỗi thẳng, 3 ngón giữa gập sát lòng bàn tay, ngón cái gấp ôm vào ngón trỏ/giữa (không xòe Shaka)
        is_i_pinky = (p_ext or p_curl <= 0.12) and (d_pw >= I_D_PW_MIN) and (p_curl <= I_PINKY_CURL_MAX)
        is_i_middle_curled = (not i_ext) and (not m_ext) and (not r_ext)
        is_i_middle_dists = (d_iw <= I_D_IW_MAX) and (d_mw <= I_D_MW_MAX) and (d_rw <= I_D_RW_MAX)
        is_i_diff = (d_pw - max(d_iw, d_mw, d_rw)) >= I_DIFF_PW_FOLDED_MIN
        is_i_thumb_tucked = (
            d_ti <= I_D_TI_MAX
            and d_tm <= I_D_TM_MAX
            and d_tw <= I_D_TW_MAX
            and (thumb_lat <= I_THUMB_LAT_MAX or ang_tp <= I_ANG_TP_MAX)
        )

        if is_i_pinky and is_i_middle_curled and is_i_middle_dists and is_i_diff and is_i_thumb_tucked:
            matches.append(("I", f"Chỉ ngón út duỗi thẳng, 4 ngón gập ôm: d_PW={d_pw:.2f}>={I_D_PW_MIN}, d_TI={d_ti:.2f}<={I_D_TI_MAX}", 0.95))

        # Chữ Y: Ngón cái và ngón út cùng duỗi thẳng (Shaka), 3 ngón giữa gập
        is_y_pinky = (p_ext or p_curl <= 0.12) and (d_pw >= 1.30)
        is_y_middle_curled = (not i_ext) and (not m_ext) and (not r_ext)
        is_y_middle_dists = (d_iw <= I_D_IW_MAX) and (d_mw <= I_D_MW_MAX) and (d_rw <= I_D_RW_MAX)
        is_y_thumb_spread = (
            d_tw >= Y_D_TW_MIN
            and d_ti >= Y_D_TI_MIN
            and (thumb_lat >= Y_THUMB_LAT_MIN or ang_tp >= Y_ANG_TP_MIN)
        )

        if is_y_pinky and is_y_middle_curled and is_y_middle_dists and is_y_thumb_spread:
            matches.append(("Y", f"Ngón cái và út cùng duỗi thẳng (Shaka): d_PW={d_pw:.2f}>=1.30, d_TW={d_tw:.2f}>={Y_D_TW_MIN}", 0.95))

        # -------------------------------------------------------------------
        # 3. Chữ V (Trỏ & Giữa xòe hình chữ V, hướng lên trên, lòng bàn tay hướng camera)
        # -------------------------------------------------------------------
        idx_horiz_ratio = float(feat.get("idx_horiz_ratio", 0.0))
        idx_tip_y = float(feat.get("idx_tip_y", 0.0))
        ang_im_dir = float(feat.get("ang_IM_dir", 0.0))
        mcp_width_2d = feat.get("mcp_width_2d")
        thumb_ring_dist = feat.get("thumb_ring_dist")

        v_width_ok = (mcp_width_2d is not None and mcp_width_2d >= V_MCP_WIDTH_MIN)
        v_thumb_ok = (thumb_ring_dist is None or thumb_ring_dist >= V_THUMB_RING_MIN)

        if i_ext and m_ext and not r_ext and not p_ext:
            if (
                d_iw >= V_D_IW_MIN
                and d_mw >= V_D_MW_MIN
                and d_im >= V_D_IM_MIN
                and diff_x_im >= V_DIFF_X_IM_MIN
                and ang_im_dir >= V_ANG_IM_MIN
                and idx_horiz_ratio <= V_IDX_HORIZ_MAX
                and idx_tip_y <= V_IDX_TIP_Y_MAX
                and v_width_ok
                and v_thumb_ok
            ):
                matches.append((
                    "V",
                    f"Trỏ & Giữa xòe chữ V: d_IM={d_im:.2f}>={V_D_IM_MIN}, ang_IM={ang_im_dir:.1f}°>={V_ANG_IM_MIN}°, MCP_width={mcp_width_2d:.2f}>={V_MCP_WIDTH_MIN}",
                    0.95
                ))

        # -------------------------------------------------------------------
        # 4. Cặp D vs L (Trỏ duỗi đơn, Giữa/Áp út/Út gập)
        # -------------------------------------------------------------------
        if i_ext and not m_ext and not r_ext and not p_ext:
            # Chữ D: Chỉ ngón trỏ duỗi thẳng, ngón cái khép chạm đầu ngón giữa
            if d_iw >= D_D_IW_MIN and d_ti <= D_D_TI_MAX and d_tm <= D_D_TM_MAX:
                matches.append(("D", f"Ngón trỏ duỗi đơn, cái chạm giữa: d_TI={d_ti:.2f}<={D_D_TI_MAX}, d_TM={d_tm:.2f}<={D_D_TM_MAX}", 0.95))

            # Chữ L: Ngón cái và ngón trỏ mở rộng góc vuông (L-shape)
            if d_iw >= L_D_IW_MIN and d_ti >= L_D_TI_MIN and d_tm >= L_D_TM_MIN and thumb_lat >= L_THUMB_LAT_MIN and ang_ti >= L_ANG_TI_MIN:
                matches.append(("L", f"Trỏ và Cái mở vuông góc L: d_TI={d_ti:.2f}>={L_D_TI_MIN}, ang_TI={ang_ti:.1f}>={L_ANG_TI_MIN}", 0.95))

        # -------------------------------------------------------------------
        # 5. Chữ T (Ngón cái kẹp giữa ngón trỏ và ngón giữa)
        # -------------------------------------------------------------------
        if d_ti <= T_D_TI_MAX and d_tm >= T_D_TM_MIN and d_im >= T_D_IM_MIN and d_iw <= T_D_IW_MAX:
            matches.append(("T", f"Ngón cái kẹp giữa trỏ & giữa: d_TI={d_ti:.2f}<={T_D_TI_MAX}, d_IM={d_im:.2f}>={T_D_IM_MIN}", 0.95))

        # -------------------------------------------------------------------
        # 6. Chữ O (Vòng tròn kín chụm đầu các ngón vào ngón cái)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if d_ti <= O_D_TI_MAX and d_tm <= O_D_TM_MAX and d_iw >= O_D_IW_MIN and d_mw >= O_D_MW_MIN:
                matches.append(("O", f"Vòng tròn kín chữ O: d_TI={d_ti:.2f}<={O_D_TI_MAX}, d_TM={d_tm:.2f}<={O_D_TM_MAX}", 0.95))

        # -------------------------------------------------------------------
        # 7. Chữ C (Vòm cong mở chữ C)
        # -------------------------------------------------------------------
        if (C_D_TI_MIN <= d_ti <= C_D_TI_MAX and C_D_TM_MIN <= d_tm <= C_D_TM_MAX and 
            C_D_IW_MIN <= d_iw <= C_D_IW_MAX and C_D_MW_MIN <= d_mw <= C_D_MW_MAX and 
            C_D_PW_MIN <= d_pw <= C_D_PW_MAX):
            matches.append(("C", f"Vòm cong mở chữ C: d_TI={d_ti:.2f}, d_IW={d_iw:.2f}", 0.95))

        # -------------------------------------------------------------------
        # 8. Chữ A (Nắm đấm với ngón cái dựng đứng thẳng cạnh bàn tay)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if d_tw >= A_D_TW_MIN and d_tm >= A_D_TM_MIN and d_iw <= A_D_IW_MAX and d_mw <= A_D_MW_MAX:
                matches.append(("A", f"Nắm tay ngón cái dựng đứng: d_TW={d_tw:.2f}>={A_D_TW_MIN}, d_TM={d_tm:.2f}>={A_D_TM_MIN}", 0.95))

        # -------------------------------------------------------------------
        # 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if (d_tw <= S_D_TW_MAX and S_D_TI_MIN <= d_ti <= S_D_TI_MAX and 
                S_D_TM_MIN <= d_tm <= S_D_TM_MAX and d_iw <= S_D_IW_MAX and d_mw <= S_D_MW_MAX):
                matches.append(("S", f"Nắm đấm ngón cái vắt ngang: d_TI={d_ti:.2f}, d_TW={d_tw:.2f}<={S_D_TW_MAX}", 0.95))

        # -------------------------------------------------------------------
        # 10. Chữ E (4 ngón tì gập lên ngón cái)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and (not p_ext or d_pw <= E_D_PW_MAX):
            if (d_tw <= E_D_TW_MAX and 
                d_ti <= E_D_TI_MAX and 
                E_D_TM_MIN <= d_tm <= E_D_TM_MAX and 
                E_D_PW_MIN <= d_pw <= E_D_PW_MAX):
                # Loại trừ chữ S (nắm đấm ngón cái vắt ngang với d_iw <= 0.90, d_mw <= 0.75)
                is_s_fist = (d_iw <= S_D_IW_MAX and d_mw <= S_D_MW_MAX and S_D_TI_MIN <= d_ti <= S_D_TI_MAX)
                if not is_s_fist:
                    matches.append(("E", f"4 ngón tì gập lên ngón cái: d_TM={d_tm:.2f}, d_TW={d_tw:.2f}<={E_D_TW_MAX}", 0.95))

        # -------------------------------------------------------------------
        # 11. Chữ G (Ngón trỏ duỗi ngang chỉ sang bên, 3 ngón giữa/áp út/út gập)
        # -------------------------------------------------------------------
        d_rw = float(feat.get("d_RW", 99.0))
        i_curl = float(feat.get("index_curl", 1.0))
        m_curl = float(feat.get("middle_curl", 0.0))
        r_curl = float(feat.get("ring_curl", 0.0))
        p_curl = float(feat.get("pinky_curl", 0.0))
        idx_horiz_ratio = float(feat.get("idx_horiz_ratio", 0.0))
        idx_straight = float(feat.get("idx_straight", 0.0))
        idx_tip_y = float(feat.get("idx_tip_y", 99.0))

        single_idx_up = (i_ext and not m_ext and not r_ext and not p_ext)
        is_d_match = (single_idx_up and d_iw >= D_D_IW_MIN and d_ti <= D_D_TI_MAX and d_tm <= D_D_TM_MAX)
        is_l_match = (
            single_idx_up and d_iw >= L_D_IW_MIN and d_ti >= L_D_TI_MIN and d_tm >= L_D_TM_MIN
            and thumb_lat >= L_THUMB_LAT_MIN and ang_ti >= L_ANG_TI_MIN
        )
        if not is_d_match and not is_l_match:
            if (
                d_iw >= G_D_IW_MIN
                and idx_straight >= G_IDX_STRAIGHT_MIN
                and i_curl <= G_IDX_CURL_MAX
                and (d_iw - d_mw) >= G_D_IW_MW_DIFF_MIN
                and d_mw <= G_D_MW_MAX
                and d_rw <= G_D_RW_MAX
                and d_pw <= G_D_PW_MAX
                and m_curl >= G_M_CURL_MIN
                and r_curl >= G_R_CURL_MIN
                and p_curl >= G_P_CURL_MIN
                and idx_horiz_ratio >= G_IDX_HORIZ_MIN
                and idx_tip_y <= G_IDX_TIP_Y_MAX
                and d_ti <= G_D_TI_MAX
                and d_tm >= G_D_TM_MIN
                and ang_ti <= G_ANG_TI_MAX
            ):
                matches.append((
                    "G",
                    f"Ngón trỏ duỗi ngang chữ G: d_IW={d_iw:.2f}>={G_D_IW_MIN}, horiz={idx_horiz_ratio:.2f}>={G_IDX_HORIZ_MIN}",
                    0.95
                ))

        # -------------------------------------------------------------------
        # 12. Chữ H (Hỗ trợ cả 3 biến thể:
        #            - H ngang chuẩn từ điển
        #            - Biến thể Miền Bắc QIPEDC D0500B: trỏ/áp út/út duỗi, giữa gập chạm cái
        #            - Biến thể Miền Nam QIPEDC D0500N: hông bàn tay hướng camera, trỏ & giữa duỗi)
        # -------------------------------------------------------------------
        mid_horiz_ratio = float(feat.get("mid_horiz_ratio", 0.0))
        mid_straight = float(feat.get("mid_straight", 0.0))
        mid_tip_y = float(feat.get("mid_tip_y", 99.0))
        ang_im_dir = float(feat.get("ang_IM_dir", 180.0))
        mcp_width_2d = feat.get("mcp_width_2d")
        thumb_ring_dist = feat.get("thumb_ring_dist")
        thumb_mid_dist = feat.get("thumb_mid_dist")

        is_b_match = (
            i_ext and m_ext and r_ext and p_ext
            and d_iw >= B_D_IW_MIN and d_mw >= B_D_MW_MIN and thumb_lat <= B_THUMB_LAT_MAX
        )
        if not is_b_match:
            im_straight_ok = (
                (idx_straight >= H_HORIZ_STRAIGHT_MIN and mid_straight >= H_HORIZ_STRAIGHT_MIN
                 and i_curl <= H_HORIZ_CURL_MAX and m_curl <= H_HORIZ_CURL_MAX)
                or (i_ext and m_ext and idx_straight >= H_HORIZ_STRAIGHT_EXT_MIN and mid_straight >= H_HORIZ_STRAIGHT_EXT_MIN)
            )
            rp_curled_ok = (
                (r_curl >= H_HORIZ_R_CURL_MIN and p_curl >= H_HORIZ_P_CURL_MIN)
                or (not r_ext and not p_ext)
            )
            is_h_horiz = (
                d_iw >= H_HORIZ_D_IW_MIN
                and d_mw >= H_HORIZ_D_MW_MIN
                and im_straight_ok
                and rp_curled_ok
                and (d_iw - d_rw) >= H_HORIZ_DIFF_RP_MIN
                and (d_mw - d_rw) >= H_HORIZ_DIFF_RP_MIN
                and (d_iw - d_pw) >= H_HORIZ_DIFF_RP_MIN
                and (d_mw - d_pw) >= H_HORIZ_DIFF_RP_MIN
                and d_rw <= H_HORIZ_D_RW_MAX
                and d_pw <= H_HORIZ_D_PW_MAX
                and idx_horiz_ratio >= H_HORIZ_RATIO_MIN
                and mid_horiz_ratio >= H_HORIZ_RATIO_MIN
                and ang_im_dir <= H_HORIZ_ANG_IM_MAX
                and d_im <= H_HORIZ_D_IM_MAX
                and idx_tip_y <= H_HORIZ_TIP_Y_MAX
                and mid_tip_y <= H_HORIZ_TIP_Y_MAX
            )
            is_h_north = (
                d_iw >= H_NORTH_D_IW_MIN
                and d_rw >= H_NORTH_D_RW_MIN
                and d_pw >= H_NORTH_D_PW_MIN
                and d_mw <= H_NORTH_D_MW_MAX
                and (d_iw - d_mw) >= H_NORTH_DIFF_IW_MW_MIN
                and (d_rw - d_mw) >= H_NORTH_DIFF_RW_MW_MIN
                and (d_pw - d_mw) >= H_NORTH_DIFF_PW_MW_MIN
                and i_curl <= H_NORTH_IRP_CURL_MAX
                and r_curl <= H_NORTH_IRP_CURL_MAX
                and p_curl <= H_NORTH_IRP_CURL_MAX
                and m_curl >= H_NORTH_M_CURL_MIN
                and d_tm <= H_NORTH_D_TM_MAX
            )
            # Biến thể H Miền Nam (QIPEDC D0500N):
            # Trỏ & Giữa duỗi hướng đứng, hông bàn tay hướng camera (mcp_width_2d <= 0.40),
            # ngón cái đè ngón áp út (thumb_ring_dist <= 0.16).
            # Không ràng buộc ang_im_dir (cho phép cả khép lẫn xòe).
            is_h_south_fingers = (
                (i_ext and m_ext and not r_ext and not p_ext)
                or (d_iw >= H_SOUTH_D_IW_MIN and d_mw >= H_SOUTH_D_MW_MIN and not r_ext and not p_ext)
            )
            is_h_south_upright = (
                idx_horiz_ratio <= H_SOUTH_IDX_HORIZ_MAX
                and idx_tip_y <= H_SOUTH_IDX_TIP_Y_MAX
            )
            is_h_south_orientation = (
                mcp_width_2d is not None
                and mcp_width_2d <= H_SOUTH_MCP_WIDTH_MAX
                and thumb_ring_dist is not None
                and thumb_ring_dist <= H_SOUTH_THUMB_RING_MAX
                and (thumb_mid_dist is None or thumb_mid_dist <= H_SOUTH_THUMB_MID_MAX)
            )
            is_h_south = is_h_south_fingers and is_h_south_upright and is_h_south_orientation

            if is_h_horiz or is_h_north or is_h_south:
                if is_h_horiz or is_h_south:
                    # Loại bỏ các phán đoán nhầm V, D, L khi cử chỉ thỏa mãn H ngang hoặc H Miền Nam
                    matches[:] = [m for m in matches if m[0] not in ("V", "D", "L")]

                if is_h_south:
                    sub_variant = "biến thể Miền Nam"
                    reason_detail = f"MCP_width={mcp_width_2d:.2f}<={H_SOUTH_MCP_WIDTH_MAX}, Thumb-Ring={thumb_ring_dist:.2f}<={H_SOUTH_THUMB_RING_MAX}"
                elif is_h_horiz:
                    sub_variant = "2 ngón ngang"
                    reason_detail = f"d_IW={d_iw:.2f}, d_MW={d_mw:.2f}"
                else:
                    sub_variant = "biến thể Miền Bắc"
                    reason_detail = f"d_IW={d_iw:.2f}, d_MW={d_mw:.2f}"

                matches.append((
                    "H",
                    f"Chữ H ({sub_variant}): {reason_detail}",
                    0.95
                ))

        # -------------------------------------------------------------------
        # 13. Chữ K (Trỏ duỗi đứng, giữa duỗi ngang, cái kẹp giữa, áp út & út gập)
        # -------------------------------------------------------------------
        # Đặc biệt: Ngón giữa duỗi ngang nên KHÔNG sử dụng middle_extended (phụ thuộc trục Y).
        # Dùng middle_curl và mid_straight để xác nhận ngón giữa duỗi thẳng,
        # mid_horiz_ratio để xác nhận hướng ngang và ang_im_dir để xác nhận góc mở chữ K.
        is_k_index = (
            d_iw >= K_D_IW_MIN
            and i_curl <= K_IDX_CURL_MAX
            and idx_horiz_ratio <= K_IDX_HORIZ_MAX
            and idx_straight >= K_IDX_STRAIGHT_MIN
        )
        is_k_middle = (
            K_D_MW_MIN <= d_mw <= K_D_MW_MAX
            and m_curl <= K_MID_CURL_MAX
            and mid_straight >= K_MID_STRAIGHT_MIN
            and mid_horiz_ratio >= K_MID_HORIZ_MIN
        )
        is_k_angle = (
            K_ANG_IM_MIN <= ang_im_dir <= K_ANG_IM_MAX
            and d_im >= K_D_IM_MIN
        )
        is_k_curled = (
            d_rw <= K_D_RW_MAX
            and d_pw <= K_D_PW_MAX
            and (d_iw - d_rw) >= K_DIFF_IW_RP_MIN
            and (d_iw - d_pw) >= K_DIFF_IW_RP_MIN
        )
        is_k_thumb = (
            d_ti <= K_D_TI_MAX
            and d_tm <= K_D_TM_MAX
            and (thumb_mid_dist is None or thumb_mid_dist <= K_THUMB_MID_MAX)
        )

        if is_k_index and is_k_middle and is_k_angle and is_k_curled and is_k_thumb:
            # Loại bỏ các phán đoán nhầm V, D, L, G, H nếu có
            matches[:] = [m for m in matches if m[0] not in ("V", "D", "L", "G", "H")]
            matches.append((
                "K",
                f"Chữ K: trỏ đứng d_IW={d_iw:.2f}, giữa ngang d_MW={d_mw:.2f}, ang_IM={ang_im_dir:.1f}°",
                0.95
            ))

        return matches

    def evaluate_rule_h_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Ghi nhận kết quả chẩn đoán từng điều kiện con của Rule V và Rule H (h_horiz & h_north)
        mà không thay đổi cách tính hay kết quả phân loại.
        Nếu đặc trưng đầu vào không tồn tại hoặc là None, trả về None cho điều kiện tương ứng
        thay vì gán giá trị giả.
        """
        if feat is None or not isinstance(feat, dict):
            return None

        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool):
                return None
            try:
                fval = float(val)
                if math.isnan(fval) or math.isinf(fval):
                    return None
                return fval
            except (ValueError, TypeError):
                return None

        def _bool(key: str) -> Optional[bool]:
            val = feat.get(key)
            if val is None:
                return None
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            return None

        def _ge(val: Optional[float], thresh: float) -> Optional[bool]:
            return (val >= thresh) if val is not None else None

        def _le(val: Optional[float], thresh: float) -> Optional[bool]:
            return (val <= thresh) if val is not None else None

        def _diff_ge(a: Optional[float], b: Optional[float], thresh: float) -> Optional[bool]:
            return ((a - b) >= thresh) if (a is not None and b is not None) else None

        def _and(*vals: Optional[bool]) -> Optional[bool]:
            if any(v is False for v in vals):
                return False
            if any(v is None for v in vals):
                return None
            return True

        def _or(*vals: Optional[bool]) -> Optional[bool]:
            if any(v is True for v in vals):
                return True
            if any(v is None for v in vals):
                return None
            return False

        def _not(val: Optional[bool]) -> Optional[bool]:
            return (not val) if val is not None else None

        i_ext = _bool("index_ext")
        m_ext = _bool("middle_ext")
        r_ext = _bool("ring_ext")
        p_ext = _bool("pinky_ext")

        i_curl = _num("index_curl")
        m_curl = _num("middle_curl")
        r_curl = _num("ring_curl")
        p_curl = _num("pinky_curl")

        d_iw = _num("d_IW")
        d_mw = _num("d_MW")
        d_rw = _num("d_RW")
        d_pw = _num("d_PW")
        d_im = _num("d_IM")
        d_tm = _num("d_TM")
        diff_x_im = _num("diff_x_IM")
        thumb_lat = _num("thumb_lat")

        idx_horiz_ratio = _num("idx_horiz_ratio")
        mid_horiz_ratio = _num("mid_horiz_ratio")
        idx_straight = _num("idx_straight")
        mid_straight = _num("mid_straight")
        idx_tip_y = _num("idx_tip_y")
        mid_tip_y = _num("mid_tip_y")
        ang_im_dir = _num("ang_IM_dir")
        mcp_width_2d = _num("mcp_width_2d")
        thumb_ring_dist = _num("thumb_ring_dist")
        thumb_mid_dist = _num("thumb_mid_dist")

        # 1. Kiểm tra is_b_match (guard của Rule H)
        is_b_match = _and(
            i_ext, m_ext, r_ext, p_ext,
            _ge(d_iw, B_D_IW_MIN),
            _ge(d_mw, B_D_MW_MIN),
            _le(thumb_lat, B_THUMB_LAT_MAX),
        )
        not_b_match = _not(is_b_match)

        # 2. Chẩn đoán Rule V (đối chứng trực tiếp khi H bị nhận thành V)
        v_conds = {
            "finger_ext_pattern": _and(i_ext, m_ext, _not(r_ext), _not(p_ext)),
            "d_IW_ge_min": _ge(d_iw, V_D_IW_MIN),
            "d_MW_ge_min": _ge(d_mw, V_D_MW_MIN),
            "d_IM_ge_min": _ge(d_im, V_D_IM_MIN),
            "diff_x_IM_ge_min": _ge(diff_x_im, V_DIFF_X_IM_MIN),
            "ang_IM_dir_ge_min": _ge(ang_im_dir, V_ANG_IM_MIN),
            "idx_horiz_ratio_le_max": _le(idx_horiz_ratio, V_IDX_HORIZ_MAX),
            "idx_tip_y_le_max": _le(idx_tip_y, V_IDX_TIP_Y_MAX),
            "mcp_width_2d_ge_min": _ge(mcp_width_2d, V_MCP_WIDTH_MIN),
            "thumb_ring_dist_ge_min": _ge(thumb_ring_dist, V_THUMB_RING_MIN) if thumb_ring_dist is not None else True,
        }
        v_matched = _and(*v_conds.values())
        v_failed = [k for k, v in v_conds.items() if v is False]

        # 3. Chẩn đoán từng điều kiện của is_h_horiz
        im_straight_by_curl = _and(
            _ge(idx_straight, H_HORIZ_STRAIGHT_MIN),
            _ge(mid_straight, H_HORIZ_STRAIGHT_MIN),
            _le(i_curl, H_HORIZ_CURL_MAX),
            _le(m_curl, H_HORIZ_CURL_MAX),
        )
        im_straight_by_ext = _and(
            i_ext,
            m_ext,
            _ge(idx_straight, H_HORIZ_STRAIGHT_EXT_MIN),
            _ge(mid_straight, H_HORIZ_STRAIGHT_EXT_MIN),
        )
        im_straight_ok = _or(im_straight_by_curl, im_straight_by_ext)

        rp_curled_by_curl = _and(
            _ge(r_curl, H_HORIZ_R_CURL_MIN),
            _ge(p_curl, H_HORIZ_P_CURL_MIN),
        )
        rp_curled_by_ext = _and(_not(r_ext), _not(p_ext))
        rp_curled_ok = _or(rp_curled_by_curl, rp_curled_by_ext)

        h_horiz_primary = {
            "not_b_match": not_b_match,
            "d_IW_ge_min": _ge(d_iw, H_HORIZ_D_IW_MIN),
            "d_MW_ge_min": _ge(d_mw, H_HORIZ_D_MW_MIN),
            "im_straight_ok": im_straight_ok,
            "rp_curled_ok": rp_curled_ok,
            "diff_IW_RW_ge_min": _diff_ge(d_iw, d_rw, H_HORIZ_DIFF_RP_MIN),
            "diff_MW_RW_ge_min": _diff_ge(d_mw, d_rw, H_HORIZ_DIFF_RP_MIN),
            "diff_IW_PW_ge_min": _diff_ge(d_iw, d_pw, H_HORIZ_DIFF_RP_MIN),
            "diff_MW_PW_ge_min": _diff_ge(d_mw, d_pw, H_HORIZ_DIFF_RP_MIN),
            "d_RW_le_max": _le(d_rw, H_HORIZ_D_RW_MAX),
            "d_PW_le_max": _le(d_pw, H_HORIZ_D_PW_MAX),
            "idx_horiz_ratio_ge_min": _ge(idx_horiz_ratio, H_HORIZ_RATIO_MIN),
            "mid_horiz_ratio_ge_min": _ge(mid_horiz_ratio, H_HORIZ_RATIO_MIN),
            "ang_IM_dir_le_max": _le(ang_im_dir, H_HORIZ_ANG_IM_MAX),
            "d_IM_le_max": _le(d_im, H_HORIZ_D_IM_MAX),
            "idx_tip_y_le_max": _le(idx_tip_y, H_HORIZ_TIP_Y_MAX),
            "mid_tip_y_le_max": _le(mid_tip_y, H_HORIZ_TIP_Y_MAX),
        }
        h_horiz_matched = _and(*h_horiz_primary.values())
        h_horiz_failed = [k for k, v in h_horiz_primary.items() if v is False]

        h_horiz_all_conds = dict(h_horiz_primary)
        h_horiz_all_conds["im_straight_by_curl"] = im_straight_by_curl
        h_horiz_all_conds["im_straight_by_ext"] = im_straight_by_ext
        h_horiz_all_conds["rp_curled_by_curl"] = rp_curled_by_curl
        h_horiz_all_conds["rp_curled_by_ext"] = rp_curled_by_ext

        # 4. Chẩn đoán từng điều kiện của is_h_north
        h_north_primary = {
            "not_b_match": not_b_match,
            "d_IW_ge_min": _ge(d_iw, H_NORTH_D_IW_MIN),
            "d_RW_ge_min": _ge(d_rw, H_NORTH_D_RW_MIN),
            "d_PW_ge_min": _ge(d_pw, H_NORTH_D_PW_MIN),
            "d_MW_le_max": _le(d_mw, H_NORTH_D_MW_MAX),
            "diff_IW_MW_ge_min": _diff_ge(d_iw, d_mw, H_NORTH_DIFF_IW_MW_MIN),
            "diff_RW_MW_ge_min": _diff_ge(d_rw, d_mw, H_NORTH_DIFF_RW_MW_MIN),
            "diff_PW_MW_ge_min": _diff_ge(d_pw, d_mw, H_NORTH_DIFF_PW_MW_MIN),
            "index_curl_le_max": _le(i_curl, H_NORTH_IRP_CURL_MAX),
            "ring_curl_le_max": _le(r_curl, H_NORTH_IRP_CURL_MAX),
            "pinky_curl_le_max": _le(p_curl, H_NORTH_IRP_CURL_MAX),
            "middle_curl_ge_min": _ge(m_curl, H_NORTH_M_CURL_MIN),
            "d_TM_le_max": _le(d_tm, H_NORTH_D_TM_MAX),
        }
        h_north_matched = _and(*h_north_primary.values())
        h_north_failed = [k for k, v in h_north_primary.items() if v is False]

        # 5. Chẩn đoán từng điều kiện của is_h_south (QIPEDC D0500N)
        h_south_finger_ext = _and(i_ext, m_ext, _not(r_ext), _not(p_ext))
        h_south_finger_dist = _and(_ge(d_iw, H_SOUTH_D_IW_MIN), _ge(d_mw, H_SOUTH_D_MW_MIN), _not(r_ext), _not(p_ext))
        h_south_fingers = _or(h_south_finger_ext, h_south_finger_dist)

        h_south_primary = {
            "not_b_match": not_b_match,
            "fingers_extended_ok": h_south_fingers,
            "idx_horiz_ratio_le_max": _le(idx_horiz_ratio, H_SOUTH_IDX_HORIZ_MAX),
            "idx_tip_y_le_max": _le(idx_tip_y, H_SOUTH_IDX_TIP_Y_MAX),
            "mcp_width_2d_le_max": _le(mcp_width_2d, H_SOUTH_MCP_WIDTH_MAX),
            "thumb_ring_dist_le_max": _le(thumb_ring_dist, H_SOUTH_THUMB_RING_MAX),
            "thumb_mid_dist_le_max": _le(thumb_mid_dist, H_SOUTH_THUMB_MID_MAX) if thumb_mid_dist is not None else True,
        }
        h_south_matched = _and(*h_south_primary.values())
        h_south_failed = [k for k, v in h_south_primary.items() if v is False]

        return {
            "rule_v": {
                "matched": v_matched,
                "conditions": v_conds,
                "failed_conditions": v_failed,
            },
            "h_horiz": {
                "matched": h_horiz_matched,
                "conditions": h_horiz_all_conds,
                "failed_conditions": h_horiz_failed,
            },
            "h_north": {
                "matched": h_north_matched,
                "conditions": h_north_primary,
                "failed_conditions": h_north_failed,
            },
            "h_south": {
                "matched": h_south_matched,
                "conditions": h_south_primary,
                "failed_conditions": h_south_failed,
            },
        }

    def evaluate_rule_k_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Ghi nhận kết quả chẩn đoán từng điều kiện con của Rule K
        mà không thay đổi cách tính hay kết quả phân loại.
        Nếu đặc trưng đầu vào không tồn tại hoặc là None, trả về None cho điều kiện tương ứng.
        """
        if feat is None or not isinstance(feat, dict):
            return None

        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool):
                return None
            try:
                fval = float(val)
                if math.isnan(fval) or math.isinf(fval):
                    return None
                return fval
            except (ValueError, TypeError):
                return None

        def _ge(val: Optional[float], thresh: float) -> Optional[bool]:
            return (val >= thresh) if val is not None else None

        def _le(val: Optional[float], thresh: float) -> Optional[bool]:
            return (val <= thresh) if val is not None else None

        def _range(val: Optional[float], low: float, high: float) -> Optional[bool]:
            return (low <= val <= high) if val is not None else None

        def _diff_ge(a: Optional[float], b: Optional[float], thresh: float) -> Optional[bool]:
            return ((a - b) >= thresh) if (a is not None and b is not None) else None

        def _and(*vals: Optional[bool]) -> Optional[bool]:
            if any(v is False for v in vals):
                return False
            if any(v is None for v in vals):
                return None
            return True

        d_iw = _num("d_IW")
        d_mw = _num("d_MW")
        d_rw = _num("d_RW")
        d_pw = _num("d_PW")
        d_im = _num("d_IM")
        d_ti = _num("d_TI")
        d_tm = _num("d_TM")
        i_curl = _num("index_curl")
        m_curl = _num("middle_curl")
        idx_horiz = _num("idx_horiz_ratio")
        mid_horiz = _num("mid_horiz_ratio")
        idx_straight = _num("idx_straight")
        mid_straight = _num("mid_straight")
        ang_im = _num("ang_IM_dir")
        thumb_mid = _num("thumb_mid_dist")

        conds = {
            "index_d_iw_ge_min": _ge(d_iw, K_D_IW_MIN),
            "index_curl_le_max": _le(i_curl, K_IDX_CURL_MAX),
            "index_horiz_le_max": _le(idx_horiz, K_IDX_HORIZ_MAX),
            "index_straight_ge_min": _ge(idx_straight, K_IDX_STRAIGHT_MIN),
            "middle_d_mw_in_range": _range(d_mw, K_D_MW_MIN, K_D_MW_MAX),
            "middle_curl_le_max": _le(m_curl, K_MID_CURL_MAX),
            "middle_straight_ge_min": _ge(mid_straight, K_MID_STRAIGHT_MIN),
            "middle_horiz_ge_min": _ge(mid_horiz, K_MID_HORIZ_MIN),
            "ang_im_in_range": _range(ang_im, K_ANG_IM_MIN, K_ANG_IM_MAX),
            "d_im_ge_min": _ge(d_im, K_D_IM_MIN),
            "ring_d_rw_le_max": _le(d_rw, K_D_RW_MAX),
            "pinky_d_pw_le_max": _le(d_pw, K_D_PW_MAX),
            "diff_iw_rw_ge_min": _diff_ge(d_iw, d_rw, K_DIFF_IW_RP_MIN),
            "diff_iw_pw_ge_min": _diff_ge(d_iw, d_pw, K_DIFF_IW_RP_MIN),
            "thumb_d_ti_le_max": _le(d_ti, K_D_TI_MAX),
            "thumb_d_tm_le_max": _le(d_tm, K_D_TM_MAX),
            "thumb_mid_dist_le_max": (_le(thumb_mid, K_THUMB_MID_MAX) if thumb_mid is not None else True),
        }

        matched = _and(*conds.values())
        failed = [k for k, v in conds.items() if v is False]

        return {
            "matched": matched,
            "conditions": conds,
            "failed_conditions": failed,
        }

    def evaluate_rule_i_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Ghi nhận kết quả chẩn đoán từng điều kiện con của Rule I và Rule Y
        mà không thay đổi cách tính hay kết quả phân loại.
        Nếu đặc trưng đầu vào không tồn tại hoặc là None, trả về None cho điều kiện tương ứng.
        """
        if feat is None or not isinstance(feat, dict):
            return None

        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool):
                return None
            try:
                fval = float(val)
                if math.isnan(fval) or math.isinf(fval):
                    return None
                return fval
            except (ValueError, TypeError):
                return None

        def _bool(key: str) -> Optional[bool]:
            val = feat.get(key)
            if val is None:
                return None
            return bool(val)

        p_ext = _bool("pinky_ext")
        i_ext = _bool("index_ext")
        m_ext = _bool("middle_ext")
        r_ext = _bool("ring_ext")

        d_pw = _num("d_PW")
        d_iw = _num("d_IW")
        d_mw = _num("d_MW")
        d_rw = _num("d_RW")
        d_tw = _num("d_TW")
        d_ti = _num("d_TI")
        d_tm = _num("d_TM")
        thumb_lat = _num("thumb_lat")
        ang_tp = _num("ang_TP")
        p_curl = _num("pinky_curl")

        def _ge(v: Optional[float], thresh: float) -> Optional[bool]:
            return None if v is None else (v >= thresh)

        def _le(v: Optional[float], thresh: float) -> Optional[bool]:
            return None if v is None else (v <= thresh)

        def _not(v: Optional[bool]) -> Optional[bool]:
            return None if v is None else (not v)

        def _and(*args: Optional[bool]) -> Optional[bool]:
            if any(a is False for a in args):
                return False
            if any(a is None for a in args):
                return None
            return True

        def _or(*args: Optional[bool]) -> Optional[bool]:
            if any(a is True for a in args):
                return True
            if any(a is None for a in args):
                return None
            return False

        max_curled = None
        if d_iw is not None and d_mw is not None and d_rw is not None:
            max_curled = max(d_iw, d_mw, d_rw)
        diff_pw_curled = None
        if d_pw is not None and max_curled is not None:
            diff_pw_curled = d_pw - max_curled

        conds = {
            "pinky_ext_or_straight": _or(p_ext, _le(p_curl, 0.12)),
            "d_PW_ge_min": _ge(d_pw, I_D_PW_MIN),
            "pinky_curl_le_max": _le(p_curl, I_PINKY_CURL_MAX),
            "middle_fingers_folded": _and(_not(i_ext), _not(m_ext), _not(r_ext)),
            "d_IW_le_max": _le(d_iw, I_D_IW_MAX),
            "d_MW_le_max": _le(d_mw, I_D_MW_MAX),
            "d_RW_le_max": _le(d_rw, I_D_RW_MAX),
            "diff_PW_curled_ge_min": _ge(diff_pw_curled, I_DIFF_PW_FOLDED_MIN),
            "d_TI_le_max": _le(d_ti, I_D_TI_MAX),
            "d_TM_le_max": _le(d_tm, I_D_TM_MAX),
            "d_TW_le_max": _le(d_tw, I_D_TW_MAX),
            "thumb_not_spread": _or(_le(thumb_lat, I_THUMB_LAT_MAX), _le(ang_tp, I_ANG_TP_MAX)),
        }
        matched = _and(*conds.values())
        failed = [k for k, v in conds.items() if v is False]

        return {
            "matched": matched,
            "conditions": conds,
            "failed_conditions": failed,
        }

    def classify_detailed(self, landmarks: Any) -> Dict[str, Any]:
        """
        Phân loại chi tiết và minh bạch.
        Trả về dictionary gồm:
        {
            "letter": "A".."Y" | "UNKNOWN",
            "confidence": float,
            "status": "MATCH" | "NO_MATCH" | "CONFLICT" | "INVALID_INPUT",
            "reason": str,
            "matched_candidates": list,
            "features": dict or None
        }
        """
        is_valid, err_msg = validate_input_landmarks(landmarks)
        if not is_valid:
            return {
                "letter": "UNKNOWN",
                "confidence": 0.0,
                "status": "INVALID_INPUT",
                "reason": f"Đầu vào không hợp lệ: {err_msg}",
                "matched_candidates": [],
                "features": None
            }

        features = self.extract_features(landmarks)
        if features is None:
            return {
                "letter": "UNKNOWN",
                "confidence": 0.0,
                "status": "INVALID_INPUT",
                "reason": "Dữ liệu mốc suy biến (palm_scale < 1e-5)",
                "matched_candidates": [],
                "features": None
            }

        matches = self.evaluate_rules(features)

        if len(matches) == 1:
            letter, reason, conf = matches[0]
            return {
                "letter": letter,
                "confidence": conf,
                "status": "MATCH",
                "reason": reason,
                "matched_candidates": [(letter, conf)],
                "features": features
            }
        elif len(matches) > 1:
            candidates = [m[0] for m in matches]
            return {
                "letter": "UNKNOWN",
                "confidence": 0.0,
                "status": "CONFLICT",
                "reason": f"Xung đột giữa nhiều quy tắc: {', '.join(candidates)}",
                "matched_candidates": [(m[0], m[2]) for m in matches],
                "features": features
            }
        else:
            return {
                "letter": "UNKNOWN",
                "confidence": 0.0,
                "status": "NO_MATCH",
                "reason": "Không khớp với bất kỳ quy tắc hình học chữ cái nào đã biết",
                "matched_candidates": [],
                "features": features
            }

    def classify(self, landmarks: Any) -> Tuple[str, float]:
        """
        Hàm phân loại nhanh trả về tuple (letter, confidence).
        Nếu không khớp hoặc xung đột/lỗi, trả về ("UNKNOWN", 0.0).
        """
        res = self.classify_detailed(landmarks)
        if res["status"] == "MATCH":
            return res["letter"], res["confidence"]
        return "UNKNOWN", 0.0


# Alias tương thích
AlphabetClassifier = VSLAlphabetClassifier

# Khởi tạo singleton instance tiện dụng
alphabet_classifier = VSLAlphabetClassifier()


def classify_vsl_alphabet(landmarks: Any) -> Tuple[str, float]:
    """Hàm tiện ích cấp module để phân loại cử chỉ chữ cái nhanh."""
    return alphabet_classifier.classify(landmarks)

