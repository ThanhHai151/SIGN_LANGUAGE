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
I_D_IW_MAX = 0.85
I_D_MW_MAX = 0.75
Y_D_TW_MIN = 1.05

# 3. Chữ V (Ngón trỏ & giữa xòe chữ V)
V_D_IW_MIN = 1.70
V_D_MW_MIN = 1.70
V_D_IM_MIN = 0.28
V_DIFF_X_IM_MIN = 0.18

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
E_D_TI_MAX = 0.35
E_D_TM_MAX = 0.38
E_D_TW_MAX = 0.95
E_D_IW_MIN = 1.05
E_D_MW_MIN = 1.05


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
        d_pw = float(calculate_distance(sh[20], sh[0]) / palm_scale)

        diff_x_im = float(abs(sh[8][0] - sh[12][0]) / palm_scale)

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
            "d_PW": round(d_pw, 4),
            "diff_x_IM": round(diff_x_im, 4)
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
        # Chữ I: Duy nhất ngón út duỗi thẳng, 4 ngón còn lại gập
        if p_ext and not i_ext and not m_ext and not r_ext and not t_ext:
            if d_pw >= I_D_PW_MIN and d_iw <= I_D_IW_MAX and d_mw <= I_D_MW_MAX:
                matches.append(("I", f"Chỉ ngón út duỗi thẳng: d_PW={d_pw:.2f}>={I_D_PW_MIN}", 0.95))

        # Chữ Y: Ngón cái và ngón út cùng duỗi thẳng (Shaka), 3 ngón giữa gập
        if p_ext and not i_ext and not m_ext and not r_ext:
            if d_pw >= I_D_PW_MIN and d_tw >= Y_D_TW_MIN and d_iw <= I_D_IW_MAX and d_mw <= I_D_MW_MAX:
                matches.append(("Y", f"Ngón cái và út cùng duỗi thẳng (Shaka): d_PW={d_pw:.2f}>={I_D_PW_MIN}, d_TW={d_tw:.2f}>={Y_D_TW_MIN}", 0.95))

        # -------------------------------------------------------------------
        # 3. Chữ V (Trỏ & Giữa xòe hình chữ V)
        # -------------------------------------------------------------------
        if i_ext and m_ext and not r_ext and not p_ext:
            if d_iw >= V_D_IW_MIN and d_mw >= V_D_MW_MIN and d_im >= V_D_IM_MIN and diff_x_im >= V_DIFF_X_IM_MIN:
                matches.append(("V", f"Trỏ & Giữa xòe chữ V: d_IM={d_im:.2f}>={V_D_IM_MIN}, diff_x={diff_x_im:.2f}>={V_DIFF_X_IM_MIN}", 0.95))

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
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if (d_ti <= E_D_TI_MAX and d_tm <= E_D_TM_MAX and d_tw <= E_D_TW_MAX and 
                d_iw >= E_D_IW_MIN and d_mw >= E_D_MW_MIN):
                matches.append(("E", f"4 ngón tì gập lên ngón cái: d_TM={d_tm:.2f}<={E_D_TM_MAX}, d_TW={d_tw:.2f}<={E_D_TW_MAX}", 0.95))

        return matches

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

