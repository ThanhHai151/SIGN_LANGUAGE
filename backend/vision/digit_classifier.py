"""
Bộ phân loại hình học Chữ số VSL độc lập (VSL Digit Classifier).
==================================================================
Phân loại hình dạng bàn tay cho các chữ số từ 1 đến 9 dựa trên 21 mốc landmark MediaPipe.

Đặc điểm kỹ thuật:
- Độc lập: Không phụ thuộc vào recognizer.py hay luồng nhận diện chính.
- An toàn: Kiểm tra nghiêm ngặt đầu vào (None, rỗng, sai số điểm, NaN/Inf).
- Minh bạch: Đánh giá độc lập các điều kiện của từng chữ số, không dùng thác lệnh if...elif
  để âm thầm ép nhãn. Phát hiện và báo rõ các trường hợp không khớp (NEUTRAL) hoặc xung đột (CONFLICT).
- Ngưỡng: Thiết lập dựa trên kết quả khảo sát tập DEV của video QIPEDC thực tế (Giai đoạn 2C.3C).
- Số 0: Đã triển khai dựa trên landmark thực tế từ video QIPEDC D0529 (Giai đoạn 2C.8).
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
    # Fallback import nếu chạy standalone
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from backend.vision.feature_extractor import (
        calculate_distance,
        calculate_angle,
        extract_dual_hand_features
    )


# ---------------------------------------------------------------------------
# Các hằng số ngưỡng chuẩn hóa từ tập DEV
# ---------------------------------------------------------------------------

THUMB_LATERAL_THRESHOLD = 0.30    # Phân biệt ngón cái khép (<=0.30) vs xòe (>0.30)
ANGLE_TI_THRESHOLD = 25.0         # Góc Cái–Trỏ: số 1 (<25°) vs số 7 (>=25°)
ANGLE_TP_THRESHOLD_8 = 25.0       # Góc Cái–Út: số 2 (<25°) vs số 8 (>=25°)
ANGLE_TP_THRESHOLD_5_6 = 35.0     # Góc Cái–Út: số 4 (<35°) vs số 5 (>=35°), và số 6 (>=35°)
INDEX_CURL_MIN_9 = 0.08           # Ngưỡng dưới tỷ lệ co gập ngón trỏ số 9 (khum móc)
INDEX_CURL_MAX_9 = 0.35           # Ngưỡng trên tỷ lệ co gập ngón trỏ số 9
D_TI_THRESHOLD_0 = 0.20           # Khoảng cách chuẩn hóa Cái–Trỏ cho số 0
D_TM_THRESHOLD_0 = 0.30           # Khoảng cách chuẩn hóa Cái–Giữa cho số 0
D_IW_MIN_0 = 0.95                 # Khoảng cách tối thiểu Trỏ–Cổ tay để tránh nắm đấm
INDEX_CURL_MIN_0 = 0.45           # Ngưỡng co gập tối thiểu cho số 0
INDEX_CURL_MAX_0 = 0.80           # Ngưỡng co gập tối đa cho số 0


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


class VSLDigitClassifier:
    """Bộ phân loại hình học độc lập cho chữ số VSL 1–9."""

    def __init__(self):
        pass

    def extract_features(self, landmarks: List[List[float]]) -> Optional[Dict[str, Any]]:
        """
        Trích xuất các đặc trưng hình học chuẩn hóa cần thiết từ 21 điểm mốc.
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

        # Các góc và khoảng cách chuẩn hóa
        ang_ti = float(calculate_angle(sh[4], sh[0], sh[8]))
        ang_tp = float(calculate_angle(sh[4], sh[0], sh[20]))
        thumb_lat = float(abs(sh[4][0] - sh[0][0]) / palm_scale)
        d_ti = float(calculate_distance(sh[4], sh[8]) / palm_scale)
        d_tm = float(calculate_distance(sh[4], sh[12]) / palm_scale)
        d_iw = float(calculate_distance(sh[0], sh[8]) / palm_scale)

        return {
            "palm_scale": round(palm_scale, 4),
            "thumb_ext": t_ext,
            "index_ext": i_ext,
            "middle_ext": m_ext,
            "ring_ext": r_ext,
            "pinky_ext": p_ext,
            "index_curl": round(i_curl, 4),
            "ang_TI": round(ang_ti, 2),
            "ang_TP": round(ang_tp, 2),
            "thumb_lat": round(thumb_lat, 4),
            "d_TI": round(d_ti, 4),
            "d_TM": round(d_tm, 4),
            "d_IW": round(d_iw, 4)
        }

    def evaluate_rules(self, feat: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """
        Đánh giá độc lập 9 quy tắc hình học cho các số từ 1 đến 9.
        Trả về danh sách các bộ (digit, reason, confidence) thỏa mãn điều kiện.
        """
        matches = []

        t_ext = feat["thumb_ext"]
        i_ext = feat["index_ext"]
        m_ext = feat["middle_ext"]
        r_ext = feat["ring_ext"]
        p_ext = feat["pinky_ext"]
        i_curl = feat["index_curl"]
        ang_ti = feat["ang_TI"]
        ang_tp = feat["ang_TP"]
        thumb_lat = feat["thumb_lat"]

        # --- QUY TẮC SỐ 0 (Vòng khuyên tròn - Zero) ---
        # Điều kiện: 4 ngón trỏ, giữa, áp út, út đều không duỗi thẳng;
        # Đầu ngón trỏ và ngón giữa chụm chạm sát đầu ngón cái (d_TI, d_TM);
        # Lòng bàn tay mở vòm rỗng (d_IW >= D_IW_MIN_0 để phân biệt với nắm đấm).
        if not i_ext and not m_ext and not r_ext and not p_ext:
            d_ti = feat.get("d_TI", 1.0)
            d_tm = feat.get("d_TM", 1.0)
            d_iw = feat.get("d_IW", 0.0)
            if d_ti <= D_TI_THRESHOLD_0 and d_tm <= D_TM_THRESHOLD_0 and d_iw >= D_IW_MIN_0:
                if INDEX_CURL_MIN_0 <= i_curl <= INDEX_CURL_MAX_0:
                    matches.append(("0", f"Vòng tròn số 0: d_TI={d_ti:.3f}<={D_TI_THRESHOLD_0}, d_TM={d_tm:.3f}<={D_TM_THRESHOLD_0}, d_IW={d_iw:.3f}>={D_IW_MIN_0}", 0.95))

        # --- QUY TẮC SỐ 6 (Shaka / Alo) ---
        # Điều kiện: Ngón cái và ngón út duỗi, 3 ngón giữa (trỏ, giữa, áp út) gập
        if t_ext and p_ext and not i_ext and not m_ext and not r_ext:
            if ang_tp >= ANGLE_TP_THRESHOLD_5_6:
                matches.append(("6", f"Cử chỉ Shaka: ang_TP={ang_tp:.1f}>={ANGLE_TP_THRESHOLD_5_6}", 0.95))

        # --- QUY TẮC SỐ 5 (5 ngón xòe rộng) ---
        # Điều kiện: 4 ngón trỏ, giữa, áp út, út duỗi VÀ ngón cái xòe rộng
        if i_ext and m_ext and r_ext and p_ext:
            if thumb_lat > THUMB_LATERAL_THRESHOLD and ang_tp >= ANGLE_TP_THRESHOLD_5_6:
                matches.append(("5", f"5 ngón xòe: thumb_lat={thumb_lat:.2f}>{THUMB_LATERAL_THRESHOLD}, ang_TP={ang_tp:.1f}>={ANGLE_TP_THRESHOLD_5_6}", 0.95))

        # --- QUY TẮC SỐ 4 (4 ngón duỗi, ngón cái khép) ---
        # Điều kiện: 4 ngón trỏ, giữa, áp út, út duỗi VÀ ngón cái khép/gập qua lòng bàn tay
        if i_ext and m_ext and r_ext and p_ext:
            if thumb_lat <= THUMB_LATERAL_THRESHOLD or ang_tp < ANGLE_TP_THRESHOLD_5_6:
                matches.append(("4", f"4 ngón duỗi, cái khép: thumb_lat={thumb_lat:.2f}<={THUMB_LATERAL_THRESHOLD}", 0.92))

        # --- QUY TẮC SỐ 3 (3 ngón giữa duỗi, ngón út gập) ---
        # Điều kiện: Trỏ, giữa, áp út duỗi; ngón út gập
        if i_ext and m_ext and r_ext and not p_ext:
            matches.append(("3", "3 ngón duỗi (Trỏ, Giữa, Áp út), Út gập", 0.95))

        # --- QUY TẮC SỐ 8 (3 ngón Cái + Trỏ + Giữa mở rộng) ---
        # Điều kiện: Trỏ và giữa duỗi, áp út và út gập VÀ ngón cái xòe rộng
        if i_ext and m_ext and not r_ext and not p_ext:
            if thumb_lat > THUMB_LATERAL_THRESHOLD and ang_tp >= ANGLE_TP_THRESHOLD_8:
                matches.append(("8", f"3 ngón Cái+Trỏ+Giữa mở: thumb_lat={thumb_lat:.2f}>{THUMB_LATERAL_THRESHOLD}, ang_TP={ang_tp:.1f}>={ANGLE_TP_THRESHOLD_8}", 0.92))

        # --- QUY TẮC SỐ 2 (Ngón trỏ & giữa chữ V, ngón cái khép) ---
        # Điều kiện: Trỏ và giữa duỗi, áp út và út gập VÀ ngón cái khép
        if i_ext and m_ext and not r_ext and not p_ext:
            if thumb_lat <= THUMB_LATERAL_THRESHOLD and ang_tp < ANGLE_TP_THRESHOLD_8:
                matches.append(("2", f"Trỏ+Giữa chữ V, cái khép: thumb_lat={thumb_lat:.2f}<={THUMB_LATERAL_THRESHOLD}", 0.92))

        # --- QUY TẮC SỐ 9 (Ngón trỏ khum móc câu) ---
        # Điều kiện: Giữa, áp út, út gập VÀ ngón trỏ cong khum (hooked)
        if not m_ext and not r_ext and not p_ext:
            if INDEX_CURL_MIN_9 <= i_curl <= INDEX_CURL_MAX_9 and ang_ti < ANGLE_TI_THRESHOLD:
                matches.append(("9", f"Ngón trỏ khum móc câu: i_curl={i_curl:.2f} in [{INDEX_CURL_MIN_9}, {INDEX_CURL_MAX_9}]", 0.90))

        # --- QUY TẮC SỐ 7 (Ngón trỏ + Cái mở rộng chữ L) ---
        # Điều kiện: Trỏ duỗi; Giữa, áp út, út gập VÀ ngón cái mở rộng L
        if i_ext and not m_ext and not r_ext and not p_ext:
            if thumb_lat > THUMB_LATERAL_THRESHOLD and ang_ti >= ANGLE_TI_THRESHOLD:
                matches.append(("7", f"Trỏ+Cái mở rộng L: thumb_lat={thumb_lat:.2f}>{THUMB_LATERAL_THRESHOLD}, ang_TI={ang_ti:.1f}>={ANGLE_TI_THRESHOLD}", 0.90))

        # --- QUY TẮC SỐ 1 (Ngón trỏ duỗi đơn, ngón cái khép) ---
        # Điều kiện: Trỏ duỗi; Giữa, áp út, út gập VÀ ngón cái khép sát cạnh bàn tay
        if i_ext and not m_ext and not r_ext and not p_ext:
            if thumb_lat <= THUMB_LATERAL_THRESHOLD and ang_ti < ANGLE_TI_THRESHOLD:
                matches.append(("1", f"Ngón trỏ duỗi đơn, cái khép: thumb_lat={thumb_lat:.2f}<={THUMB_LATERAL_THRESHOLD}", 0.92))

        return matches

    def classify_detailed(self, landmarks: Any) -> Dict[str, Any]:
        """
        Phân loại chi tiết và minh bạch.
        Trả về dictionary gồm:
        {
            "digit": "1".."9" | "NEUTRAL" | "CONFLICT",
            "confidence": float,
            "status": "MATCH" | "NO_MATCH" | "CONFLICT" | "INVALID_INPUT",
            "reason": str,
            "matched_digits": list,
            "features": dict or None
        }
        """
        is_valid, err_msg = validate_input_landmarks(landmarks)
        if not is_valid:
            return {
                "digit": "NEUTRAL",
                "confidence": 0.0,
                "status": "INVALID_INPUT",
                "reason": f"Đầu vào không hợp lệ: {err_msg}",
                "matched_digits": [],
                "features": None
            }

        features = self.extract_features(landmarks)
        if features is None:
            return {
                "digit": "NEUTRAL",
                "confidence": 0.0,
                "status": "INVALID_INPUT",
                "reason": "Dữ liệu mốc suy biến (palm_scale < 1e-5)",
                "matched_digits": [],
                "features": None
            }

        matches = self.evaluate_rules(features)

        if len(matches) == 1:
            digit, reason, conf = matches[0]
            return {
                "digit": digit,
                "confidence": conf,
                "status": "MATCH",
                "reason": reason,
                "matched_digits": [digit],
                "features": features
            }

        if len(matches) == 0:
            return {
                "digit": "NEUTRAL",
                "confidence": 0.0,
                "status": "NO_MATCH",
                "reason": "Không khớp bất kỳ quy tắc chữ số nào",
                "matched_digits": [],
                "features": features
            }

        # len(matches) > 1: Xung đột nhiều chữ số
        matched_digits = [m[0] for m in matches]
        conflict_reasons = "; ".join([f"Số {m[0]}: {m[1]}" for m in matches])
        return {
            "digit": "CONFLICT",
            "confidence": 0.0,
            "status": "CONFLICT",
            "reason": f"Xung đột nhiều nhãn ({', '.join(matched_digits)}): {conflict_reasons}",
            "matched_digits": matched_digits,
            "features": features
        }

    def classify(self, landmarks: Any) -> Tuple[str, float]:
        """
        Giao diện nhanh tương thích (tuple: sign, confidence).
        Trả về ("1".."9", confidence) nếu khớp duy nhất; ngược lại trả về ("NEUTRAL", 0.0).
        """
        res = self.classify_detailed(landmarks)
        if res["status"] == "MATCH":
            return res["digit"], res["confidence"]
        return "NEUTRAL", 0.0


# Khởi tạo singleton instance tiện dụng
digit_classifier = VSLDigitClassifier()


def classify_vsl_digit(landmarks: Any) -> Tuple[str, float]:
    """Hàm tiện ích cấp module để phân loại cử chỉ số nhanh."""
    return digit_classifier.classify(landmarks)

