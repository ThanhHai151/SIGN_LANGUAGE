"""
Kiểm thử Hình học (Geometric Tests) cho các Chữ số VSL 1–9.
==========================================================
Giai đoạn 2C.3C: Khảo sát đặc trưng, đề xuất quy tắc ứng viên trên tập DEV
và đánh giá độc lập trên tập EVAL dựa trên mốc Landmark MediaPipe Hands thực tế.

Ràng buộc tuân thủ:
- Dữ liệu nguồn: real_digit_landmarks.json (18 mẫu mốc thực tế từ video QIPEDC).
- Không sửa classify_geometry() hay recognizer.py trong giai đoạn này.
- Quy tắc và ngưỡng được thiết kế 100% dựa trên tập Dev, đóng băng trước khi đánh giá Eval.
- Khảo sát xung đột thực nghiệm với các chữ cái VSL tương đồng (D, V, U, L, Y, CAM_ON).
"""

import os
import sys
import json
import pytest
from typing import Tuple, Dict, Any

# Đảm bảo import được module của backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.vision.feature_extractor import (
    calculate_distance,
    calculate_angle,
    extract_dual_hand_features
)
from backend.vision.recognizer import VSLRecognizer
from backend.training.extract_video_landmarks import validate_landmark_format


# ---------------------------------------------------------------------------
# Helper: Bộ phân loại quy tắc ứng viên chữ số (Đề xuất dựa trên tập DEV)
# ---------------------------------------------------------------------------

def classify_digit_candidate_rules(landmarks: Any) -> Tuple[str, str]:
    """
    Quy tắc ứng viên phân loại số 1–9 được đề xuất dựa trên khảo sát tập DEV.
    
    Các đặc trưng cốt lõi:
    - primary_states (trạng thái duỗi/gập ngón từ extract_dual_hand_features).
    - thumb_lat: Độ dang ngang của ngón cái tương đối với cổ tay (chuẩn hóa theo palm_scale).
    - ang_ti: Góc giữa đầu ngón cái (4) - cổ tay (0) - đầu ngón trỏ (8).
    - ang_tp: Góc giữa đầu ngón cái (4) - cổ tay (0) - đầu ngón út (20).
    - index_curl: Tỷ lệ co gập ngón trỏ (nhận diện ngón trỏ khum móc câu của Số 9).
    
    Trả về: (predicted_digit, reason_string)
    """
    if not landmarks or len(landmarks) < 21:
        return "NEUTRAL", "Thiếu điểm landmark"

    _, meta = extract_dual_hand_features(landmarks, None)
    st = meta.get("primary_states", {})

    p = landmarks
    wrist = p[0]
    sh = [[p[i][0] - wrist[0], p[i][1] - wrist[1], p[i][2] - wrist[2]] for i in range(21)]
    palm_scale = calculate_distance(sh[0], sh[9])
    if palm_scale < 1e-5:
        return "NEUTRAL", "Palm scale bằng 0"

    t_ext = st.get("thumb_extended", False)
    i_ext = st.get("index_extended", False)
    m_ext = st.get("middle_extended", False)
    r_ext = st.get("ring_extended", False)
    p_ext = st.get("pinky_extended", False)

    i_curl = st.get("index_curl", 0.5)

    ang_ti = calculate_angle(sh[4], sh[0], sh[8])
    ang_tp = calculate_angle(sh[4], sh[0], sh[20])
    thumb_lat = abs(sh[4][0] - sh[0][0]) / palm_scale

    # 1. Số 6: Cử chỉ Shaka / Alo (Ngón cái + Ngón út duỗi, 3 ngón giữa gập)
    if t_ext and p_ext and not i_ext and not m_ext and not r_ext:
        if ang_tp >= 35.0:
            return "6", f"Shaka (Cái+Út mở rộng): ang_TP={ang_tp:.1f}>=35"

    # 2. Số 5 vs Số 4: 4 ngón (Trỏ, Giữa, Áp út, Út) đều duỗi
    if i_ext and m_ext and r_ext and p_ext:
        # Số 5: Ngón cái xòe rộng ngang sang bên
        if thumb_lat > 0.30 and ang_tp >= 35.0:
            return "5", f"5 ngón xòe rộng: thumb_lat={thumb_lat:.2f}>0.3, ang_TP={ang_tp:.1f}>=35"
        # Số 4: Ngón cái khép gập qua lòng bàn tay
        else:
            return "4", f"4 ngón duỗi, cái khép: thumb_lat={thumb_lat:.2f}<=0.3"

    # 3. Số 3: Trỏ, Giữa, Áp út duỗi; Ngón út gập
    if i_ext and m_ext and r_ext and not p_ext:
        return "3", "3 ngón giữa duỗi (Trỏ, Giữa, Áp út), Út gập"

    # 4. Số 8 vs Số 2: Trỏ + Giữa duỗi (hình chữ V), Áp út + Út gập
    if i_ext and m_ext and not r_ext and not p_ext:
        # Số 8: Ngón cái mở rộng sang bên (3 ngón Cái + Trỏ + Giữa)
        if thumb_lat > 0.30 and ang_tp >= 25.0:
            return "8", f"3 ngón Cái+Trỏ+Giữa mở: thumb_lat={thumb_lat:.2f}>0.3, ang_TP={ang_tp:.1f}>=25"
        # Số 2: Ngón cái khép gập vào lòng bàn tay
        else:
            return "2", f"Trỏ+Giữa chữ V, cái khép: thumb_lat={thumb_lat:.2f}<=0.3"

    # 5. Số 7 vs Số 1 vs Số 9: Áp út và Út gập, Giữa gập
    if not m_ext and not r_ext and not p_ext:
        # Số 9: Ngón trỏ cong khum (hooked finger: i_curl nằm giữa duỗi và gập)
        if 0.08 <= i_curl <= 0.35 and ang_ti < 25.0:
            return "9", f"Ngón trỏ cong khum (hooked): i_curl={i_curl:.2f} in [0.08, 0.35]"

        # Số 7 vs Số 1: Ngón trỏ duỗi thẳng
        if i_ext:
            # Số 7: Ngón cái mở rộng tạo góc súng / chữ L (nhấp tay)
            if thumb_lat > 0.30 and ang_ti >= 25.0:
                return "7", f"Trỏ+Cái mở rộng hình chữ L: thumb_lat={thumb_lat:.2f}>0.3, ang_TI={ang_ti:.1f}>=25"
            # Số 1: Ngón cái khép sát cạnh bàn tay
            else:
                return "1", f"Ngón trỏ duỗi đơn, cái khép: thumb_lat={thumb_lat:.2f}<=0.3"

    return "NEUTRAL", "Không khớp quy tắc chữ số nào"


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_data():
    """Tải tệp fixture real_digit_landmarks.json."""
    fpath = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json")
    assert os.path.exists(fpath), f"Không tìm thấy fixture {fpath}"
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def registry_data():
    """Tải tệp fixture digit_video_registry.json."""
    fpath = os.path.join(os.path.dirname(__file__), "fixtures", "digit_video_registry.json")
    assert os.path.exists(fpath), f"Không tìm thấy registry {fpath}"
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


class TestFixtureValidation:
    """Xác thực tính toàn vẹn và an toàn của các tệp fixture."""

    def test_fixture_schema_and_counts(self, fixture_data, registry_data):
        assert fixture_data["schema_version"] == "1.0"
        assert fixture_data["total_samples"] == 18
        assert fixture_data["successful_samples"] == 18

        samples = fixture_data["samples"]
        assert len(samples) == 18

        dev_samples = [s for s in samples if s["split"] == "dev"]
        eval_samples = [s for s in samples if s["split"] == "eval"]
        assert len(dev_samples) == 9, "Tập dev phải có đúng 9 mẫu (số 1-9)"
        assert len(eval_samples) == 9, "Tập eval phải có đúng 9 mẫu (số 1-9)"

        # Kiểm tra metadata khớp registry
        for s in samples:
            d = s["digit"]
            reg_entry = registry_data["digits"][d]
            assert s["video_id"] == reg_entry["source_video_id"]
            assert s["source_video_file"] == reg_entry["source_video_file"]
            assert s["quality_status"] == "ok"
            assert s["confidence"] >= 0.90
            assert s["handedness"] == "Right"
            assert s["coordinate_type"] == "normalized_mediapipe_hands"

            # Kiểm tra định dạng 21 mốc landmark
            is_valid, msg = validate_landmark_format(s["landmarks"])
            assert is_valid, f"Số {d} ({s['split']}) landmark không hợp lệ: {msg}"

    def test_digit_0_status_uncompromised(self, registry_data):
        """Xác minh số 0 không bị gắn nhãn giả lập và giữ nguyên trạng thái pending."""
        reg_zero = registry_data["digits"]["0"]
        assert reg_zero["has_ground_truth"] is False
        assert reg_zero["file_exists_locally"] is False
        assert reg_zero["status"] == "CHƯA CÓ GROUND TRUTH"


class TestDevGeometricAnalysis:
    """Khảo sát đặc trưng và kiểm tra quy tắc ứng viên trên tập DEV."""

    @pytest.mark.parametrize("target_digit", [str(i) for i in range(1, 10)])
    def test_dev_candidate_rules_classify_correctly(self, fixture_data, target_digit):
        dev_sample = next(
            s for s in fixture_data["samples"]
            if s["digit"] == target_digit and s["split"] == "dev"
        )
        pred, reason = classify_digit_candidate_rules(dev_sample["landmarks"])
        assert pred == target_digit, (
            f"Mẫu DEV số {target_digit} bị phân loại sai thành {pred}. Lý do: {reason}"
        )


class TestEvalGeometricVerification:
    """Đánh giá độc lập quy tắc ứng viên trên tập EVAL (nghiêm cấm tinh chỉnh ngưỡng)."""

    @pytest.mark.parametrize("target_digit", [str(i) for i in range(1, 10)])
    def test_eval_samples_independent_verification(self, fixture_data, target_digit):
        eval_sample = next(
            s for s in fixture_data["samples"]
            if s["digit"] == target_digit and s["split"] == "eval"
        )
        pred, reason = classify_digit_candidate_rules(eval_sample["landmarks"])
        assert pred == target_digit, (
            f"Mẫu EVAL số {target_digit} bị phân loại sai thành {pred}. Lý do: {reason}"
        )


class TestLetterConflictEmpiricalEvidence:
    """
    Khảo sát xung đột thực nghiệm giữa chữ số và chữ cái.
    Đưa landmark chữ số thật vào bộ phân loại classify_geometry() hiện tại để chứng minh xung đột.
    """

    def test_empirical_conflict_with_current_alphabet_rules(self, fixture_data):
        rec = VSLRecognizer()

        # 1. Số 1 xung đột trực tiếp với Chữ D
        s1 = next(s for s in fixture_data["samples"] if s["digit"] == "1" and s["split"] == "dev")
        _, meta1 = extract_dual_hand_features(s1["landmarks"], None)
        sign1, conf1 = rec.classify_geometry(s1["landmarks"], meta1["primary_states"])
        assert sign1 == "D", f"Kỳ vọng Số 1 kích hoạt Chữ D trong bộ phân loại cũ, nhận {sign1}"

        # 2. Số 2 xung đột với Chữ U / V
        s2 = next(s for s in fixture_data["samples"] if s["digit"] == "2" and s["split"] == "dev")
        _, meta2 = extract_dual_hand_features(s2["landmarks"], None)
        sign2, conf2 = rec.classify_geometry(s2["landmarks"], meta2["primary_states"])
        assert sign2 in ["U", "V"], f"Kỳ vọng Số 2 kích hoạt U hoặc V, nhận {sign2}"

        # 3. Số 6 xung đột với Chữ Y (do ngón cái và ngón út cùng duỗi)
        s6 = next(s for s in fixture_data["samples"] if s["digit"] == "6" and s["split"] == "dev")
        _, meta6 = extract_dual_hand_features(s6["landmarks"], None)
        sign6, conf6 = rec.classify_geometry(s6["landmarks"], meta6["primary_states"])
        assert sign6 == "Y", f"Kỳ vọng Số 6 kích hoạt Chữ Y trong bộ phân loại cũ, nhận {sign6}"

        # 4. Số 4 và 5 xung đột với Cử chỉ mở tay (CAM_ON)
        s5 = next(s for s in fixture_data["samples"] if s["digit"] == "5" and s["split"] == "dev")
        _, meta5 = extract_dual_hand_features(s5["landmarks"], None)
        sign5, conf5 = rec.classify_geometry(s5["landmarks"], meta5["primary_states"])
        assert sign5 in ["CAM_ON", "XIN_CHAO", "XIN_VUI_LONG"], f"Kỳ vọng Số 5 kích hoạt mở tay, nhận {sign5}"

