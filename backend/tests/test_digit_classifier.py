"""
Bộ kiểm thử Toàn diện cho Digit Classifier độc lập (VSL Digit Classifier).
==========================================================================
Giai đoạn 2C.4A: Kiểm chứng độc lập các quy tắc và ngưỡng phân loại số 1–9.

Các nhóm kiểm thử:
A. Dữ liệu DEV (1-9 từ real_digit_landmarks.json).
B. Dữ liệu EVAL (1-9 từ real_digit_landmarks.json, đánh giá độc lập không chỉnh ngưỡng).
C. Đầu vào không hợp lệ (None, rỗng, thiếu/thừa điểm, NaN, Inf, palm_scale=0).
D. Trường hợp biên, không khớp (NO_MATCH) và xung đột (CONFLICT).
"""

import os
import sys
import json
import math
# pyrefly: ignore[missing-import]
import pytest
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.vision.digit_classifier import (
    VSLDigitClassifier,
    digit_classifier,
    classify_vsl_digit,
    validate_input_landmarks,
    THUMB_LATERAL_THRESHOLD,
    ANGLE_TI_THRESHOLD,
    ANGLE_TP_THRESHOLD_8,
    ANGLE_TP_THRESHOLD_5_6,
    INDEX_CURL_MIN_9,
    INDEX_CURL_MAX_9
)


@pytest.fixture(scope="module")
def fixture_samples():
    """Tải dữ liệu fixture mốc thực tế từ real_digit_landmarks.json."""
    fpath = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


# ===========================================================================
# Nhóm A: Kiểm thử trên Tập DEV (Dữ liệu cơ sở xây dựng ngưỡng)
# ===========================================================================

class TestDevDataset:
    """Kiểm tra độ chính xác của bộ phân loại trên 9 mẫu DEV thực tế."""

    @pytest.mark.parametrize("digit_str", [str(i) for i in range(1, 10)])
    def test_dev_sample_classification(self, fixture_samples, digit_str):
        sample = next(s for s in fixture_samples if s["digit"] == digit_str and s["split"] == "dev")
        clf = VSLDigitClassifier()
        res = clf.classify_detailed(sample["landmarks"])

        assert res["status"] == "MATCH", (
            f"Số {digit_str} DEV thất bại. Status: {res['status']}, Reason: {res['reason']}"
        )
        assert res["digit"] == digit_str, (
            f"Số {digit_str} DEV bị phân loại sai thành {res['digit']}. Reason: {res['reason']}"
        )
        assert res["confidence"] >= 0.90
        assert res["matched_digits"] == [digit_str]

    def test_dev_overall_summary(self, fixture_samples):
        """Tổng hợp độ chính xác trên toàn bộ tập DEV."""
        dev_samples = [s for s in fixture_samples if s["split"] == "dev"]
        assert len(dev_samples) == 9

        correct = 0
        for s in dev_samples:
            d, conf = classify_vsl_digit(s["landmarks"])
            if d == s["digit"]:
                correct += 1

        assert correct == 9, f"Tập DEV đạt {correct}/9 mẫu."


# ===========================================================================
# Nhóm B: Kiểm thử Độc lập trên Tập EVAL (Không tinh chỉnh ngưỡng)
# ===========================================================================

class TestEvalDataset:
    """
    Đánh giá độc lập trên 9 mẫu EVAL.
    Lưu ý: Mẫu eval được trích xuất từ cùng nguồn video QIPEDC tại thời điểm frame khác với dev;
    không đại diện cho độ chính xác tổng quát hóa trên người ký hiệu mới.
    """

    @pytest.mark.parametrize("digit_str", [str(i) for i in range(1, 10)])
    def test_eval_sample_classification(self, fixture_samples, digit_str):
        sample = next(s for s in fixture_samples if s["digit"] == digit_str and s["split"] == "eval")
        clf = VSLDigitClassifier()
        res = clf.classify_detailed(sample["landmarks"])

        assert res["status"] == "MATCH", (
            f"Số {digit_str} EVAL thất bại. Status: {res['status']}, Reason: {res['reason']}"
        )
        assert res["digit"] == digit_str, (
            f"Số {digit_str} EVAL bị phân loại sai thành {res['digit']}. Reason: {res['reason']}"
        )
        assert res["confidence"] >= 0.90
        assert res["matched_digits"] == [digit_str]

    def test_eval_overall_summary(self, fixture_samples):
        """Tổng hợp độ chính xác trên toàn bộ tập EVAL."""
        eval_samples = [s for s in fixture_samples if s["split"] == "eval"]
        assert len(eval_samples) == 9

        correct = 0
        unmatched = 0
        conflicts = 0

        for s in eval_samples:
            res = digit_classifier.classify_detailed(s["landmarks"])
            if res["status"] == "MATCH" and res["digit"] == s["digit"]:
                correct += 1
            elif res["status"] == "NO_MATCH":
                unmatched += 1
            elif res["status"] == "CONFLICT":
                conflicts += 1

        assert correct == 9, (
            f"Tập EVAL: {correct} đúng, {unmatched} không khớp, {conflicts} xung đột."
        )


# ===========================================================================
# Nhóm C: Kiểm thử Đầu vào Không hợp lệ (Invalid Inputs Handling)
# ===========================================================================

class TestInvalidInputs:
    """Xác nhận xử lý an toàn tuyệt đối trước mọi đầu vào bất thường."""

    def test_none_input(self):
        res = digit_classifier.classify_detailed(None)
        assert res["status"] == "INVALID_INPUT"
        assert res["digit"] == "NEUTRAL"
        assert res["confidence"] == 0.0

        d, conf = classify_vsl_digit(None)
        assert d == "NEUTRAL" and conf == 0.0

    def test_empty_input(self):
        res = digit_classifier.classify_detailed([])
        assert res["status"] == "INVALID_INPUT"
        assert res["digit"] == "NEUTRAL"

    def test_incorrect_number_of_landmarks(self):
        # Ít hơn 21 điểm
        too_few = [[0.5, 0.5, 0.0]] * 20
        res_few = digit_classifier.classify_detailed(too_few)
        assert res_few["status"] == "INVALID_INPUT"
        assert "21 điểm" in res_few["reason"]

        # Nhiều hơn 21 điểm
        too_many = [[0.5, 0.5, 0.0]] * 22
        res_many = digit_classifier.classify_detailed(too_many)
        assert res_many["status"] == "INVALID_INPUT"

    def test_missing_coordinates_in_landmark(self):
        # Điểm mốc chỉ có 2 tọa độ (x, y)
        bad_dim = [[0.5, 0.5]] * 21
        res = digit_classifier.classify_detailed(bad_dim)
        assert res["status"] == "INVALID_INPUT"
        assert "không đủ 3 tọa độ" in res["reason"]

    def test_nan_values_in_landmarks(self):
        lms = [[0.5, 0.5, 0.0]] * 21
        lms[4] = [float("nan"), 0.5, 0.0]
        res = digit_classifier.classify_detailed(lms)
        assert res["status"] == "INVALID_INPUT"
        assert "NaN" in res["reason"]

    def test_inf_values_in_landmarks(self):
        lms = [[0.5, 0.5, 0.0]] * 21
        lms[8] = [0.5, float("inf"), 0.0]
        res = digit_classifier.classify_detailed(lms)
        assert res["status"] == "INVALID_INPUT"
        assert "vô cực" in res["reason"]

    def test_non_numeric_coordinates(self):
        lms = [[0.5, 0.5, 0.0]] * 21
        lms[0] = ["invalid", "str", "coords"]
        res = digit_classifier.classify_detailed(lms)
        assert res["status"] == "INVALID_INPUT"

    def test_degenerate_palm_scale(self):
        # Toàn bộ 21 điểm trùng nhau tại gốc tọa độ => palm_scale = 0
        zero_hand = [[0.0, 0.0, 0.0]] * 21
        res = digit_classifier.classify_detailed(zero_hand)
        assert res["status"] == "INVALID_INPUT"
        assert "palm_scale < 1e-5" in res["reason"]


# ===========================================================================
# Nhóm D: Kiểm thử Giá trị Biên, Không khớp (NO_MATCH) và Xung đột (CONFLICT)
# ===========================================================================

class TestBoundaryAndConflicts:
    """Kiểm tra các cử chỉ không khớp nhãn nào và trường hợp cận biên."""

    def test_fist_gesture_returns_no_match(self):
        """Cử chỉ nắm đấm (như chữ A/E) không được ép vào bất kỳ chữ số nào."""
        from backend.tests.test_geometry import _make_hand
        fist = _make_hand()

        res = digit_classifier.classify_detailed(fist)
        assert res["status"] == "NO_MATCH"
        assert res["digit"] == "NEUTRAL"
        assert len(res["matched_digits"]) == 0

    def test_isolated_middle_finger_returns_no_match(self):
        """Cử chỉ chỉ duỗi ngón giữa (không nằm trong quy ước số 1-9) phải trả về NO_MATCH."""
        wx, wy, wz = 0.5, 0.7, 0.0
        hand = [[wx, wy, wz] for _ in range(21)]
        hand[0] = [wx, wy, wz]
        hand[9] = [wx, wy - 0.25, wz]
        # Chỉ ngón giữa duỗi
        hand[10] = [wx, wy - 0.30, wz]
        hand[11] = [wx, wy - 0.35, wz]
        hand[12] = [wx, wy - 0.45, wz]

        res = digit_classifier.classify_detailed(hand)
        assert res["status"] == "NO_MATCH"
        assert res["digit"] == "NEUTRAL"

    def test_boundary_thumb_lateral_discrimination(self, fixture_samples):
        """
        Kiểm tra phân định ngón cái khép vs xòe qua ngưỡng THUMB_LATERAL_THRESHOLD (0.30).
        Số 1 (thumb_lat <= 0.30) vs Số 7 (thumb_lat > 0.30).
        """
        s1 = next(s for s in fixture_samples if s["digit"] == "1" and s["split"] == "dev")
        feat1 = digit_classifier.extract_features(s1["landmarks"])
        assert feat1["thumb_lat"] <= THUMB_LATERAL_THRESHOLD
        assert feat1["ang_TI"] < ANGLE_TI_THRESHOLD

        s7 = next(s for s in fixture_samples if s["digit"] == "7" and s["split"] == "dev")
        feat7 = digit_classifier.extract_features(s7["landmarks"])
        assert feat7["thumb_lat"] > THUMB_LATERAL_THRESHOLD
        assert feat7["ang_TI"] >= ANGLE_TI_THRESHOLD

    def test_evaluate_rules_returns_empty_when_no_rules_match(self):
        """Hàm evaluate_rules trả về danh sách rỗng khi đặc trưng không thỏa mãn quy tắc nào."""
        # Tạo dictionary đặc trưng nhân tạo đại diện cho cử chỉ không xác định
        dummy_feat = {
            "palm_scale": 0.25,
            "thumb_ext": False,
            "index_ext": False,
            "middle_ext": True,   # Chỉ ngón giữa
            "ring_ext": False,
            "pinky_ext": False,
            "index_curl": 0.8,
            "ang_TI": 10.0,
            "ang_TP": 10.0,
            "thumb_lat": 0.1
        }
        matches = digit_classifier.evaluate_rules(dummy_feat)
        assert len(matches) == 0

    def test_conflict_detection_mechanism(self):
        """
        Kiểm tra cơ chế phát hiện xung đột khi một bộ đặc trưng giả lập
        cùng lúc kích hoạt 2 quy tắc số.
        """
        clf = VSLDigitClassifier()
        # Tạo hàm mock evaluate_rules để kiểm tra xử lý xung đột của classify_detailed
        original_eval = clf.evaluate_rules
        try:
            clf.evaluate_rules = lambda f: [("1", "Lý do 1", 0.9), ("7", "Lý do 7", 0.9)]
            
            # Đưa một mẫu hợp lệ vào để trigger classify_detailed
            dummy_lms = [[0.5, 0.7 - i * 0.02, 0.0] for i in range(21)]
            dummy_lms[9] = [0.5, 0.4, 0.0]
            res = clf.classify_detailed(dummy_lms)

            assert res["status"] == "CONFLICT"
            assert res["digit"] == "CONFLICT"
            assert res["confidence"] == 0.0
            assert "1" in res["matched_digits"] and "7" in res["matched_digits"]
            assert "Xung đột nhiều nhãn" in res["reason"]
        finally:
            clf.evaluate_rules = original_eval
