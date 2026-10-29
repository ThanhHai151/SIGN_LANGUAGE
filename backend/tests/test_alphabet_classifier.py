"""
Unit tests cho VSLAlphabetClassifier (backend/vision/alphabet_classifier.py).
=============================================================================
Kiểm tra an toàn đầu vào, phân loại độc lập các cặp đối lập (S/T, C/O, D/L, A/E),
các chữ độc lập (B, I, Y, V), và tính an toàn bảo lưu UNKNOWN cho các chữ chưa hỗ trợ (R, U, M, N).
"""

import json
import math
import os
import pytest
from typing import List

from backend.vision.alphabet_classifier import (
    VSLAlphabetClassifier,
    AlphabetClassifier,
    validate_input_landmarks
)


# Fixture nạp dữ liệu landmark bảng chữ cái thật từ Task 3A
FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "real_alphabet_landmarks.json"
)


@pytest.fixture(scope="module")
def alphabet_fixture_data():
    """Tải fixture landmark bảng chữ cái thật."""
    assert os.path.exists(FIXTURE_PATH), f"Fixture không tồn tại: {FIXTURE_PATH}"
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="module")
def classifier():
    """Khởi tạo một instance bộ phân loại VSLAlphabetClassifier."""
    return VSLAlphabetClassifier()


# ===========================================================================
# 1. KIỂM THỬ AN TOÀN ĐẦU VÀO (INPUT VALIDATION)
# ===========================================================================

class TestAlphabetClassifierInputValidation:
    """Kiểm tra phản ứng của bộ phân loại trước mọi loại đầu vào lỗi, bẩn, suy biến."""

    def test_alias_compatibility(self):
        """Đảm bảo AlphabetClassifier là alias tương thích của VSLAlphabetClassifier."""
        assert AlphabetClassifier is VSLAlphabetClassifier

    def test_none_input(self, classifier):
        """Đầu vào None phải trả về INVALID_INPUT và ("UNKNOWN", 0.0)."""
        valid, msg = validate_input_landmarks(None)
        assert not valid
        assert "None" in msg

        res = classifier.classify_detailed(None)
        assert res["status"] == "INVALID_INPUT"
        assert res["letter"] == "UNKNOWN"
        assert res["confidence"] == 0.0
        assert res["features"] is None

        letter, conf = classifier.classify(None)
        assert letter == "UNKNOWN"
        assert conf == 0.0

    @pytest.mark.parametrize("bad_type", [123, "landmarks_string", {"x": 1}, True])
    def test_invalid_data_types(self, classifier, bad_type):
        """Đầu vào sai kiểu dữ liệu (không phải list/tuple) phải bị từ chối an toàn."""
        valid, _ = validate_input_landmarks(bad_type)
        assert not valid
        res = classifier.classify_detailed(bad_type)
        assert res["status"] == "INVALID_INPUT"

    @pytest.mark.parametrize("bad_len", [0, 5, 20, 22, 42])
    def test_invalid_landmark_count(self, classifier, bad_len):
        """Đầu vào không đủ hoặc thừa 21 điểm phải bị từ chối an toàn."""
        pts = [[0.1, 0.2, 0.3]] * bad_len
        valid, msg = validate_input_landmarks(pts)
        assert not valid
        assert "21 điểm" in msg
        res = classifier.classify_detailed(pts)
        assert res["status"] == "INVALID_INPUT"

    def test_non_numeric_coordinates(self, classifier):
        """Tọa độ chứa giá trị chuỗi hoặc None phải bị từ chối."""
        pts = [[0.1, 0.2, 0.3]] * 21
        pts[5] = ["a", 0.2, 0.3]
        valid, msg = validate_input_landmarks(pts)
        assert not valid
        assert "không phải số" in msg
        res = classifier.classify_detailed(pts)
        assert res["status"] == "INVALID_INPUT"

    def test_nan_coordinates(self, classifier):
        """Tọa độ chứa NaN phải bị từ chối ngay lập tức."""
        pts = [[0.1, 0.2, 0.3]] * 21
        pts[10] = [float("nan"), 0.2, 0.3]
        valid, msg = validate_input_landmarks(pts)
        assert not valid
        assert "NaN" in msg
        res = classifier.classify_detailed(pts)
        assert res["status"] == "INVALID_INPUT"

    def test_inf_coordinates(self, classifier):
        """Tọa độ chứa vô cực (Inf) phải bị từ chối."""
        pts = [[0.1, 0.2, 0.3]] * 21
        pts[8] = [0.1, float("inf"), 0.3]
        valid, msg = validate_input_landmarks(pts)
        assert not valid
        assert "Inf" in msg
        res = classifier.classify_detailed(pts)
        assert res["status"] == "INVALID_INPUT"

    def test_degenerate_zero_scale(self, classifier):
        """Trường hợp 21 điểm mốc trùng khít tại gốc (palm_scale = 0)."""
        pts = [[0.0, 0.0, 0.0]] * 21
        valid, _ = validate_input_landmarks(pts)
        assert valid  # Về kiểu số thì hợp lệ nhưng suy biến hình học
        feat = classifier.extract_features(pts)
        assert feat is None
        res = classifier.classify_detailed(pts)
        assert res["status"] == "INVALID_INPUT"
        assert "suy biến" in res["reason"]


# ===========================================================================
# 2. KIỂM THỬ CÁC CẶP ĐỐI LẬP (CONTRASTING PAIRS)
# ===========================================================================

class TestAlphabetClassifierContrastingPairs:
    """Kiểm tra tính phân tách độc lập và chính xác của các cặp dễ nhầm lẫn."""

    def test_s_vs_t_distinction(self, classifier, alphabet_fixture_data):
        """
        Cặp S vs T:
        - S: Nắm đấm tròn, ngón cái vắt ngang mặt ngoài.
        - T: Ngón cái kẹp giữa ngón trỏ và ngón giữa.
        Khẳng định: Không bao giờ nhầm lẫn chéo giữa S và T trên dữ liệu thật.
        """
        samples = alphabet_fixture_data["samples"]
        s_samples = [s for s in samples if s["letter_upper"] == "S" and s.get("quality_status") == "ok"]
        t_samples = [s for s in samples if s["letter_upper"] == "T" and s.get("quality_status") == "ok"]

        assert len(s_samples) > 0 and len(t_samples) > 0

        # Kiểm tra mẫu T: Không bao giờ ra S
        for s in t_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "S", f"Mẫu T frame {s['frame_idx']} bị nhầm thành S!"
            if s.get("phase") == "stable_pose":
                assert res["status"] == "MATCH"
                assert res["letter"] == "T"

        # Kiểm tra mẫu S: Không bao giờ ra T
        for s in s_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "T", f"Mẫu S frame {s['frame_idx']} bị nhầm thành T!"
            if s.get("phase") == "stable_pose" and s["frame_idx"] in [29, 36]:
                assert res["status"] == "MATCH"
                assert res["letter"] == "S"

    def test_c_vs_o_distinction(self, classifier, alphabet_fixture_data):
        """
        Cặp C vs O:
        - C: Các ngón khum cong mở rộng hình vòm chữ C.
        - O: Các ngón cong khép kín chạm đầu ngón cái thành vòng tròn kín.
        Khẳng định: Tuyệt đối không nhầm chéo C và O.
        """
        samples = alphabet_fixture_data["samples"]
        c_samples = [s for s in samples if s["letter_upper"] == "C" and s.get("quality_status") == "ok"]
        o_samples = [s for s in samples if s["letter_upper"] == "O" and s.get("quality_status") == "ok"]

        # Kiểm tra O
        for s in o_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "C", f"Mẫu O frame {s['frame_idx']} bị nhầm thành C!"
            if s.get("phase") == "stable_pose":
                assert res["status"] == "MATCH"
                assert res["letter"] == "O"

        # Kiểm tra C
        for s in c_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "O", f"Mẫu C frame {s['frame_idx']} bị nhầm thành O!"
            if s.get("phase") == "stable_pose":
                assert res["status"] == "MATCH"
                assert res["letter"] == "C"

    def test_d_vs_l_distinction(self, classifier, alphabet_fixture_data):
        """
        Cặp D vs L:
        - D: Chỉ ngón trỏ duỗi thẳng, ngón cái khép chạm đầu ngón giữa.
        - L: Ngón trỏ và ngón cái duỗi vuông góc (L-shape).
        Khẳng định: Tuyệt đối không nhầm chéo D và L.
        """
        samples = alphabet_fixture_data["samples"]
        d_samples = [s for s in samples if s["letter_upper"] == "D" and s.get("quality_status") == "ok"]
        l_samples = [s for s in samples if s["letter_upper"] == "L" and s.get("quality_status") == "ok"]

        for s in d_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "L", f"Mẫu D frame {s['frame_idx']} bị nhầm thành L!"
            if s.get("phase") == "stable_pose" and s["frame_idx"] in [35, 44]:
                assert res["status"] == "MATCH"
                assert res["letter"] == "D"

        for s in l_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "D", f"Mẫu L frame {s['frame_idx']} bị nhầm thành D!"
            if s.get("phase") == "stable_pose" and s["frame_idx"] in [26, 33]:
                assert res["status"] == "MATCH"
                assert res["letter"] == "L"

    def test_a_vs_e_distinction(self, classifier, alphabet_fixture_data):
        """
        Cặp A vs E:
        - A: Nắm đấm với ngón cái dựng đứng thẳng cạnh bàn tay.
        - E: 4 ngón cong gập tì lên đầu ngón cái.
        Khẳng định: Tuyệt đối không nhầm chéo A và E.
        """
        samples = alphabet_fixture_data["samples"]
        a_samples = [s for s in samples if s["letter_upper"] == "A" and s.get("quality_status") == "ok"]
        e_samples = [s for s in samples if s["letter_upper"] == "E" and s.get("quality_status") == "ok"]

        for s in a_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "E", f"Mẫu A frame {s['frame_idx']} bị nhầm thành E!"
            assert res["status"] == "MATCH"
            assert res["letter"] == "A"

        for s in e_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "A", f"Mẫu E frame {s['frame_idx']} bị nhầm thành A!"
            if s.get("phase") == "stable_pose" and s["frame_idx"] in [20, 32]:
                assert res["status"] == "MATCH"
                assert res["letter"] == "E"


# ===========================================================================
# 3. KIỂM THỬ CÁC CHỮ TĨNH ĐỘC LẬP (B, I, Y, V)
# ===========================================================================

class TestAlphabetClassifierDistinctLetters:
    """Kiểm tra các chữ tĩnh có hình thái độc lập cao."""

    def test_letter_b(self, classifier, alphabet_fixture_data):
        """Chữ B: 4 ngón duỗi thẳng khép sát, ngón cái khép."""
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] == "B" and s.get("quality_status") == "ok"]
        stable = [s for s in samples if s["frame_idx"] in [23, 37, 46]]
        for s in stable:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH"
            assert res["letter"] == "B"

    def test_letter_i(self, classifier, alphabet_fixture_data):
        """Chữ I: Chỉ duy nhất ngón út duỗi thẳng."""
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] == "I" and s.get("quality_status") == "ok"]
        for s in samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH"
            assert res["letter"] == "I"

    def test_letter_y(self, classifier, alphabet_fixture_data):
        """Chữ Y: Cả ngón cái và ngón út cùng duỗi thẳng (Shaka)."""
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] == "Y" and s.get("quality_status") == "ok"]
        for s in samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH"
            assert res["letter"] == "Y"

    def test_letter_v(self, classifier, alphabet_fixture_data):
        """Chữ V: Ngón trỏ và ngón giữa xòe chữ V."""
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] == "V" and s.get("quality_status") == "ok"]
        stable = [s for s in samples if s["frame_idx"] in [16, 25, 31, 37]]
        for s in stable:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH"
            assert res["letter"] == "V"


# ===========================================================================
# 4. KIỂM THỬ BẢO VỆ CHỮ CHƯA HỖ TRỢ (UNSUPPORTED / UNKNOWN SAFETY)
# ===========================================================================

class TestAlphabetClassifierUnsupportedSafety:
    """Đảm bảo các chữ khó phân biệt (R/U, M/N) được giữ UNKNOWN an toàn."""

    def test_r_and_u_kept_unknown(self, classifier, alphabet_fixture_data):
        """
        Cặp R vs U: Hiện chưa đủ căn cứ phân tách ổn định trong 2D đơn signer.
        Bộ phân loại phải trả về UNKNOWN hoặc NO_MATCH, không được gán nhãn tùy tiện.
        """
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] in ["R", "U"] and s.get("quality_status") == "ok"]
        assert len(samples) > 0
        for s in samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] in ["UNKNOWN", None]
            assert res["status"] in ["NO_MATCH", "CONFLICT"]

    def test_m_and_n_kept_unknown(self, classifier, alphabet_fixture_data):
        """
        Cặp M vs N: Vị trí ngón cái bị che khuất trong 2D chính diện.
        Bộ phân loại phải trả về UNKNOWN / NO_MATCH.
        """
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] in ["M", "N"] and s.get("quality_status") == "ok"]
        assert len(samples) > 0
        for s in samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] in ["UNKNOWN", None]
            assert res["status"] in ["NO_MATCH", "CONFLICT"]

    def test_no_false_positive_cross_labeling_among_supported(self, classifier, alphabet_fixture_data):
        """
        Kiểm tra tổng quát: Trong toàn bộ 12 chữ được hỗ trợ (A, B, C, D, E, I, L, O, S, T, V, Y),
        không một mẫu nào bị phân loại chéo sang một chữ khác trong nhóm này.
        """
        target_letters = {"A", "B", "C", "D", "E", "I", "L", "O", "S", "T", "V", "Y"}
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] in target_letters and s.get("quality_status") == "ok"]

        wrong_count = 0
        for s in samples:
            true_letter = s["letter_upper"]
            res = classifier.classify_detailed(s["landmarks"])
            pred = res["letter"]
            if pred != "UNKNOWN" and pred != true_letter:
                wrong_count += 1

        assert wrong_count == 0, f"Có {wrong_count} mẫu bị phân loại sai chéo giữa các chữ được hỗ trợ!"

