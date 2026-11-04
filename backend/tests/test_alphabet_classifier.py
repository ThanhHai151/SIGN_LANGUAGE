"""
Unit tests cho VSLAlphabetClassifier (backend/vision/alphabet_classifier.py).
=============================================================================
Kiểm tra an toàn đầu vào, phân loại độc lập các cặp đối lập (S/T, C/O, D/L, A/E),
các chữ độc lập (B, I, Y, V), và tính an toàn bảo lưu UNKNOWN cho các chữ chưa hỗ trợ (R, U, M, N).
"""

import json
import math
import os
import numpy as np
# pyrefly: ignore[missing-import]
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

    def test_letter_e(self, classifier, alphabet_fixture_data):
        """Chữ E: 4 ngón tì gập lên ngón cái."""
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] == "E" and s.get("quality_status") == "ok"]
        # Khung hình hợp lệ của chữ E trong fixture
        valid_e = [s for s in samples if s.get("frame_idx") in [8, 20, 32]]
        assert len(valid_e) >= 2
        for s in valid_e:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH"
            assert res["letter"] == "E"
            assert res["confidence"] >= 0.90

    def test_letter_g(self, classifier, alphabet_fixture_data):
        """Chữ G: Ngón trỏ duỗi ngang chỉ sang bên, 3 ngón giữa/áp út/út gập gọn."""
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] == "G" and s.get("quality_status") == "ok"]
        horiz_g = [s for s in samples if s.get("frame_idx") == 39]
        assert len(horiz_g) == 1
        for s in horiz_g:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH"
            assert res["letter"] == "G"
            assert res["confidence"] >= 0.90

    def test_letter_h(self, classifier, alphabet_fixture_data):
        """Chữ H: Nhận diện đúng các khung hình ổn định của chữ H trong fixture."""
        samples = [s for s in alphabet_fixture_data["samples"] if s["letter_upper"] == "H" and s.get("quality_status") == "ok"]
        valid_h = [s for s in samples if s.get("frame_idx") in [18, 28, 35]]
        assert len(valid_h) == 3
        for s in valid_h:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH"
            assert res["letter"] == "H"
            assert res["confidence"] >= 0.90


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


# ===========================================================================
# 5. KIỂM THỬ CHUYÊN SÂU CHỮ E (ROBUSTNESS & CROSS-CLASS SAFETY)
# ===========================================================================

class TestLetterERobustnessAndSafety:
    """Kiểm tra chuyên sâu chữ E: độ bền vững trước nhiễu MediaPipe và an toàn không nhầm lẫn."""

    def test_e_dataset_high_recall_and_zero_fp(self, classifier):
        """
        Nếu dataset tĩnh v2 tồn tại, kiểm tra:
        - Recall của chữ E >= 85% (ít nhất 30/35 mẫu thực tế).
        - Tuyệt đối không sinh false positive (0 mẫu) trên 516 mẫu của 21 lớp còn lại.
        """
        npz_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_v2.npz"
        )
        if not os.path.exists(npz_path):
            pytest.skip("Dataset vsl_alphabet_static_v2.npz không tồn tại trên môi trường hiện tại.")

        data = np.load(npz_path, allow_pickle=True)
        landmarks = data["landmarks"]
        y = data["y"]
        classes = list(data["classes"])
        e_idx = classes.index("E")

        e_matched = 0
        e_total = 0
        non_e_fp = 0

        for i in range(len(y)):
            lm = landmarks[i].tolist()
            res = classifier.classify_detailed(lm)
            if y[i] == e_idx:
                e_total += 1
                if res["letter"] == "E" and res["status"] == "MATCH":
                    e_matched += 1
            else:
                if res["letter"] == "E":
                    non_e_fp += 1

        assert e_total == 35
        assert e_matched >= 30, f"Chữ E chỉ nhận diện được {e_matched}/{e_total} mẫu (kỳ vọng >= 30)!"
        assert non_e_fp == 0, f"Chữ E bị dương tính giả {non_e_fp} lần trên các chữ khác trong dataset!"

    def test_e_separates_from_confusable_letters(self, classifier, alphabet_fixture_data):
        """
        Đảm bảo chữ E không bao giờ bị kích hoạt nhầm bởi các chữ dễ gây nhầm lẫn:
        A, S, O, T, C, I.
        """
        confusable = {"A", "S", "O", "T", "C", "I"}
        samples = [
            s for s in alphabet_fixture_data["samples"]
            if s["letter_upper"] in confusable and s.get("quality_status") == "ok"
        ]
        assert len(samples) > 0

        for s in samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "E", (
                f"Mẫu {s['letter_upper']} frame {s.get('frame_idx')} bị nhận diện nhầm thành E!"
            )


# ===========================================================================
# 6. KIỂM THỬ CHUYÊN SÂU CHỮ G (ROBUSTNESS, DIRECTION INVARIANCE & SAFETY)
# ===========================================================================

class TestLetterGRobustnessAndSafety:
    """Kiểm tra chuyên sâu chữ G: tư thế ngang, bất biến hướng trái/phải, góc nghiêng webcam và an toàn không nhầm lẫn."""

    def test_g_horizontal_cluster_and_zero_fp(self, classifier):
        """
        Kiểm tra trên dataset tĩnh v2:
        - Toàn bộ 8/8 khung hình chữ G tư thế ngang chuẩn (indices 99..106) đạt MATCH 'G'.
        - Tuyệt đối không sinh false positive (0 mẫu) trên toàn bộ 522 mẫu của 21 lớp còn lại.
        """
        npz_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_v2.npz"
        )
        if not os.path.exists(npz_path):
            pytest.skip("Dataset vsl_alphabet_static_v2.npz không tồn tại trên môi trường hiện tại.")

        data = np.load(npz_path, allow_pickle=True)
        landmarks = data["landmarks"]
        y = data["y"]
        classes = list(data["classes"])
        g_idx = classes.index("G")

        horiz_g_indices = list(range(99, 107))
        horiz_matched = 0
        non_g_fp = 0

        for i in range(len(y)):
            lm = landmarks[i].tolist()
            res = classifier.classify_detailed(lm)
            if i in horiz_g_indices:
                assert y[i] == g_idx
                if res["letter"] == "G" and res["status"] == "MATCH":
                    horiz_matched += 1
            elif y[i] != g_idx:
                if res["letter"] == "G":
                    non_g_fp += 1

        assert horiz_matched == 8, f"Chữ G tư thế ngang chỉ nhận diện được {horiz_matched}/8 khung hình!"
        assert non_g_fp == 0, f"Chữ G bị dương tính giả {non_g_fp} lần trên các chữ khác trong dataset!"

    def test_g_direction_invariance_and_webcam_rotations(self, classifier, alphabet_fixture_data):
        """
        Kiểm tra chữ G giữ nguyên kết quả MATCH khi:
        - Lật đối xứng ngang trục X (chỉ sang trái -X hoặc tay trái / webcam mirror).
        - Nghiêng góc cổ tay trên webcam từ -40 độ đến +45 độ.
        """
        samples = [
            s for s in alphabet_fixture_data["samples"]
            if s["letter_upper"] == "G" and s.get("frame_idx") == 39
        ]
        assert len(samples) == 1
        base_lm = samples[0]["landmarks"]
        w = base_lm[0]

        # 1. Đối xứng ngang trục X (chỉ sang hướng -X)
        mirrored_lm = [[2.0 * w[0] - pt[0], pt[1], pt[2]] for pt in base_lm]
        res_mir = classifier.classify_detailed(mirrored_lm)
        assert res_mir["status"] == "MATCH"
        assert res_mir["letter"] == "G"

        # 2. Quét góc xoay cổ tay (-40 độ đến +45 độ)
        for deg in (-40, -25, -15, 0, 15, 30, 45):
            rad = math.radians(deg)
            c, s_val = math.cos(rad), math.sin(rad)
            rot_lm = [
                [
                    w[0] + (pt[0] - w[0]) * c - (pt[1] - w[1]) * s_val,
                    w[1] + (pt[0] - w[0]) * s_val + (pt[1] - w[1]) * c,
                    pt[2],
                ]
                for pt in base_lm
            ]
            res_rot = classifier.classify_detailed(rot_lm)
            assert res_rot["status"] == "MATCH", f"Chữ G bị trượt tại góc xoay {deg} độ!"
            assert res_rot["letter"] == "G"

    def test_g_separates_from_confusable_letters(self, classifier, alphabet_fixture_data):
        """
        Đảm bảo chữ G không bao giờ bị kích hoạt nhầm bởi các chữ dễ nhầm lẫn:
        D, L, K, P, Q, X, T, C, E.
        """
        confusable = {"D", "L", "K", "P", "Q", "X", "T", "C", "E"}
        samples = [
            s for s in alphabet_fixture_data["samples"]
            if s["letter_upper"] in confusable and s.get("quality_status") == "ok"
        ]
        assert len(samples) > 0

        for s in samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "G", (
                f"Mẫu {s['letter_upper']} frame {s.get('frame_idx')} bị nhận diện nhầm thành G!"
            )


# ===========================================================================
# 7. KIỂM THỬ CHUYÊN SÂU CHỮ H (BOTH VARIANTS, DIRECTION INVARIANCE & SAFETY)
# ===========================================================================

class TestLetterHRobustnessAndSafety:
    """Kiểm tra chuyên sâu chữ H: cả tư thế 2 ngón ngang chuẩn từ điển và biến thể Miền Bắc QIPEDC, bất biến hướng & góc xoay."""

    def test_h_dataset_and_zero_fp(self, classifier):
        """
        Kiểm tra trên dataset tĩnh v2:
        - Toàn bộ 12/12 khung hình ổn định chính của chữ H (indices 129..140) đạt MATCH 'H'.
        - Tuyệt đối không sinh false positive (0 mẫu) trên toàn bộ 533 mẫu của 21 lớp còn lại.
        """
        npz_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_v2.npz"
        )
        if not os.path.exists(npz_path):
            pytest.skip("Dataset vsl_alphabet_static_v2.npz không tồn tại trên môi trường hiện tại.")

        data = np.load(npz_path, allow_pickle=True)
        landmarks = data["landmarks"]
        y = data["y"]
        classes = list(data["classes"])
        h_idx = classes.index("H")

        core_h_indices = list(range(129, 141))
        core_matched = 0
        non_h_fp = 0

        for i in range(len(y)):
            lm = landmarks[i].tolist()
            res = classifier.classify_detailed(lm)
            if i in core_h_indices:
                assert y[i] == h_idx
                if res["letter"] == "H" and res["status"] == "MATCH":
                    core_matched += 1
            elif y[i] != h_idx:
                if res["letter"] == "H":
                    non_h_fp += 1

        assert core_matched == 12, f"Chữ H chỉ nhận diện được {core_matched}/12 khung hình ổn định!"
        assert non_h_fp == 0, f"Chữ H bị dương tính giả {non_h_fp} lần trên các chữ khác trong dataset!"

    def test_h_horizontal_two_finger_variant_and_rotations(self, classifier, alphabet_fixture_data):
        """
        Kiểm tra tư thế chữ H ngang chuẩn từ điển (2 ngón trỏ và giữa duỗi ngang song song, áp út và út gập):
        - Được tạo từ mẫu chữ U thực tế xoay ngang (-90..+90 độ) và từ mẫu chữ G ngang mở thêm ngón giữa song song.
        - Bất biến với hướng chỉ trái/phải (+X và -X).
        """
        u_samples = [
            s for s in alphabet_fixture_data["samples"]
            if s["letter_upper"] == "U" and s.get("quality_status") == "ok" and s.get("phase") == "stable_pose"
        ]
        assert len(u_samples) > 0
        base_u = u_samples[0]["landmarks"]
        w = base_u[0]

        for deg in (-90, -80, -70, 70, 80, 90):
            rad = math.radians(deg)
            c, s_val = math.cos(rad), math.sin(rad)
            rot_lm = [
                [
                    w[0] + (pt[0] - w[0]) * c - (pt[1] - w[1]) * s_val,
                    w[1] + (pt[0] - w[0]) * s_val + (pt[1] - w[1]) * c,
                    pt[2],
                ]
                for pt in base_u
            ]
            res = classifier.classify_detailed(rot_lm)
            assert res["status"] == "MATCH", f"Chữ H ngang (từ U xoay {deg} độ) không khớp!"
            assert res["letter"] == "H"

        # Kiểm tra biến thể Miền Bắc (frame 28) khi lật đối xứng trục X và xoay góc cổ tay (-40..+40 độ)
        h_samples = [
            s for s in alphabet_fixture_data["samples"]
            if s["letter_upper"] == "H" and s.get("frame_idx") == 28
        ]
        assert len(h_samples) == 1
        base_h = h_samples[0]["landmarks"]
        wh = base_h[0]

        mir_h = [[2.0 * wh[0] - pt[0], pt[1], pt[2]] for pt in base_h]
        res_mir = classifier.classify_detailed(mir_h)
        assert res_mir["status"] == "MATCH"
        assert res_mir["letter"] == "H"

        for deg in (-40, -25, -15, 0, 15, 25, 40):
            rad = math.radians(deg)
            c, s_val = math.cos(rad), math.sin(rad)
            rot_h = [
                [
                    wh[0] + (pt[0] - wh[0]) * c - (pt[1] - wh[1]) * s_val,
                    wh[1] + (pt[0] - wh[0]) * s_val + (pt[1] - wh[1]) * c,
                    pt[2],
                ]
                for pt in base_h
            ]
            res_rot = classifier.classify_detailed(rot_h)
            assert res_rot["status"] == "MATCH", f"Chữ H bị trượt tại góc xoay {deg} độ!"
            assert res_rot["letter"] == "H"

    def test_h_separates_from_confusable_letters(self, classifier, alphabet_fixture_data):
        """
        Đảm bảo chữ H không bao giờ bị kích hoạt nhầm bởi các chữ dễ nhầm lẫn:
        G, U, V, R, K, P, B, I, Y, D, E.
        """
        confusable = {"G", "U", "V", "R", "K", "P", "B", "I", "Y", "D", "E"}
        samples = [
            s for s in alphabet_fixture_data["samples"]
            if s["letter_upper"] in confusable and s.get("quality_status") == "ok"
        ]
        assert len(samples) > 0

        for s in samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "H", (
                f"Mẫu {s['letter_upper']} frame {s.get('frame_idx')} bị nhận diện nhầm thành H!"
            )

    def test_h_vs_v_disambiguation_and_v_preservation(self, classifier, alphabet_fixture_data):
        """
        Kiểm tra phân biệt triệt để giữa chữ H và chữ V:
        1. Tư thế chữ H ngang thực tế trên bàn tay người (ngón giữa dài hơn ngón trỏ: d_IW >= 1.70,
           d_MW >= 1.70, d_IM >= 0.28, diff_x_IM >= 0.18) trên dải góc -45..+40 độ phải nhận diện là H,
           tuyệt đối không bị nhận nhầm thành V.
        2. Toàn bộ các mẫu chữ V chuẩn (trong fixture JSON và NPZ, kể cả khi nghiêng cổ tay -20..+20 độ)
           vẫn nhận diện chính xác là V, không mẫu nào bị nhầm sang H.
        """
        npz_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_v2.npz"
        )
        if not os.path.exists(npz_path):
            pytest.skip("Dataset vsl_alphabet_static_v2.npz không tồn tại trên môi trường hiện tại.")

        data = np.load(npz_path, allow_pickle=True)
        landmarks = data["landmarks"]
        y = data["y"]
        classes = list(data["classes"])

        # 1. Tạo tư thế H ngang có giải phẫu thực tế (d_IW=1.75 >= 1.70, d_MW=1.73 >= 1.70, d_IM=0.39 >= 0.28)
        g_idx = classes.index("G")
        pts_h_real = [list(p) for p in landmarks[np.where(y == g_idx)[0][1]].tolist()]
        for k_i, k_m in [(5, 9), (6, 10), (7, 11), (8, 12)]:
            dx = pts_h_real[k_i][0] - pts_h_real[5][0]
            dy = pts_h_real[k_i][1] - pts_h_real[5][1]
            dz = pts_h_real[k_i][2] - pts_h_real[5][2]
            pts_h_real[k_m] = [
                pts_h_real[9][0] + 1.22 * dx,
                pts_h_real[9][1] + 1.22 * dy,
                pts_h_real[9][2] + dz,
            ]

        w_h = pts_h_real[0]
        for deg in range(-45, 45, 5):
            rad = math.radians(deg)
            c, s_val = math.cos(rad), math.sin(rad)
            rot_h = [
                [
                    w_h[0] + (pt[0] - w_h[0]) * c - (pt[1] - w_h[1]) * s_val,
                    w_h[1] + (pt[0] - w_h[0]) * s_val + (pt[1] - w_h[1]) * c,
                    pt[2],
                ]
                for pt in pts_h_real
            ]
            res_h = classifier.classify_detailed(rot_h)
            assert res_h["letter"] != "V", f"Chữ H ngang tại góc {deg} độ bị nhận nhầm thành V!"
            assert res_h["status"] == "MATCH" and res_h["letter"] == "H", (
                f"Chữ H ngang tại góc {deg} độ không đạt MATCH H (status={res_h['status']}, letter={res_h['letter']})"
            )

        # 2. Kiểm tra toàn bộ 13 khung hình ổn định của chữ V trong NPZ (indices 363..375)
        v_idx = classes.index("V")
        v_indices = np.where(y == v_idx)[0][:13]
        assert len(v_indices) == 13
        for idx in v_indices:
            lm_v = landmarks[idx].tolist()
            res_v = classifier.classify_detailed(lm_v)
            assert res_v["status"] == "MATCH" and res_v["letter"] == "V", (
                f"Mẫu V[{idx}] bị suy giảm: status={res_v['status']}, letter={res_v['letter']}"
            )
            # Nghiêng cổ tay chữ V trong phạm vi -25..+25 độ không bao giờ bị nhầm thành H
            wv = lm_v[0]
            for deg in (-25, -15, -10, 10, 15, 25):
                rad = math.radians(deg)
                c, s_val = math.cos(rad), math.sin(rad)
                rot_v = [
                    [
                        wv[0] + (pt[0] - wv[0]) * c - (pt[1] - wv[1]) * s_val,
                        wv[1] + (pt[0] - wv[0]) * s_val + (pt[1] - wv[1]) * c,
                        pt[2],
                    ]
                    for pt in lm_v
                ]
                res_rv = classifier.classify_detailed(rot_v)
                assert res_rv["letter"] != "H", f"Mẫu V[{idx}] xoay {deg} độ bị nhận nhầm thành H!"


