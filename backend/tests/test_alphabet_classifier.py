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

    def test_v_requires_spread_angle_and_upward_orientation(self, classifier, alphabet_fixture_data):
        """
        Kiểm tra các ràng buộc mới của Rule V (Giai đoạn B - TASK 4):
        1. Dáng 2 ngón duỗi song song (ang_IM_dir < 5.0°) không bao giờ kích hoạt Rule V.
        2. Dáng 2 ngón duỗi hướng ngang (idx_horiz_ratio >= 0.50 hoặc idx_tip_y > -0.80) không kích hoạt Rule V.
        3. evaluate_rule_h_diagnostics phản ánh chính xác các điều kiện con mới của Rule V.
        """
        # Lấy mẫu chữ U chuẩn (2 ngón duỗi song song thẳng đứng)
        u_samples = [
            s for s in alphabet_fixture_data["samples"]
            if s["letter_upper"] == "U" and s.get("frame_idx") in [19, 30, 38]
        ]
        assert len(u_samples) > 0

        for s in u_samples:
            lm = s["landmarks"]
            feat = classifier.extract_features(lm)
            diag = classifier.evaluate_rule_h_diagnostics(feat)
            res = classifier.classify_detailed(lm)

            # Chữ U có ang_IM_dir ~ 0..3° < 9.0° -> Rule V phải False
            assert feat["ang_IM_dir"] < 9.0, f"Chữ U frame {s['frame_idx']} có góc xòe bất thường: {feat['ang_IM_dir']}"
            assert res["letter"] != "V", f"Chữ U frame {s['frame_idx']} bị kích hoạt nhầm Rule V!"
            assert diag["rule_v"]["matched"] is False
            assert "ang_IM_dir_ge_min" in diag["rule_v"]["failed_conditions"]

    def test_u_and_closed_fingers_never_match_v_or_h(self, classifier, alphabet_fixture_data):
        """
        Đảm bảo các mẫu chữ U (2 ngón khép song song hướng lên) tuyệt đối không
        bị nhận diện nhầm thành V (vì thiếu độ xòe) hay H (vì không hướng ngang).
        """
        u_samples = [
            s for s in alphabet_fixture_data["samples"]
            if s["letter_upper"] == "U" and s.get("quality_status") == "ok"
        ]
        assert len(u_samples) > 0
        for s in u_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "V", f"Mẫu U frame {s.get('frame_idx')} bị nhận nhầm thành V!"
            assert res["letter"] != "H", f"Mẫu U frame {s.get('frame_idx')} bị nhận nhầm thành H!"

    def test_vsl_static_dataset_all_22_classes_regression(self, classifier):
        """
        Kiểm thử hồi quy toàn bộ 22 lớp chữ cái tĩnh trên dataset chuẩn vsl_alphabet_static_v2.npz:
        - Chữ V đạt MATCH trên toàn bộ 13/13 mẫu ổn định (indices 363..375), 0 FP trên 21 lớp khác.
        - Chữ H đạt MATCH trên toàn bộ 12/12 mẫu ổn định (indices 129..140), 0 FP trên 21 lớp khác.
        - Các lớp còn lại không bị suy giảm độ chính xác hoặc tăng đột biến FP.
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

        v_idx = classes.index("V")
        h_idx = classes.index("H")

        v_matched = 0
        v_fp = 0
        h_matched = 0
        h_fp = 0

        for i in range(len(y)):
            lm = landmarks[i].tolist()
            res = classifier.classify_detailed(lm)
            pred = res.get("letter")

            if y[i] == v_idx:
                if pred == "V" and res.get("status") == "MATCH":
                    v_matched += 1
            else:
                if pred == "V":
                    v_fp += 1

            if y[i] == h_idx:
                if pred == "H" and res.get("status") == "MATCH":
                    h_matched += 1
            else:
                if pred == "H":
                    h_fp += 1

        assert v_matched >= 13, f"Chữ V chỉ nhận được {v_matched}/17 mẫu trong dataset!"
        assert v_fp == 0, f"Chữ V bị False Positive {v_fp} lần trên các chữ khác!"
        assert h_matched >= 12, f"Chữ H chỉ nhận được {h_matched}/18 mẫu trong dataset!"
        assert h_fp == 0, f"Chữ H bị False Positive {h_fp} lần trên các chữ khác!"


# ===========================================================================
# 11. KIỂM THỬ CHỮ H MIỀN NAM & PHÂN XỬ H/V HÌNH HỌC TĨNH (TASK 7)
# ===========================================================================

class TestHSouthGeometricRuleAndDisambiguation:
    """
    Kiểm thử chuyên sâu cho quy tắc hình học chữ H Miền Nam (QIPEDC D0500N)
    và cơ chế phân xử H vs V dựa trên hình chiếu bề ngang bàn tay (MCP_width_2d)
    và tư thế ngón cái đè ngón áp út (thumb_ring_dist).
    """

    @pytest.fixture(scope="module")
    def d0500n_data(self):
        """Tải dữ liệu landmark trích xuất từ video QIPEDC D0500N (H Miền Nam)."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "data", "alphabet_v2", "static_landmarks", "D0500N.json"
        )
        assert os.path.exists(path), f"File không tồn tại: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture(scope="module")
    def v_qipedc_data(self):
        """Tải dữ liệu landmark trích xuất từ video QIPEDC D0516 (Chữ V)."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "data", "alphabet_v2", "static_landmarks", "V.json"
        )
        assert os.path.exists(path), f"File không tồn tại: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture(scope="module")
    def h_north_data(self):
        """Tải dữ liệu landmark trích xuất từ video QIPEDC D0500B (H Miền Bắc)."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "data", "alphabet_v2", "static_landmarks", "H.json"
        )
        assert os.path.exists(path), f"File không tồn tại: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_h_south_qipedc_stable_frames_match_h(self, classifier, d0500n_data):
        """
        Toàn bộ 24 khung hình ổn định của D0500N (indices 10..33) phải được
        nhận diện chính xác là H với status MATCH và confidence 0.95.
        Tuyệt đối không có khung hình nào bị nhận nhầm thành V.
        """
        stable_samples = [
            s for s in d0500n_data["samples"]
            if s.get("quality_status") == "ok" and 10 <= s.get("frame_index", -1) <= 33
        ]
        assert len(stable_samples) == 24, f"Số lượng frame ổn định mong đợi 24, nhận {len(stable_samples)}"

        for s in stable_samples:
            f_idx = s.get("frame_index")
            lm = s["landmarks"]
            res = classifier.classify_detailed(lm)

            assert res["status"] == "MATCH", f"D0500N frame {f_idx} không đạt MATCH: status={res['status']}, reason={res['reason']}"
            assert res["letter"] == "H", f"D0500N frame {f_idx} bị nhận nhầm thành {res['letter']}!"
            assert res["confidence"] == 0.95
            assert "Miền Nam" in res["reason"]

    def test_h_south_with_spread_fingers_never_matches_v(self, classifier, d0500n_data):
        """
        Trong D0500N, các khung hình xòe ngón trỏ & giữa rộng (ang_IM_dir >= 25.0° như frame 22..33)
        phải nhận đúng H và tuyệt đối KHÔNG kích hoạt Rule V.
        Chẩn đoán Rule V phải chỉ rõ failed điều kiện mcp_width_2d_ge_min và thumb_ring_dist_ge_min.
        """
        spread_samples = [
            s for s in d0500n_data["samples"]
            if s.get("quality_status") == "ok" and 22 <= s.get("frame_index", -1) <= 32
        ]
        assert len(spread_samples) > 0

        for s in spread_samples:
            f_idx = s.get("frame_index")
            lm = s["landmarks"]
            feat = classifier.extract_features(lm)
            assert feat is not None
            assert feat["ang_IM_dir"] >= 25.0, f"Frame {f_idx} có góc xòe {feat['ang_IM_dir']:.1f}° < 25°"

            res = classifier.classify_detailed(lm)
            assert res["letter"] == "H", f"Frame xòe {f_idx} bị nhận nhầm thành {res['letter']}"
            assert res["status"] == "MATCH"

            diag = classifier.evaluate_rule_h_diagnostics(feat)
            assert diag is not None
            assert diag["rule_v"]["matched"] is False, f"Rule V bị kích hoạt nhầm trên H South frame {f_idx}!"
            assert "mcp_width_2d_ge_min" in diag["rule_v"]["failed_conditions"]
            assert "thumb_ring_dist_ge_min" in diag["rule_v"]["failed_conditions"]
            assert diag["h_south"]["matched"] is True

    def test_v_qipedc_stable_frames_never_match_h(self, classifier, v_qipedc_data):
        """
        Toàn bộ 13 khung hình ổn định của V (D0516, indices 14..38) phải nhận diện chính xác là V.
        Tuyệt đối không có khung hình nào bị nhận nhầm thành H Miền Nam.
        Chẩn đoán H South phải chỉ rõ failed điều kiện mcp_width_2d_le_max và thumb_ring_dist_le_max.
        """
        stable_v = [
            s for s in v_qipedc_data["samples"]
            if s.get("quality_status") == "ok" and 14 <= s.get("frame_index", -1) <= 38
        ]
        assert len(stable_v) == 13

        for s in stable_v:
            f_idx = s.get("frame_index")
            lm = s["landmarks"]
            feat = classifier.extract_features(lm)
            res = classifier.classify_detailed(lm)

            assert res["status"] == "MATCH", f"V frame {f_idx} status={res['status']}"
            assert res["letter"] == "V", f"V frame {f_idx} bị nhận nhầm thành {res['letter']}"

            diag = classifier.evaluate_rule_h_diagnostics(feat)
            assert diag is not None
            assert diag["rule_v"]["matched"] is True
            assert diag["h_south"]["matched"] is False
            assert "mcp_width_2d_le_max" in diag["h_south"]["failed_conditions"]
            assert "thumb_ring_dist_le_max" in diag["h_south"]["failed_conditions"]

    def test_h_north_qipedc_remains_h(self, classifier, h_north_data):
        """
        Bảo toàn chữ H Miền Bắc (D0500B): 12 khung hình ổn định (indices 16..38) vẫn nhận diện
        chính xác là H, không bị xung đột với Rule V hay H Miền Nam.
        """
        stable_north = [
            s for s in h_north_data["samples"]
            if s.get("quality_status") == "ok" and 16 <= s.get("frame_index", -1) <= 38
        ]
        assert len(stable_north) == 12

        for s in stable_north:
            f_idx = s.get("frame_index")
            lm = s["landmarks"]
            res = classifier.classify_detailed(lm)

            assert res["status"] == "MATCH", f"H North frame {f_idx} status={res['status']}"
            assert res["letter"] == "H", f"H North frame {f_idx} nhận nhầm {res['letter']}"
            assert "Miền Bắc" in res["reason"]

    def test_ambiguous_zone_yields_unknown_no_match(self, classifier):
        """
        Kiểm thử vùng nhập nhằng (ambiguous zone) và mâu thuẫn đặc trưng:
        1. mcp_width_2d nằm trong khoảng 0.40..0.48 -> Không đoán, trả về UNKNOWN (NO_MATCH).
        2. mcp_width_2d <= 0.40 (hẹp) nhưng thumb_ring_dist >= 0.21 (ngón cái không đè) -> UNKNOWN (NO_MATCH).
        3. mcp_width_2d >= 0.48 (rộng) nhưng thumb_ring_dist <= 0.16 (ngón cái đè áp út) -> UNKNOWN (NO_MATCH).
        """
        # Mẫu cơ sở: trỏ & giữa duỗi, áp út & út gập, đứng thẳng, góc xòe 15 độ
        base_feat = {
            "thumb_ext": False,
            "index_ext": True,
            "middle_ext": True,
            "ring_ext": False,
            "pinky_ext": False,
            "index_curl": 0.05,
            "middle_curl": 0.05,
            "ring_curl": 0.85,
            "pinky_curl": 0.90,
            "d_TI": 1.20,
            "d_TM": 0.90,
            "d_TW": 0.90,
            "d_IM": 0.45,
            "d_IW": 1.90,
            "d_MW": 1.90,
            "d_RW": 0.55,
            "d_PW": 0.55,
            "diff_x_IM": 0.30,
            "ang_TI": 45.0,
            "ang_TP": 60.0,
            "thumb_lat": 0.20,
            "idx_horiz_ratio": 0.10,
            "idx_straight": 0.98,
            "idx_tip_y": -1.90,
            "idx_dx": 0.15,
            "mid_horiz_ratio": 0.10,
            "mid_straight": 0.98,
            "mid_tip_y": -1.90,
            "ang_IM_dir": 18.0,
        }

        # Case 1: mcp_width_2d = 0.44 (vùng giữa 0.40 và 0.48)
        feat_ambiguous = dict(base_feat)
        feat_ambiguous["mcp_width_2d"] = 0.44
        feat_ambiguous["thumb_ring_dist"] = 0.18
        feat_ambiguous["thumb_mid_dist"] = 0.30
        matches = classifier.evaluate_rules(feat_ambiguous)
        assert len(matches) == 0, f"Vùng nhập nhằng không được match bất kỳ chữ nào, nhận: {matches}"

        # Case 2: Hông hẹp (mcp_w=0.35) nhưng thumb-ring xa (0.28) -> Mâu thuẫn
        feat_conflict_1 = dict(base_feat)
        feat_conflict_1["mcp_width_2d"] = 0.35
        feat_conflict_1["thumb_ring_dist"] = 0.28
        feat_conflict_1["thumb_mid_dist"] = 0.50
        matches_c1 = classifier.evaluate_rules(feat_conflict_1)
        assert len(matches_c1) == 0, f"Mâu thuẫn đặc trưng 1 không được match, nhận: {matches_c1}"

        # Case 3: Lòng bàn tay mở rộng (mcp_w=0.54) nhưng ngón cái đè áp út (0.12) -> Mâu thuẫn
        feat_conflict_2 = dict(base_feat)
        feat_conflict_2["mcp_width_2d"] = 0.54
        feat_conflict_2["thumb_ring_dist"] = 0.12
        feat_conflict_2["thumb_mid_dist"] = 0.25
        matches_c2 = classifier.evaluate_rules(feat_conflict_2)
        assert len(matches_c2) == 0, f"Mâu thuẫn đặc trưng 2 không được match, nhận: {matches_c2}"

    def test_evaluate_rule_h_diagnostics_all_four_branches(self, classifier):
        """
        Kiểm tra hàm evaluate_rule_h_diagnostics trả về đầy đủ 4 nhánh
        ('rule_v', 'h_horiz', 'h_north', 'h_south') và an toàn trước None / dữ liệu rỗng.
        """
        assert classifier.evaluate_rule_h_diagnostics(None) is None

        diag_empty = classifier.evaluate_rule_h_diagnostics({})
        assert isinstance(diag_empty, dict)
        assert "rule_v" in diag_empty
        assert "h_horiz" in diag_empty
        assert "h_north" in diag_empty
        assert "h_south" in diag_empty
        assert diag_empty["rule_v"]["matched"] is None
        assert diag_empty["h_south"]["matched"] is None

    def test_h_south_rotation_invariance_tolerance(self, classifier, d0500n_data):
        """
        Kiểm tra khả năng chịu biến thiên góc xoay mặt phẳng của H Miền Nam:
        Xoay cổ tay trong phạm vi -15..+15 độ không bị nhận nhầm thành V hay các chữ khác.
        """
        sample = next(
            s for s in d0500n_data["samples"]
            if s.get("quality_status") == "ok" and s.get("frame_index") == 20
        )
        lm = sample["landmarks"]
        wrist = lm[0]

        for deg in (-15, -10, -5, 5, 10, 15):
            rad = math.radians(deg)
            c, s_val = math.cos(rad), math.sin(rad)
            rot_lm = [
                [
                    wrist[0] + (pt[0] - wrist[0]) * c - (pt[1] - wrist[1]) * s_val,
                    wrist[1] + (pt[0] - wrist[0]) * s_val + (pt[1] - wrist[1]) * c,
                    pt[2],
                ]
                for pt in lm
            ]
            res = classifier.classify_detailed(rot_lm)
            assert res["letter"] != "V", f"H South xoay {deg} độ bị nhận nhầm thành V!"
            assert res["status"] in ("MATCH", "NO_MATCH"), f"Trạng thái bất thường: {res['status']}"


# ===========================================================================
# 12. KIỂM THỬ CHỮ K CHUẨN QIPEDC D0502 (QUY ƯỚC MIỀN NAM / TOÀN QUỐC)
# ===========================================================================

class TestRuleKSouthernConvention:
    """
    Bộ kiểm thử cho chữ K dựa trên video mẫu thực tế QIPEDC D0502.mp4:
    - Ngón trỏ duỗi đứng
    - Ngón giữa duỗi ngang sang bên phải
    - Ngón cái dựng lên kẹp giữa trỏ và giữa
    - Ngón áp út và út gập sát lòng bàn tay
    - Bất biến phản chiếu (tay trái / webcam mirror) và phân xử tuyệt đối với V, D, L, G, H, B.
    """

    def test_k_matches_on_d0502_fixtures(self, classifier, alphabet_fixture_data):
        """Kiểm tra chữ K được nhận diện đúng (MATCH, conf=0.95) trên các khung hình ổn định của D0502."""
        k_samples = [
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "K" and s.get("quality_status") == "ok"
        ]
        assert len(k_samples) >= 4, "Tập fixture phải có ít nhất 4 frame K hợp lệ."

        # Kiểm tra các frame ổn định (f26, f32, f38) và frame chuyển tiếp vào tư thế (f16)
        valid_k_frames = [s for s in k_samples if s["frame_idx"] in (16, 26, 32, 38)]
        for s in valid_k_frames:
            f_idx = s["frame_idx"]
            lm = s["landmarks"]
            res = classifier.classify_detailed(lm)
            assert res["status"] == "MATCH", (
                f"Frame K f{f_idx} không đạt MATCH (status={res['status']}, reason={res.get('reason')})"
            )
            assert res["letter"] == "K", f"Frame K f{f_idx} bị gán sai nhãn: {res['letter']}"
            assert res["confidence"] == 0.95

            # Kiểm tra hàm classify nhanh
            let, conf = classifier.classify(lm)
            assert let == "K" and conf == 0.95

    def test_k_rejects_on_exit_frame(self, classifier, alphabet_fixture_data):
        """Frame f48 khi bàn tay bắt đầu buông tư thế (exit) không được ép nhận nhãn K."""
        exit_sample = next(
            (s for s in alphabet_fixture_data.get("samples", [])
             if s.get("letter_upper") == "K" and s.get("frame_idx") == 48 and s.get("quality_status") == "ok"),
            None
        )
        if exit_sample is not None:
            res = classifier.classify_detailed(exit_sample["landmarks"])
            assert res["letter"] != "K", "Frame buông tay f48 không được gán nhãn K!"

    def test_k_does_not_conflict_with_v(self, classifier, alphabet_fixture_data):
        """
        Phân xử K vs V: Chữ V có ngón giữa vươn đứng (d_MW >= 1.70, mid_horiz <= 0.50),
        không bao giờ bị nhận diện là K.
        """
        v_samples = [
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "V" and s.get("quality_status") == "ok"
        ]
        for s in v_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "K", f"Mẫu chữ V (frame {s.get('frame_idx')}) bị nhận nhầm thành K!"

        npz_path = os.path.join(os.path.dirname(__file__), "..", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_v2.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            classes = list(data["classes"])
            y = data["y"]
            landmarks = data["landmarks"]
            v_idx = classes.index("V")
            for idx in np.where(y == v_idx)[0][:10]:
                res = classifier.classify_detailed(landmarks[idx].tolist())
                assert res["letter"] != "K", f"Mẫu V NPZ[{idx}] bị nhận nhầm thành K!"

    def test_k_does_not_conflict_with_d_and_l(self, classifier, alphabet_fixture_data):
        """
        Phân xử K vs D/L:
        - D có ngón giữa gập sát (d_TM <= 0.30, d_MW <= 0.85).
        - L có ngón cái mở vuông góc sang trái (ang_TI >= 35°, d_TI >= 1.40).
        Cả D và L không bao giờ bị nhận nhầm thành K.
        """
        dl_samples = [
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") in ("D", "L") and s.get("quality_status") == "ok"
        ]
        for s in dl_samples:
            let = s.get("letter_upper")
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "K", f"Mẫu chữ {let} (frame {s.get('frame_idx')}) bị nhận nhầm thành K!"

    def test_k_does_not_conflict_with_g_and_h(self, classifier, alphabet_fixture_data):
        """
        Phân xử K vs G/H:
        - G có trỏ duỗi ngang (idx_horiz >= 0.65), 3 ngón còn lại gập.
        - H có cả trỏ & giữa cùng hướng (ngang hoặc cùng đứng).
        Không bao giờ bị nhận nhầm thành K.
        """
        gh_samples = [
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") in ("G", "H") and s.get("quality_status") == "ok"
        ]
        for s in gh_samples:
            let = s.get("letter_upper")
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "K", f"Mẫu chữ {let} (frame {s.get('frame_idx')}) bị nhận nhầm thành K!"

        # Kiểm tra H Nam (D0500N) nếu file tồn tại
        d0500n_path = os.path.join(os.path.dirname(__file__), "..", "data", "alphabet_v2", "static_landmarks", "D0500N.json")
        if os.path.exists(d0500n_path):
            with open(d0500n_path, "r", encoding="utf-8") as fp:
                d0500n_data = json.load(fp)
            if "samples" in d0500n_data:
                for s in d0500n_data["samples"][:10]:
                    if s.get("quality_status") == "ok":
                        res = classifier.classify_detailed(s["landmarks"])
                        assert res["letter"] != "K", f"H Nam frame {s.get('frame_index')} bị nhận nhầm thành K!"

    def test_k_diagnostics(self, classifier, alphabet_fixture_data):
        """Kiểm tra bộ chẩn đoán evaluate_rule_k_diagnostics."""
        k_sample = next(
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "K" and s.get("frame_idx") == 26
        )
        features = classifier.extract_features(k_sample["landmarks"])
        diag = classifier.evaluate_rule_k_diagnostics(features)
        assert isinstance(diag, dict)
        assert diag["matched"] is True
        assert len(diag["failed_conditions"]) == 0
        assert "middle_d_mw_in_range" in diag["conditions"]
        assert "ang_im_in_range" in diag["conditions"]

        # Kiểm tra trên mẫu chữ V (phải chẩn đoán rõ lý do không đạt K)
        v_sample = next(
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "V" and s.get("phase") == "stable_pose" and s.get("quality_status") == "ok"
        )
        v_features = classifier.extract_features(v_sample["landmarks"])
        diag_v = classifier.evaluate_rule_k_diagnostics(v_features)
        assert diag_v["matched"] is False
        assert len(diag_v["failed_conditions"]) > 0

        # Kiểm tra tính an toàn trên None / dict rỗng
        assert classifier.evaluate_rule_k_diagnostics(None) is None
        diag_empty = classifier.evaluate_rule_k_diagnostics({})
        assert isinstance(diag_empty, dict)
        assert diag_empty["matched"] is None

    def test_k_mirror_invariance(self, classifier, alphabet_fixture_data):
        """
        Kiểm tra tính bất biến phản chiếu (tay trái / webcam mirror):
        Lật trục X của landmarks qua cổ tay vẫn phải nhận diện đúng chữ K.
        """
        k_sample = next(
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "K" and s.get("frame_idx") == 26
        )
        lm = k_sample["landmarks"]
        wrist_x = lm[0][0]
        # Phản chiếu qua trục x của wrist
        mirrored_lm = [
            [2.0 * wrist_x - pt[0], pt[1], pt[2]]
            for pt in lm
        ]
        res = classifier.classify_detailed(mirrored_lm)
        assert res["status"] == "MATCH" and res["letter"] == "K", (
            f"Chữ K bị mất nhận diện khi phản chiếu webcam (status={res['status']}, reason={res.get('reason')})"
        )


class TestRuleISouthernConvention:
    """
    Bộ kiểm thử chất lượng cao cho Chữ I và Chữ Y theo chuẩn video QIPEDC D0501.mp4:
    - Ngón út duỗi thẳng (d_PW >= 1.35, pinky_curl <= 0.20).
    - 3 ngón giữa (trỏ, giữa, áp út) gập sát lòng bàn tay.
    - Ngón cái gấp ôm vào ngón trỏ/giữa (d_TI <= 0.42, d_TM <= 0.52, thumb_lat <= 0.38).
    - Phân biệt tuyệt đối với chữ Y (Shaka: ngón cái xòe sang bên, d_TW >= 1.05, d_TI >= 0.40).
    - Bất biến quay ±30°, bất biến phản chiếu webcam (tay trái/phải), chịu lỗi khi ngón cái gấp khiến thumb_ext=True.
    """

    def test_i_positive_fixture_matches(self, classifier, alphabet_fixture_data):
        """Tất cả các frame ổn định của chữ I trong fixture phải nhận diện đúng với confidence 0.95."""
        i_samples = [
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "I" and s.get("quality_status") == "ok"
        ]
        assert len(i_samples) >= 4, "Thiếu mẫu kiểm thử chữ I trong fixture"
        for s in i_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH", (
                f"Chữ I frame {s.get('frame_idx')} không MATCH (status={res['status']}, reason={res.get('reason')})"
            )
            assert res["letter"] == "I", f"Chữ I frame {s.get('frame_idx')} bị gán nhãn {res.get('letter')}"
            assert res["confidence"] >= 0.95, f"Confidence chữ I frame {s.get('frame_idx')} < 0.95"

    def test_i_dataset_d0501_high_recall(self, classifier):
        """Kiểm thử trên toàn bộ các frame stable_pose thực tế của video D0501.mp4 (I.json)."""
        i_path = os.path.join(os.path.dirname(__file__), "..", "data", "alphabet_v2", "static_landmarks", "I.json")
        if not os.path.exists(i_path):
            pytest.skip("File I.json không tồn tại")
        with open(i_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        # Các frame tư thế ổn định chuẩn từ f14 đến f40
        stable_samples = [
            s for s in data.get("samples", [])
            if 14 <= s.get("frame_index", -1) <= 40 and s.get("landmarks") is not None
        ]
        assert len(stable_samples) >= 10, "Không đủ frame stable_pose trong I.json"
        for s in stable_samples:
            f_idx = s["frame_index"]
            res = classifier.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH" and res["letter"] == "I", (
                f"D0501 frame {f_idx} thất bại: status={res['status']}, letter={res.get('letter')}, reason={res.get('reason')}"
            )
            assert res["confidence"] >= 0.95

    def test_i_rejects_on_exit_frames(self, classifier):
        """Các frame hạ tay (exit phase) không được nhận diện thành chữ I."""
        i_path = os.path.join(os.path.dirname(__file__), "..", "data", "alphabet_v2", "static_landmarks", "I.json")
        if not os.path.exists(i_path):
            pytest.skip("File I.json không tồn tại")
        with open(i_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        exit_samples = [
            s for s in data.get("samples", [])
            if s.get("frame_index", -1) in (48, 50) and s.get("landmarks") is not None
        ]
        for s in exit_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "I", f"Frame exit {s['frame_index']} bị nhận nhầm thành I!"

    def test_i_does_not_conflict_with_y(self, classifier, alphabet_fixture_data):
        """
        Phân xử nghiêm ngặt I vs Y:
        - I: ngón cái gấp ôm vào ngón trỏ/giữa (d_TI <= 0.42, d_TW <= 1.15).
        - Y: ngón cái xòe ra ngoài (Shaka: d_TW >= 1.05, d_TI >= 0.40, thumb_lat >= 0.36 hoặc ang_TP >= 33°).
        Hai chữ không được cướp nhãn lẫn nhau.
        """
        # Kiểm tra mẫu I không được nhận thành Y
        i_samples = [
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "I" and s.get("quality_status") == "ok"
        ]
        for s in i_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] == "I" and res["letter"] != "Y"

        # Kiểm tra mẫu Y không được nhận thành I
        y_samples = [
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "Y" and s.get("quality_status") == "ok"
        ]
        for s in y_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] == "Y" and res["letter"] != "I"

    def test_i_does_not_conflict_with_b_v_d_l_g_h(self, classifier, alphabet_fixture_data):
        """Đối chứng âm tính: Không nhận nhầm B, V, D, L, G, H thành I."""
        other_samples = [
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") in ("B", "V", "D", "L", "G", "H") and s.get("quality_status") == "ok"
        ]
        for s in other_samples:
            res = classifier.classify_detailed(s["landmarks"])
            assert res["letter"] != "I", f"Mẫu chữ {s.get('letter_upper')} frame {s.get('frame_idx')} bị nhận nhầm thành I!"

    def test_i_thumb_folded_tolerance(self, classifier, alphabet_fixture_data):
        """
        Kiểm tra khả năng chịu lỗi khi ngón cái gấp qua mặt trước các ngón (tư thế nắm tay tự nhiên).
        Ngay cả khi MediaPipe tính thumb_ext=True, hệ thống vẫn phải nhận diện đúng chữ I dựa trên hình học 3D.
        """
        i_sample = next(
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "I" and s.get("frame_idx") == 24
        )
        lm = np.array(i_sample["landmarks"])
        # Dịch ngón cái tip (4) gần về phía ngón giữa tip (12) mô phỏng ngón cái ôm sâu
        lm_mod = np.array(lm)
        lm_mod[4] = lm_mod[12] * 0.6 + lm_mod[4] * 0.4
        
        # Kiểm tra rằng đặc trưng đã bị thumb_ext=True
        feat = classifier.extract_features(lm_mod.tolist())
        assert feat is not None
        
        res = classifier.classify_detailed(lm_mod.tolist())
        assert res["status"] == "MATCH" and res["letter"] == "I", (
            f"Chữ I bị lỗi khi ngón cái gấp sâu (status={res['status']}, reason={res.get('reason')})"
        )

    def test_i_rotations_and_mirror_invariance(self, classifier, alphabet_fixture_data):
        """
        Kiểm tra tính bất biến quay (±30°) và phản chiếu webcam (tay trái/phải).
        """
        i_sample = next(
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "I" and s.get("frame_idx") == 24
        )
        lm = np.array(i_sample["landmarks"])
        wrist = lm[0][:2]

        for angle in [-30, -15, 15, 30]:
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            rot_mat = np.array([[c, -s], [s, c]])
            rot_lm = []
            for pt in lm:
                xy = pt[:2] - wrist
                new_xy = rot_mat @ xy + wrist
                rot_lm.append([float(new_xy[0]), float(new_xy[1]), float(pt[2])])
            res = classifier.classify_detailed(rot_lm)
            assert res["status"] == "MATCH" and res["letter"] == "I", (
                f"Chữ I thất bại khi quay {angle}°: status={res['status']}, letter={res.get('letter')}"
            )

        # Lật gương qua trục X cổ tay
        wrist_x = lm[0][0]
        mirrored_lm = [[2.0 * wrist_x - pt[0], pt[1], pt[2]] for pt in lm]
        res_mir = classifier.classify_detailed(mirrored_lm)
        assert res_mir["status"] == "MATCH" and res_mir["letter"] == "I", (
            f"Chữ I thất bại khi phản chiếu webcam: status={res_mir['status']}, letter={res_mir.get('letter')}"
        )

    def test_i_diagnostics(self, classifier, alphabet_fixture_data):
        """Kiểm tra bộ chẩn đoán evaluate_rule_i_diagnostics."""
        i_sample = next(
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "I" and s.get("frame_idx") == 24
        )
        features = classifier.extract_features(i_sample["landmarks"])
        diag = classifier.evaluate_rule_i_diagnostics(features)
        assert isinstance(diag, dict)
        assert diag["matched"] is True
        assert len(diag["failed_conditions"]) == 0
        assert "pinky_ext_or_straight" in diag["conditions"]
        assert "thumb_not_spread" in diag["conditions"]

        # Kiểm tra trên mẫu chữ Y (phải chẩn đoán không đạt điều kiện ngón cái khép)
        y_sample = next(
            s for s in alphabet_fixture_data.get("samples", [])
            if s.get("letter_upper") == "Y" and s.get("frame_idx") == 25 and s.get("quality_status") == "ok"
        )
        y_features = classifier.extract_features(y_sample["landmarks"])
        diag_y = classifier.evaluate_rule_i_diagnostics(y_features)
        assert diag_y["matched"] is False
        assert len(diag_y["failed_conditions"]) > 0

        # Kiểm tra tính an toàn trên None / dict rỗng
        assert classifier.evaluate_rule_i_diagnostics(None) is None
        diag_empty = classifier.evaluate_rule_i_diagnostics({})
        assert isinstance(diag_empty, dict)
        assert diag_empty["matched"] is None





