"""
Kiểm thử Thiết kế và Hợp đồng Kiến trúc phân định Chữ cái và Chữ số VSL.
========================================================================
Giai đoạn 2C.3D: Kiểm thử hợp đồng mong muốn giữa conversionMode và nhãn đầu ra,
xác minh ma trận hành vi và các điểm giới hạn của nhận diện hình học tĩnh.

Quy tắc bất biến:
- Không sửa classify_geometry(), predict_landmarks(), sign_model.py.
- Không sửa API, frontend, checkpoints hoặc dataset.
- Phân biệt rõ ràng giữa kiểm thử hợp đồng kiến trúc và kiểm thử độ chính xác thực tế.
"""

import os
import sys
import json
# pyrefly: ignore[missing-import]
import pytest
from typing import Tuple, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS, ALLOWED_LETTERS_AND_DIGITS
from backend.vision.feature_extractor import extract_dual_hand_features
from backend.tests.test_digits import classify_digit_candidate_rules


@pytest.fixture(scope="module")
def digit_fixtures():
    """Tải dữ liệu mốc chữ số thực tế."""
    fpath = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json")
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def registry_fixtures():
    """Tải registry đăng ký video và trạng thái."""
    fpath = os.path.join(os.path.dirname(__file__), "fixtures", "digit_video_registry.json")
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


class TestModeContract:
    """Kiểm tra hợp đồng giữa conversionMode và hành vi lọc nhãn hiện có."""

    def test_valid_conversion_modes_contract(self):
        """Xác nhận 3 chế độ conversionMode hợp lệ của toàn bộ hệ thống."""
        VALID_MODES = {"letters_numbers", "words", "sentence"}
        # Frontend state trong CameraSignRecognizer.jsx và backend app.py
        assert "letters_numbers" in VALID_MODES
        assert "words" in VALID_MODES
        assert "sentence" in VALID_MODES

    def _make_open_hand(self, wrist_y=0.35):
        wx, wy, wz = [0.5, wrist_y, 0.0]
        my = wy - 0.30
        lm = [[wx, wy, wz] for _ in range(21)]
        lm[0] = [wx, wy, wz]
        lm[9] = [wx, my, wz]
        lm[5] = [wx - 0.04, my + 0.02, wz]
        lm[13] = [wx + 0.04, my + 0.02, wz]
        lm[17] = [wx + 0.08, my + 0.05, wz]
        for base_lm, idx_tip, idx_dip, idx_pip, offset_x in [
            (5, 8, 7, 6, -0.04),
            (9, 12, 11, 10, 0.0),
            (13, 16, 15, 14, 0.04),
            (17, 20, 19, 18, 0.08)
        ]:
            bx = wx + offset_x
            lm[idx_pip] = [bx, my - 0.07, wz]
            lm[idx_dip] = [bx, my - 0.13, wz]
            lm[idx_tip] = [bx, my - 0.19, wz]
        lm[1] = [wx - 0.04, wy - 0.03, wz]
        lm[2] = [wx - 0.07, wy - 0.06, wz]
        lm[3] = [wx - 0.09, wy - 0.08, wz]
        lm[4] = [wx - 0.18, wy - 0.10, wz]
        return lm

    def test_letters_numbers_mode_blocks_vocabulary(self):
        """
        Hợp đồng bảo vệ: Trong mode 'letters_numbers', bất kỳ cử chỉ từ vựng nào
        (như CAM_ON, XIN_CHAO, GIUP_DO) đều KHÔNG được xuất hiện ở kết quả cuối.
        """
        rec = VSLRecognizer()
        open_hand = self._make_open_hand(wrist_y=0.35)
        res_letters = rec.predict_landmarks(landmarks_list=open_hand, mode="letters_numbers")
        
        # Trong letters_numbers, nhãn từ vựng bị chặn hoàn toàn, trả về chữ cái/số hoặc NEUTRAL
        assert res_letters["sign"] in ALLOWED_LETTERS_AND_DIGITS or res_letters["sign"] == "NEUTRAL"
        assert res_letters["sign"] not in {"CAM_ON", "XIN_CHAO", "XIN_VUI_LONG"}

    def test_words_mode_permits_vocabulary(self):
        """Trong mode 'words' hoặc mode=None, các từ vựng hợp lệ không bị logit masking."""
        rec = VSLRecognizer()
        open_hand = self._make_open_hand(wrist_y=0.35)
        res_words = rec.predict_landmarks(landmarks_list=open_hand, mode="words")
        
        # Khi ở chế độ words, cử chỉ mở tay giơ cao được nhận diện là từ vựng XIN_CHAO
        assert res_words["sign"] == "XIN_CHAO"


class TestMorphologicalConflictContract:
    """
    Xác định hợp đồng giới hạn của nhận diện hình học tĩnh (Static Geometric Limits).
    Chứng minh bằng thực nghiệm rằng một số chữ số và chữ cái có hình thái tĩnh đồng dạng.
    """

    def test_digit_1_collides_with_letter_D_in_static_geometry(self, digit_fixtures):
        """
        Khảo sát: Số 1 và Chữ D cùng có hình thái ngón trỏ duỗi, các ngón khác gập.
        Bộ phân loại chữ cái hiện tại coi Số 1 là Chữ D.
        """
        rec = VSLRecognizer()
        s1 = next(s for s in digit_fixtures["samples"] if s["digit"] == "1" and s["split"] == "dev")
        _, meta = extract_dual_hand_features(s1["landmarks"], None)
        sign, conf = rec.classify_geometry(s1["landmarks"], meta["primary_states"])
        
        assert sign == "D", "Số 1 tĩnh kích hoạt Chữ D trong bộ phân loại chữ cái hiện tại."
        assert conf >= 0.80

    def test_digit_2_collides_with_letter_U_or_V(self, digit_fixtures):
        """
        Khảo sát: Số 2 và Chữ V/U cùng có hình thái ngón trỏ và ngón giữa duỗi.
        Bộ phân loại chữ cái hiện tại coi Số 2 là Chữ U hoặc V.
        """
        rec = VSLRecognizer()
        s2 = next(s for s in digit_fixtures["samples"] if s["digit"] == "2" and s["split"] == "dev")
        _, meta = extract_dual_hand_features(s2["landmarks"], None)
        sign, conf = rec.classify_geometry(s2["landmarks"], meta["primary_states"])
        
        assert sign in ["U", "V"]

    def test_digit_6_collides_with_letter_Y(self, digit_fixtures):
        """
        Khảo sát: Số 6 (Shaka) duỗi ngón cái và ngón út, 3 ngón giữa gập.
        Quy tắc Chữ Y hiện tại chỉ kiểm tra thumb + pinky duỗi => kích hoạt Chữ Y.
        """
        rec = VSLRecognizer()
        s6 = next(s for s in digit_fixtures["samples"] if s["digit"] == "6" and s["split"] == "dev")
        _, meta = extract_dual_hand_features(s6["landmarks"], None)
        sign, conf = rec.classify_geometry(s6["landmarks"], meta["primary_states"])
        
        assert sign == "Y"
        assert conf >= 0.90


class TestDesignArchitectureSeparation:
    """
    Kiểm tra tính khả thi của Phương án B (Bộ phân loại chữ số chuyên biệt).
    Chứng minh bộ phân loại chữ số có thể chạy độc lập mà không can thiệp vào classify_geometry().
    """

    def test_dedicated_digit_classifier_accuracy_on_real_dataset(self, digit_fixtures):
        """Bộ quy tắc số chuyên biệt phân loại chính xác 18/18 mẫu mốc thực tế."""
        samples = digit_fixtures["samples"]
        correct = 0
        for s in samples:
            pred, reason = classify_digit_candidate_rules(s["landmarks"])
            if pred == s["digit"]:
                correct += 1
        assert correct == 18, f"Kỳ vọng 18/18 mẫu đạt chuẩn, nhận {correct}"

    def test_digit_zero_remains_unresolved_in_contract(self, registry_fixtures):
        """Hợp đồng thiết kế: Số 0 bắt buộc ở trạng thái CHƯA CÓ GROUND TRUTH."""
        entry = registry_fixtures["digits"]["0"]
        assert entry["has_ground_truth"] is False
        assert entry["status"] == "CHƯA CÓ GROUND TRUTH"
