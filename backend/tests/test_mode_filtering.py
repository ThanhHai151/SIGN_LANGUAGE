"""
Unit tests for Stage 2B: Mode-based Label Filtering (Chế độ Chữ cái & Số).
Verifies:
1. Valid letters are recognized when mode="letters_numbers".
2. Conversational/vocabulary labels (XIN_CHAO, CAM_ON, XIN_VUI_LONG, THOI_QUEN, HANG_NGAY)
   CANNOT leak out when mode="letters_numbers", from both geometry rules and MLP.
3. Vocabulary labels produced by geometry (5 open fingers) are blocked and mapped to NEUTRAL.
4. Neutral state is correctly handled.
5. Backward compatibility: when mode is omitted or None, original behavior is preserved.
6. Sequence recognition (recognize_sequence) remains completely unchanged.
"""

import unittest
import numpy as np
import torch

from backend.models.sign_model import CLASSES
from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS

try:
    from backend.app import LandmarkRequest, SequenceRequest
    HAS_APP = True
except ImportError:
    HAS_APP = False


class TestModeFiltering(unittest.TestCase):
    def setUp(self):
        self.recognizer = VSLRecognizer(confidence_threshold=0.70)

    def _make_dummy_hand(self):
        """Tạo 21 landmark bàn tay chuẩn làm mốc cơ sở."""
        wx, wy, wz = [0.5, 0.8, 0.0]
        mx, my, mz = [0.5, 0.5, 0.0]
        lm = [[wx, wy, wz] for _ in range(21)]
        lm[0] = [wx, wy, wz]
        lm[9] = [mx, my, mz]
        lm[5] = [mx - 0.04, my + 0.02, mz]
        lm[13] = [mx + 0.04, my + 0.02, mz]
        lm[17] = [mx + 0.08, my + 0.05, mz]
        return lm

    def _make_5_fingers_open_landmarks(self, wrist_y=0.35):
        """Tạo landmark 5 ngón xòe (thường kích hoạt Rule 9: XIN_CHAO khi wrist_y < 0.40)."""
        wx, wy, wz = [0.5, wrist_y, 0.0]
        my = wy - 0.30
        lm = [[wx, wy, wz] for _ in range(21)]
        lm[0] = [wx, wy, wz]
        lm[9] = [wx, my, wz]
        lm[5] = [wx - 0.04, my + 0.02, wz]
        lm[13] = [wx + 0.04, my + 0.02, wz]
        lm[17] = [wx + 0.08, my + 0.05, wz]

        # 4 ngón duỗi thẳng lên trên (TIP.y < PIP.y)
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

        # Ngón cái duỗi ngang ra xa
        lm[1] = [wx - 0.04, wy - 0.03, wz]
        lm[2] = [wx - 0.07, wy - 0.06, wz]
        lm[3] = [wx - 0.09, wy - 0.08, wz]
        lm[4] = [wx - 0.18, wy - 0.10, wz] # TIP xa ngón trỏ

        return lm

    def _make_v_sign_landmarks(self):
        """Tạo landmark chữ V (ngón trỏ và ngón giữa duỗi xòe, các ngón khác gập)."""
        wx, wy, wz = [0.5, 0.8, 0.0]
        my = wy - 0.30
        lm = [[wx, wy, wz] for _ in range(21)]
        lm[0] = [wx, wy, wz]
        lm[9] = [wx, my, wz]
        lm[5] = [wx - 0.04, my + 0.02, wz]
        lm[13] = [wx + 0.04, my + 0.02, wz]
        lm[17] = [wx + 0.08, my + 0.05, wz]

        # Trỏ và giữa duỗi xòe chữ V
        lm[6] = [wx - 0.05, my - 0.07, wz]; lm[7] = [wx - 0.08, my - 0.13, wz]; lm[8] = [wx - 0.12, my - 0.20, wz]
        lm[10] = [wx + 0.02, my - 0.07, wz]; lm[11] = [wx + 0.05, my - 0.13, wz]; lm[12] = [wx + 0.09, my - 0.20, wz]

        # Áp út và út gập (TIP.y > PIP.y)
        lm[14] = [wx + 0.04, my + 0.04, wz]; lm[15] = [wx + 0.04, my + 0.06, wz]; lm[16] = [wx + 0.04, my + 0.08, wz]
        lm[18] = [wx + 0.08, my + 0.07, wz]; lm[19] = [wx + 0.08, my + 0.09, wz]; lm[20] = [wx + 0.08, my + 0.11, wz]

        # Ngón cái gập
        lm[1] = [wx - 0.04, wy - 0.03, wz]; lm[2] = [wx - 0.05, wy - 0.05, wz]
        lm[3] = [wx - 0.03, wy - 0.05, wz]; lm[4] = [wx - 0.02, wy - 0.04, wz]

        return lm

    def test_allowed_letters_set(self):
        """Xác nhận ALLOWED_LETTERS chứa chính xác 27 chữ cái VSL đầu tiên."""
        self.assertEqual(len(ALLOWED_LETTERS), 27)
        expected = set(CLASSES[:27])
        self.assertEqual(ALLOWED_LETTERS, expected)
        self.assertIn("A", ALLOWED_LETTERS)
        self.assertIn("Đ", ALLOWED_LETTERS)
        self.assertIn("Z", ALLOWED_LETTERS)
        self.assertNotIn("XIN_CHAO", ALLOWED_LETTERS)
        self.assertNotIn("THOI_QUEN", ALLOWED_LETTERS)
        self.assertNotIn("HANG_NGAY", ALLOWED_LETTERS)

    def test_valid_letter_recognized_in_letters_numbers_mode(self):
        """Chữ cái hợp lệ (chữ V) vẫn được nhận diện khi mode='letters_numbers'."""
        lm = self._make_v_sign_landmarks()
        result = self.recognizer.predict_landmarks(lm, mode="letters_numbers")
        self.assertEqual(result["sign"], "V")
        self.assertGreaterEqual(result["confidence"], 0.70)

    def test_vocabulary_sign_blocked_in_letters_numbers_mode(self):
        """
        Cử chỉ 5 ngón xòe (thường là XIN_CHAO/CAM_ON) bị chặn hoàn toàn
        khi mode='letters_numbers' và trả về NEUTRAL.
        """
        lm = self._make_5_fingers_open_landmarks(wrist_y=0.35)
        result = self.recognizer.predict_landmarks(lm, mode="letters_numbers")
        self.assertNotIn(result["sign"], ["XIN_CHAO", "CAM_ON", "XIN_VUI_LONG"])
        # Khi ở letters_numbers, nhãn từ vựng XIN_CHAO bị loại, chuyển thành số 5 hoặc NEUTRAL
        self.assertIn(result["sign"], {"NEUTRAL", "5"})
        if result["sign"] == "5":
            self.assertEqual(result["type"], "digit")
        else:
            self.assertEqual(result["confidence"], 0.0)

    def test_backward_compatibility_when_mode_is_omitted(self):
        """
        Khi client cũ KHÔNG truyền mode (mode=None), cử chỉ 5 ngón xòe
        vẫn trả về từ chào hỏi bình thường (bảo toàn hành vi cũ).
        """
        lm = self._make_5_fingers_open_landmarks(wrist_y=0.35)
        result = self.recognizer.predict_landmarks(lm, mode=None)
        # Ở chế độ cũ, 5 ngón xòe giơ cao (wrist_y < 0.40) ra XIN_CHAO
        self.assertEqual(result["sign"], "XIN_CHAO")
        self.assertGreaterEqual(result["confidence"], 0.85)

    def test_dual_hands_vocabulary_blocked_in_letters_numbers_mode(self):
        """Cử chỉ 2 tay từ vựng (GIUP_DO, YEU_THUONG) bị chặn khi mode='letters_numbers'."""
        lm1 = self._make_dummy_hand()
        lm2 = self._make_dummy_hand()
        multi_hands = [
            {"handedness": "Right", "landmarks": lm1},
            {"handedness": "Left", "landmarks": lm2}
        ]
        result = self.recognizer.predict_landmarks(lm1, multi_hands=multi_hands, mode="letters_numbers")
        self.assertNotIn(result["sign"], ["GIUP_DO", "YEU_THUONG", "GIA_DINH", "CAM_ON", "XIN_CHAO"])
        self.assertTrue(result["sign"] == "NEUTRAL" or result["sign"] in ALLOWED_LETTERS)

    def test_neutral_state_handling(self):
        """Trạng thái thiếu landmark hoặc rỗng xử lý đúng NEUTRAL."""
        result = self.recognizer.predict_landmarks([], mode="letters_numbers")
        self.assertEqual(result["sign"], "NEUTRAL")
        self.assertEqual(result["confidence"], 0.0)

    def test_landmark_request_schema_supports_mode(self):
        """Kiểm tra Pydantic model LandmarkRequest hỗ trợ trường mode tùy chọn."""
        if not HAS_APP:
            self.skipTest("fastapi not installed in current python environment")
        req_with_mode = LandmarkRequest(landmarks=[], mode="letters_numbers")
        self.assertEqual(req_with_mode.mode, "letters_numbers")

        req_without_mode = LandmarkRequest(landmarks=[])
        self.assertIsNone(req_without_mode.mode)

    def test_sequence_request_unchanged(self):
        """Kiểm tra SequenceRequest hoàn toàn không bị ảnh hưởng."""
        if not HAS_APP:
            self.skipTest("fastapi not installed in current python environment")
        dummy_seq = [[[0.0, 0.0, 0.0] for _ in range(75)] for _ in range(30)]
        req = SequenceRequest(sequence=dummy_seq, threshold=0.7)
        self.assertEqual(len(req.sequence), 30)
        self.assertEqual(req.threshold, 0.7)


if __name__ == "__main__":
    unittest.main()
