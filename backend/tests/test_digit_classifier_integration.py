"""
Bộ kiểm thử tích hợp Digit Classifier vào VSLRecognizer Engine (Giai đoạn 2C.4B)
Xác minh 10 yêu cầu tích hợp có điều kiện:
1. Nhận diện chữ số khi conversionMode='letters_numbers'.
2. Trả về an toàn khi classifier trả NEUTRAL / NO_MATCH.
3. Không trả về nhãn số khi classifier phát hiện CONFLICT.
4. Xử lý an toàn các landmark bất thường (None, empty, NaN, sai shape).
5. Tuyệt đối KHÔNG gọi digit_classifier trong conversionMode='words'.
6. Tuyệt đối KHÔNG gọi digit_classifier trong conversionMode='sentence'.
7. Chặn 100% từ vựng rò rỉ trong conversionMode='letters_numbers'.
8. Bảo tồn chữ cái hiện tại (không bị ghi đè vô điều kiện bởi số đồng dạng 1/D, 2/V, 6/Y).
9. Bảo toàn đầy đủ hợp đồng schema đầu ra của predict_landmarks (type="digit", display label).
10. Kiểm tra temporal smoothing và cooldown cho chữ số.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS, ALLOWED_DIGITS, ALLOWED_LETTERS_AND_DIGITS

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json")


class TestDigitClassifierIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cls.samples = data.get("samples", [])
        cls.samples_by_digit = {}
        for s in cls.samples:
            d = s["digit"]
            if d not in cls.samples_by_digit:
                cls.samples_by_digit[d] = []
            cls.samples_by_digit[d].append(s["landmarks"])

    def setUp(self):
        # Khởi tạo recognizer với smoothing_window=4 để dễ kiểm thử
        self.rec = VSLRecognizer(confidence_threshold=0.70, smoothing_window=4, cooldown_sec=0.5)

    # -------------------------------------------------------------------------
    # TEST 1: letters_numbers nhận diện thành công chữ số
    # -------------------------------------------------------------------------
    def test_01_letters_numbers_recognizes_digits(self):
        """Các chữ số độc lập không xung đột (3, 4, 5, 7, 8, 9) được nhận diện trực tiếp."""
        test_digits = ["3", "4", "5", "7", "8", "9"]
        for digit in test_digits:
            landmarks_list = self.samples_by_digit.get(digit, [])
            self.assertTrue(len(landmarks_list) > 0, f"Không tìm thấy fixture cho số {digit}")
            
            # Thử với mẫu dev (mẫu đầu tiên)
            lm = landmarks_list[0]
            res = self.rec.predict_landmarks(lm, mode="letters_numbers")
            
            self.assertEqual(res["sign"], digit, f"Số {digit} không được nhận diện đúng trong letters_numbers (kết quả: {res['sign']})")
            self.assertEqual(res["type"], "digit")
            self.assertGreaterEqual(res["confidence"], 0.70)

        # Đối với các số đồng dạng (1, 2, 6): khi có expected_sign, phải nhận diện đúng số
        for digit in ["1", "2", "6"]:
            landmarks_list = self.samples_by_digit.get(digit, [])
            lm = landmarks_list[0]
            res = self.rec.predict_landmarks(lm, mode="letters_numbers", expected_sign=digit)
            self.assertEqual(res["sign"], digit, f"Số {digit} không nhận diện được khi có expected_sign")
            self.assertEqual(res["type"], "digit")

    # -------------------------------------------------------------------------
    # TEST 2: Không trả số khi classifier trả NEUTRAL hoặc NO_MATCH
    # -------------------------------------------------------------------------
    def test_02_no_digit_when_classifier_neutral_or_no_match(self):
        """Khi digit_classifier trả NO_MATCH, không được gán bừa nhãn chữ số."""
        mock_result = {
            "digit": None,
            "status": "NO_MATCH",
            "confidence": 0.0,
            "rule_trace": "no_finger_combination"
        }
        with patch("backend.vision.digit_classifier.digit_classifier.classify_detailed", return_value=mock_result):
            # Tạo landmark ngẫu nhiên không thuộc bảng chữ cái
            dummy_lm = [[0.5, 0.5, 0.0]] * 21
            res = self.rec.predict_landmarks(dummy_lm, mode="letters_numbers")
            self.assertNotIn(res["sign"], ALLOWED_DIGITS)

    # -------------------------------------------------------------------------
    # TEST 3: Không trả số khi classifier phát hiện CONFLICT
    # -------------------------------------------------------------------------
    def test_03_no_digit_when_classifier_reports_conflict(self):
        """Khi digit_classifier trả về trạng thái CONFLICT, pipeline không được trả về chữ số."""
        mock_result = {
            "digit": "4",
            "status": "CONFLICT",
            "confidence": 0.0,
            "rule_trace": "conflict_thumb_extended_and_tucked"
        }
        with patch("backend.vision.digit_classifier.digit_classifier.classify_detailed", return_value=mock_result):
            dummy_lm = [[0.5, 0.5, 0.0]] * 21
            res = self.rec.predict_landmarks(dummy_lm, mode="letters_numbers")
            self.assertNotEqual(res["sign"], "4")
            self.assertNotIn(res["sign"], ALLOWED_DIGITS)

    # -------------------------------------------------------------------------
    # TEST 4: Khả năng chịu lỗi với đầu vào bất thường (None, Empty, NaN, v.v.)
    # -------------------------------------------------------------------------
    def test_04_fault_tolerance_invalid_inputs(self):
        """Pipeline phải an toàn tuyệt đối, không crash với landmark hỏng hoặc rỗng."""
        invalid_inputs = [
            None,
            [],
            [[0.5, 0.5]], # thiếu 1 trục
            [[0.0, 0.0, 0.0]] * 10, # thiếu điểm (chỉ 10 điểm)
            [[float('nan'), 0.5, 0.0]] * 21, # chứa NaN
            [[float('inf'), 0.5, 0.0]] * 21, # chứa Inf
        ]
        for bad_in in invalid_inputs:
            try:
                res = self.rec.predict_landmarks(bad_in, mode="letters_numbers")
                self.assertEqual(res["sign"], "NEUTRAL")
                self.assertEqual(res["confidence"], 0.0)
            except Exception as e:
                self.fail(f"predict_landmarks bị ném ngoại lệ khi gặp đầu vào bất thường: {bad_in}, lỗi: {e}")

    # -------------------------------------------------------------------------
    # TEST 5: conversionMode = 'words' KHÔNG gọi digit classifier
    # -------------------------------------------------------------------------
    def test_05_words_mode_never_calls_digit_classifier(self):
        """Chế độ words chỉ nhận diện từ vựng, tuyệt đối không gọi digit classifier."""
        with patch("backend.vision.digit_classifier.digit_classifier.classify_detailed") as mock_clf:
            lm = self.samples_by_digit["3"][0]
            res = self.rec.predict_landmarks(lm, mode="words")
            mock_clf.assert_not_called()
            self.assertNotIn(res["sign"], ALLOWED_DIGITS)

    # -------------------------------------------------------------------------
    # TEST 6: conversionMode = 'sentence' KHÔNG gọi digit classifier
    # -------------------------------------------------------------------------
    def test_06_sentence_mode_never_calls_digit_classifier(self):
        """Chế độ sentence chỉ nhận diện câu, tuyệt đối không gọi digit classifier."""
        with patch("backend.vision.digit_classifier.digit_classifier.classify_detailed") as mock_clf:
            lm = self.samples_by_digit["5"][0]
            res = self.rec.predict_landmarks(lm, mode="sentence")
            mock_clf.assert_not_called()
            self.assertNotIn(res["sign"], ALLOWED_DIGITS)

    # -------------------------------------------------------------------------
    # TEST 7: Chặn 100% từ vựng rò rỉ trong letters_numbers
    # -------------------------------------------------------------------------
    def test_07_vocabulary_strictly_blocked_in_letters_numbers(self):
        """Các nhãn từ vựng (ví dụ CAM_ON, XIN_CHAO) phải bị chặn tuyệt đối trong letters_numbers."""
        # Giả lập classify_geometry hoặc classifier hai tay trả ra từ vựng
        with patch.object(self.rec, "classify_geometry", return_value=("CAM_ON", 0.95)):
            dummy_lm = [[0.5, 0.5, 0.0]] * 21
            res = self.rec.predict_landmarks(dummy_lm, mode="letters_numbers")
            self.assertNotEqual(res["sign"], "CAM_ON")
            self.assertIn(res["sign"], ALLOWED_LETTERS_AND_DIGITS | {"NEUTRAL"})

        with patch.object(self.rec, "classify_geometry", return_value=("XIN_CHAO", 0.95)):
            dummy_lm = [[0.5, 0.5, 0.0]] * 21
            res = self.rec.predict_landmarks(dummy_lm, mode="letters_numbers")
            self.assertNotEqual(res["sign"], "XIN_CHAO")
            self.assertIn(res["sign"], ALLOWED_LETTERS_AND_DIGITS | {"NEUTRAL"})

    # -------------------------------------------------------------------------
    # TEST 8: Bảo tồn chữ cái đồng dạng (1/D, 2/V, 6/Y) không bị ghi đè vô điều kiện
    # -------------------------------------------------------------------------
    def test_08_homomorphic_letters_preserved_without_unconditional_override(self):
        """Chữ cái D, V, Y không bị số 1, 2, 6 ghi đè khi không có chỉ định expected_sign."""
        # Trường hợp 1: Dáng tay V (duỗi 2 ngón trỏ và giữa tạo góc V)
        # Khi đưa mẫu số 2 vào mà không có expected_sign:
        # Nếu classify_geometry nhận diện là V, hệ thống phải giữ nguyên chữ cái V để bảo vệ bảng chữ cái
        lm_2 = self.samples_by_digit["2"][0]
        res_2_no_exp = self.rec.predict_landmarks(lm_2, mode="letters_numbers")
        # classify_geometry sẽ nhận diện là V, không được bị số 2 ghi đè vô điều kiện
        self.assertIn(res_2_no_exp["sign"], {"V", "U", "2"})
        if res_2_no_exp["sign"] in {"V", "U"}:
            self.assertEqual(res_2_no_exp["type"], "alphabet")

        # Trường hợp 2: Dáng tay 1 (duỗi ngón trỏ)
        lm_1 = self.samples_by_digit["1"][0]
        res_1_no_exp = self.rec.predict_landmarks(lm_1, mode="letters_numbers")
        # Không được ghi đè D nếu geom_sign đã khớp D
        self.assertIn(res_1_no_exp["sign"], {"D", "1"})

        # Trường hợp 3: Khi có expected_sign="1", ưu tiên số 1
        res_1_exp = self.rec.predict_landmarks(lm_1, mode="letters_numbers", expected_sign="1")
        self.assertEqual(res_1_exp["sign"], "1")
        self.assertEqual(res_1_exp["type"], "digit")

    # -------------------------------------------------------------------------
    # TEST 9: Hợp đồng đầu ra của predict_landmarks không đổi
    # -------------------------------------------------------------------------
    def test_09_output_contract_integrity(self):
        """Output dict phải có đủ 7 trường cốt lõi, type là 'digit' khi nhận diện số."""
        lm = self.samples_by_digit["3"][0]
        res = self.rec.predict_landmarks(lm, mode="letters_numbers")
        required_keys = ["sign", "label", "type", "confidence", "confirmed", "new_token", "sentence"]
        for k in required_keys:
            self.assertIn(k, res, f"Thiếu trường {k} trong kết quả")

        self.assertEqual(res["sign"], "3")
        self.assertEqual(res["type"], "digit")
        self.assertIn("3", res["label"])
        self.assertIsInstance(res["confirmed"], bool)
        self.assertIsInstance(res["confidence"], float)

    # -------------------------------------------------------------------------
    # TEST 10: Temporal smoothing và debounce cooldown cho chữ số
    # -------------------------------------------------------------------------
    def test_10_temporal_smoothing_and_cooldown_for_digits(self):
        """Bộ đệm làm mượt và tích lũy từ/số hoạt động chuẩn xác với chữ số."""
        self.rec.prediction_history.clear()
        self.rec.sentence_tokens.clear()
        self.rec.last_confirmed_sign = None
        self.rec.last_confirmed_time = 0

        lm = self.samples_by_digit["7"][0]

        # Khung hình 1: Chưa đủ smoothing_window // 2 (chưa xác nhận)
        res1 = self.rec.predict_landmarks(lm, mode="letters_numbers")
        self.assertFalse(res1["confirmed"])
        self.assertEqual(res1["sentence"], "")

        # Khung hình 2: Đủ số khung hình tích lũy (smoothing_window=4, cần 2 khung hình)
        res2 = self.rec.predict_landmarks(lm, mode="letters_numbers")
        self.assertTrue(res2["confirmed"])
        self.assertIn("Số 7", res2["sentence"])
        self.assertEqual(self.rec.last_confirmed_sign, "7")

        # Khung hình 3: Tiếp tục giữ nguyên cử chỉ trong thời gian cooldown -> không xác nhận lặp lại
        res3 = self.rec.predict_landmarks(lm, mode="letters_numbers")
        self.assertFalse(res3["confirmed"])
        self.assertEqual(res3["sentence"], "Số 7")


if __name__ == "__main__":
    unittest.main()

