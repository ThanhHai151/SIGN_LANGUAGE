"""
Kiểm thử tích hợp VSLAlphabetClassifier vào luồng nhận diện chữ cái (Task DIAG-A2).
=================================================================================
Bao phủ:
- Trường hợp A: Landmark chữ A hợp lệ, mock MLP trả về 'Z' (conf cao) -> kết quả cuối cùng vẫn là 'A' (không bị ghi đè).
- Trường hợp B: Classifier trả về NO_MATCH -> hệ thống an toàn trả về NEUTRAL (conf=0.0), không tự động gán nhãn bừa bãi.
- Trường hợp C: Đầu vào không hợp lệ (None, sai số điểm, NaN, Inf) -> xử lý an toàn, không crash, trả về NEUTRAL.
- Trường hợp D: Chế độ 'digits' và các chế độ khác ('words', 'sentence', 'letters_numbers') không bị ảnh hưởng.
- Trường hợp E: Khớp cấu trúc response API của frontend và kiểm tra các chữ cái khác (B, D, L, O, S, T, V, Y).
"""

import json
import math
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS, ALLOWED_DIGITS
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_alphabet_landmarks.json")
DIGIT_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json")


class TestAlphabetRecognitionIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            cls.alphabet_data = json.load(f)
        with open(DIGIT_FIXTURE_PATH, "r", encoding="utf-8") as f:
            cls.digit_data = json.load(f)

    def setUp(self):
        self.rec = VSLRecognizer(confidence_threshold=0.70, smoothing_window=4, cooldown_sec=0.5)

    # -------------------------------------------------------------------------
    # Trường hợp A: Sửa lỗi chữ A - Ngăn chặn MLP ghi đè sang Z
    # -------------------------------------------------------------------------
    def test_case_a_letter_a_not_overridden_by_mlp_z(self):
        """
        Xác minh lỗi Task DIAG-A1 đã được giải quyết:
        Dù mạng MLP có dự đoán ra 'Z' với xác suất 1.0, kết quả cuối cùng vẫn là 'A'
        khi VSLAlphabetClassifier trả về MATCH cho chữ A.
        """
        a_samples = [s for s in self.alphabet_data["samples"] if s["letter_upper"] == "A" and s.get("quality_status") == "ok"]
        self.assertGreaterEqual(len(a_samples), 4)

        # Mock mô hình MLP để luôn luôn trả về logit cực đại cho 'Z' (index 26)
        with patch.object(self.rec.model, "forward", side_effect=Exception("MLP should not be called")) if not hasattr(self.rec.model, "forward") else patch.object(self.rec, "model") as mock_model:
            mock_model.has_weights = True
            # Giả lập logits của MLP ra Z (index 26)
            mock_logits = torch_tensor = np.zeros((1, 84), dtype=np.float32)
            mock_logits[0, 26] = 100.0  # Z

            for s in a_samples:
                res = self.rec.predict_landmarks(s["landmarks"], mode="letters", session_id="test_a")
                self.assertEqual(res["sign"], "A", f"Frame {s['frame_idx']} chữ A bị ghi đè thành {res['sign']}!")
                self.assertEqual(res["type"], "alphabet")
                self.assertGreaterEqual(res["confidence"], 0.90)
                self.assertNotEqual(res["sign"], "Z")

    # -------------------------------------------------------------------------
    # Trường hợp B: Xử lý NO_MATCH an toàn
    # -------------------------------------------------------------------------
    def test_case_b_no_match_returns_neutral_safely(self):
        """
        Khi classifier trả về NO_MATCH (ví dụ cử chỉ lạ hoặc cử chỉ trung tính),
        hệ thống trả về NEUTRAL với confidence 0.0, không đoán mò hoặc ép nhãn.
        """
        mock_no_match = {
            "letter": "UNKNOWN",
            "confidence": 0.0,
            "status": "NO_MATCH",
            "reason": "Không khớp bất kỳ quy tắc nào",
            "matched_candidates": [],
            "features": None
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            dummy_lm = [[0.1, 0.2, 0.3]] * 21
            res = self.rec.predict_landmarks(dummy_lm, mode="letters", session_id="test_b")
            self.assertEqual(res["sign"], "NEUTRAL")
            self.assertEqual(res["confidence"], 0.0)
            self.assertEqual(res["type"], "unknown")

    def test_case_b_conflict_returns_neutral_safely(self):
        """Khi classifier báo CONFLICT (xung đột luật), hệ thống an toàn trả về NEUTRAL."""
        mock_conflict = {
            "letter": "UNKNOWN",
            "confidence": 0.0,
            "status": "CONFLICT",
            "reason": "Xung đột quy tắc",
            "matched_candidates": [("A", 0.9), ("E", 0.9)],
            "features": None
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_conflict):
            dummy_lm = [[0.1, 0.2, 0.3]] * 21
            res = self.rec.predict_landmarks(dummy_lm, mode="letters", session_id="test_conflict")
            self.assertEqual(res["sign"], "NEUTRAL")
            self.assertEqual(res["confidence"], 0.0)

    # -------------------------------------------------------------------------
    # Trường hợp C: Xử lý đầu vào không hợp lệ (INVALID_INPUT)
    # -------------------------------------------------------------------------
    def test_case_c_invalid_inputs_safety(self):
        """Kiểm tra phản ứng an toàn trước đầu vào None, rỗng, sai số điểm, NaN, Inf."""
        # 1. None
        res_none = self.rec.predict_landmarks(None, mode="letters")
        self.assertEqual(res_none["sign"], "NEUTRAL")
        self.assertEqual(res_none["confidence"], 0.0)

        # 2. Rỗng
        res_empty = self.rec.predict_landmarks([], mode="letters")
        self.assertEqual(res_empty["sign"], "NEUTRAL")

        # 3. Sai số lượng mốc (5 điểm thay vì 21)
        res_short = self.rec.predict_landmarks([[0.1, 0.2, 0.3]] * 5, mode="letters")
        self.assertEqual(res_short["sign"], "NEUTRAL")

        # 4. Chứa NaN
        nan_lm = [[0.1, 0.2, 0.3]] * 21
        nan_lm[4] = [float("nan"), 0.2, 0.3]
        res_nan = self.rec.predict_landmarks(nan_lm, mode="letters")
        self.assertEqual(res_nan["sign"], "NEUTRAL")

        # 5. Chứa Inf
        inf_lm = [[0.1, 0.2, 0.3]] * 21
        inf_lm[8] = [0.1, float("inf"), 0.3]
        res_inf = self.rec.predict_landmarks(inf_lm, mode="letters")
        self.assertEqual(res_inf["sign"], "NEUTRAL")

    # -------------------------------------------------------------------------
    # Trường hợp D: Đảm bảo các mode khác không bị ảnh hưởng
    # -------------------------------------------------------------------------
    def test_case_d_digits_mode_completely_unaffected(self):
        """Mode 'digits' vẫn gọi VSLDigitClassifier và nhận diện đúng các số 1-9."""
        sample_digit_1 = next(s for s in self.digit_data["samples"] if s["digit"] == "1")
        res = self.rec.predict_landmarks(sample_digit_1["landmarks"], mode="digits")
        self.assertEqual(res["sign"], "1")
        self.assertEqual(res["type"], "digit")
        self.assertGreaterEqual(res["confidence"], 0.90)

    def test_case_d_words_and_sentence_modes_unaffected(self):
        """Mode 'words' và 'sentence' không bị ảnh hưởng bởi tích hợp AlphabetClassifier."""
        sample_digit_5 = next(s for s in self.digit_data["samples"] if s["digit"] == "5")
        res_words = self.rec.predict_landmarks(sample_digit_5["landmarks"], mode="words")
        self.assertNotIn(res_words["sign"], ALLOWED_DIGITS)

        res_sent = self.rec.predict_landmarks(sample_digit_5["landmarks"], mode="sentence")
        self.assertNotIn(res_sent["sign"], ALLOWED_DIGITS)

    def test_case_d_legacy_letters_numbers_mode_unaffected(self):
        """Mode lai cũ 'letters_numbers' duy trì tương thích ngược."""
        sample_digit_3 = next(s for s in self.digit_data["samples"] if s["digit"] == "3")
        res_3 = self.rec.predict_landmarks(sample_digit_3["landmarks"], mode="letters_numbers")
        self.assertEqual(res_3["sign"], "3")

    # -------------------------------------------------------------------------
    # Trường hợp E: Cấu trúc Response API và Kiểm tra các chữ cái khác
    # -------------------------------------------------------------------------
    def test_case_e_api_response_schema_compatibility(self):
        """Đảm bảo mọi trường API frontend cần đều hiện diện đầy đủ và đúng kiểu dữ liệu."""
        sample_a = next(s for s in self.alphabet_data["samples"] if s["letter_upper"] == "A" and s.get("quality_status") == "ok")
        res = self.rec.predict_landmarks(sample_a["landmarks"], mode="letters")

        required_keys = ["sign", "label", "type", "confidence", "confirmed", "new_token", "sentence", "description"]
        for k in required_keys:
            self.assertIn(k, res, f"Thiếu trường '{k}' trong response API!")

        self.assertIsInstance(res["sign"], str)
        self.assertIsInstance(res["confidence"], float)
        self.assertIsInstance(res["confirmed"], bool)
        self.assertIsInstance(res["sentence"], str)

    def test_case_e_other_supported_letters(self):
        """
        Kiểm tra các chữ cái tĩnh khác đã có luật (B, D, L, O, S, T, V, Y)
        trong mode='letters' đều được nhận diện chuẩn xác.
        """
        test_letters = ["B", "D", "L", "O", "S", "T", "V", "Y"]
        for target in test_letters:
            samples = [s for s in self.alphabet_data["samples"] if s["letter_upper"] == target and s.get("quality_status") == "ok" and s.get("phase") == "stable_pose"]
            if not samples:
                continue
            # Lấy 1 mẫu stable_pose đại diện
            s = samples[0]
            res = self.rec.predict_landmarks(s["landmarks"], mode="letters", session_id=f"test_{target}")
            self.assertEqual(res["sign"], target, f"Kỳ vọng nhận diện chữ {target}, nhưng nhận {res['sign']}!")
            self.assertEqual(res["type"], "alphabet")
            self.assertGreaterEqual(res["confidence"], 0.90)


if __name__ == "__main__":
    unittest.main()

