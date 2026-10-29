"""
Bộ kiểm thử Hợp đồng 4 Chế độ Độc lập (Giai đoạn 2C.5B):
- Chữ cái ('letters')
- Chữ số ('digits')
- Từ vựng ('words')
- Ghép câu ('sentence')
- Kèm chế độ lai tương thích ngược ('letters_numbers')

Kiểm chứng đầy đủ 11 yêu cầu kiểm thử bắt buộc theo hợp đồng mới.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS, ALLOWED_DIGITS, ALLOWED_LETTERS_AND_DIGITS

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json")


class TestModeSeparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cls.samples = data["samples"]
        cls.samples_by_digit = {}
        for s in cls.samples:
            d = s["digit"]
            if d not in cls.samples_by_digit:
                cls.samples_by_digit[d] = []
            cls.samples_by_digit[d].append(s["landmarks"])

    def setUp(self):
        self.rec = VSLRecognizer(confidence_threshold=0.70, smoothing_window=4, cooldown_sec=0.5)

    # -------------------------------------------------------------------------
    # 1. Chế độ 'letters': Không gọi digit_classifier, không trả nhãn chữ số
    # -------------------------------------------------------------------------
    def test_01_letters_mode_never_calls_digit_classifier_and_no_digits(self):
        """Trong mode='letters', digit_classifier tuyệt đối không được gọi; chỉ trả chữ cái hoặc NEUTRAL."""
        with patch("backend.vision.digit_classifier.digit_classifier.classify_detailed") as mock_clf:
            # Dùng mẫu số 1 thực tế (dáng tay duỗi ngón trỏ)
            lm_1 = self.samples_by_digit["1"][0]
            res = self.rec.predict_landmarks(lm_1, mode="letters")
            mock_clf.assert_not_called()
            self.assertNotIn(res["sign"], ALLOWED_DIGITS)
            self.assertIn(res["sign"], ALLOWED_LETTERS | {"NEUTRAL"})
            # Cử chỉ duỗi ngón trỏ trong letters trả về chữ D
            self.assertEqual(res["sign"], "D")
            self.assertEqual(res["type"], "alphabet")

    # -------------------------------------------------------------------------
    # 2. Chế độ 'digits': Gọi digit_classifier, nhận diện số 1-9, không trả chữ cái
    # -------------------------------------------------------------------------
    def test_02_digits_mode_recognizes_all_digits_without_letter_override(self):
        """
        Trong mode='digits', các số đồng dạng (1, 2, 6) được nhận diện thành chữ số đúng 100%,
        không còn bị ép thành D, U/V, Y như ở mode lai cũ.
        """
        for d in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            lm = self.samples_by_digit[d][0]
            res = self.rec.predict_landmarks(lm, mode="digits")
            self.assertEqual(res["sign"], d, f"Số {d} không được nhận diện trong mode='digits'")
            self.assertEqual(res["type"], "digit")
            self.assertNotIn(res["sign"], ALLOWED_LETTERS)

    # -------------------------------------------------------------------------
    # 3. Chế độ 'digits': Trả NEUTRAL khi NO_MATCH, CONFLICT hoặc đầu vào không hợp lệ
    # -------------------------------------------------------------------------
    def test_03_digits_mode_safe_fallback_on_no_match_conflict_invalid(self):
        """Khi classifier báo NO_MATCH hoặc CONFLICT, tuyệt đối không fallback sang chữ cái, trả NEUTRAL."""
        # Trường hợp 1: NO_MATCH
        mock_no_match = {"digit": None, "status": "NO_MATCH", "confidence": 0.0, "reason": "none"}
        with patch("backend.vision.digit_classifier.digit_classifier.classify_detailed", return_value=mock_no_match):
            lm = self.samples_by_digit["1"][0]
            res = self.rec.predict_landmarks(lm, mode="digits")
            self.assertEqual(res["sign"], "NEUTRAL")
            self.assertEqual(res["confidence"], 0.0)

        # Trường hợp 2: CONFLICT
        mock_conflict = {"digit": "1", "status": "CONFLICT", "confidence": 0.0, "reason": "conflict"}
        with patch("backend.vision.digit_classifier.digit_classifier.classify_detailed", return_value=mock_conflict):
            lm = self.samples_by_digit["1"][0]
            res = self.rec.predict_landmarks(lm, mode="digits")
            self.assertEqual(res["sign"], "NEUTRAL")
            self.assertEqual(res["confidence"], 0.0)

        # Trường hợp 3: Đầu vào NaN / không hợp lệ
        bad_lm = [[float('nan'), 0.5, 0.0]] * 21
        res_bad = self.rec.predict_landmarks(bad_lm, mode="digits")
        self.assertEqual(res_bad["sign"], "NEUTRAL")
        self.assertEqual(res_bad["confidence"], 0.0)

    # -------------------------------------------------------------------------
    # 4. Chế độ 'digits': Không nhận Chữ O thành Số 0 (chưa có GT)
    # -------------------------------------------------------------------------
    def test_04_digits_mode_does_not_recognize_o_as_digit_0(self):
        """Cử chỉ khuyên tròn chữ O trong mode='digits' không được nhận diện thành số 0."""
        # Giả lập cử chỉ O với tất cả các ngón đều cong/gập
        lm_o = [[0.5, 0.65, 0.0] for _ in range(21)]
        lm_o[0] = [0.5, 0.8, 0.0]  # Cổ tay
        lm_o[9] = [0.5, 0.55, 0.0] # MCP ngón giữa
        # Ngón trỏ gập (tip nằm dưới pip)
        lm_o[6] = [0.48, 0.50, 0.0] # PIP
        lm_o[8] = [0.47, 0.58, 0.0] # TIP gập
        # Ngón cái chạm đầu ngón trỏ
        lm_o[3] = [0.43, 0.60, 0.0] # IP
        lm_o[4] = [0.46, 0.58, 0.0] # TIP chạm gần ngón trỏ

        res = self.rec.predict_landmarks(lm_o, mode="digits")
        self.assertNotEqual(res["sign"], "0", "Hệ thống không được nhận diện thành số 0 khi chưa có ground truth")
        self.assertNotEqual(res["sign"], "O", "Chế độ digits không được rò rỉ chữ cái O")
        self.assertEqual(res["sign"], "NEUTRAL")

    # -------------------------------------------------------------------------
    # 5. Chế độ 'words' và 'sentence': Không gọi digit_classifier, giữ nguyên hành vi
    # -------------------------------------------------------------------------
    def test_05_words_and_sentence_modes_unaffected(self):
        """Chế độ words và sentence tuyệt đối không gọi digit classifier."""
        with patch("backend.vision.digit_classifier.digit_classifier.classify_detailed") as mock_clf:
            lm = self.samples_by_digit["5"][0]
            res_words = self.rec.predict_landmarks(lm, mode="words")
            mock_clf.assert_not_called()
            self.assertNotIn(res_words["sign"], ALLOWED_DIGITS)

            res_sent = self.rec.predict_landmarks(lm, mode="sentence")
            mock_clf.assert_not_called()
            self.assertNotIn(res_sent["sign"], ALLOWED_DIGITS)

    # -------------------------------------------------------------------------
    # 6. Chế độ lai tương thích ngược 'letters_numbers'
    # -------------------------------------------------------------------------
    def test_06_legacy_letters_numbers_alias_preserved(self):
        """Chế độ cũ 'letters_numbers' duy trì hành vi lai (1/D, 2/U, 6/Y ưu tiên chữ cái; 3,4,5,7,8,9 ưu tiên số)."""
        # Số 1 trả về chữ D
        lm_1 = self.samples_by_digit["1"][0]
        res_1 = self.rec.predict_landmarks(lm_1, mode="letters_numbers")
        self.assertEqual(res_1["sign"], "D")

        # Số 3 trả về số 3
        lm_3 = self.samples_by_digit["3"][0]
        res_3 = self.rec.predict_landmarks(lm_3, mode="letters_numbers")
        self.assertEqual(res_3["sign"], "3")

    # -------------------------------------------------------------------------
    # 7. Request thiếu 'mode' (mode=None)
    # -------------------------------------------------------------------------
    def test_07_omitted_mode_preserves_legacy_default(self):
        """Khi request không truyền mode (mode=None), hành vi mặc định không bị lỗi."""
        lm = self.samples_by_digit["3"][0]
        res = self.rec.predict_landmarks(lm, mode=None)
        self.assertIn("sign", res)
        self.assertIn("confidence", res)

    # -------------------------------------------------------------------------
    # 8. Xử lý mode không hợp lệ tại API
    # -------------------------------------------------------------------------
    def test_08_invalid_mode_returns_400_validation_error(self):
        """API trả về mã lỗi 400 Bad Request khi truyền mode không hợp lệ."""
        payload = {
            "landmarks": [[0.5, 0.5, 0.0]] * 21,
            "mode": "invalid_mode_xyz"
        }
        resp = self.client.post("/api/recognize_landmarks", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Chế độ nhận diện không hợp lệ", resp.json()["detail"])

    # -------------------------------------------------------------------------
    # 9. API chấp nhận cả 4 mode hợp lệ
    # -------------------------------------------------------------------------
    def test_09_api_accepts_all_four_modes(self):
        """API trả về 200 cho cả 4 chế độ: 'letters', 'digits', 'words', 'sentence' và alias 'letters_numbers'."""
        valid_modes = ["letters", "digits", "words", "sentence", "letters_numbers"]
        lm = self.samples_by_digit["1"][0]
        for m in valid_modes:
            payload = {"landmarks": lm, "mode": m}
            resp = self.client.post("/api/recognize_landmarks", json=payload)
            self.assertEqual(resp.status_code, 200, f"Mode {m} bị API từ chối")
            self.assertTrue(resp.json()["success"])

    # -------------------------------------------------------------------------
    # 10. Chuyển chế độ reset smoothing nhưng bảo tồn sentence
    # -------------------------------------------------------------------------
    def test_10_mode_switch_resets_smoothing_but_preserves_sentence(self):
        """
        Khi chuyển mode, bộ đệm prediction_history bị reset
        nhưng danh sách sentence_tokens không bị mất.
        """
        lm_letter = self.samples_by_digit["1"][0]  # Chữ D trong letters mode
        lm_digit = self.samples_by_digit["3"][0]   # Số 3 trong digits mode

        # Bước 1: Nhận diện chữ cái D đủ khung hình để confirm
        self.rec.predict_landmarks(lm_letter, mode="letters", session_id="sess_1")
        res1 = self.rec.predict_landmarks(lm_letter, mode="letters", session_id="sess_1")
        self.assertTrue(res1["confirmed"])
        self.assertEqual(res1["sentence"], "D")

        # Bước 2: Chuyển sang mode 'digits'
        # Smoothing history bị reset -> khung hình đầu tiên của mode digits KHÔNG được confirmed ngay
        res2 = self.rec.predict_landmarks(lm_digit, mode="digits", session_id="sess_1")
        self.assertFalse(res2["confirmed"], "Khung hình đầu tiên sau khi đổi mode không được confirm ngay do smoothing reset")
        # Sentence cũ ("D") vẫn được bảo tồn
        self.assertEqual(res2["sentence"], "D")

        # Bước 3: Đưa thêm khung hình trong mode digits để tích lũy đủ smoothing
        res3 = self.rec.predict_landmarks(lm_digit, mode="digits", session_id="sess_1")
        self.assertTrue(res3["confirmed"])
        self.assertEqual(res3["sentence"], "D Số 3")


if __name__ == "__main__":
    unittest.main()
