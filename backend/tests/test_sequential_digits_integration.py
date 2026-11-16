# -*- coding: utf-8 -*-
"""
Bộ kiểm thử Xác minh Loại bỏ ghép số liên tiếp và Xử lý Độc lập từng chữ số VSL
=============================================================================
Kiểm tra toàn diện:
1. Nhận diện riêng từng số từ 0 đến 9 tiếp tục hoạt động chính xác 100%.
2. Giơ 1 rồi 0 KHÔNG còn tự ghép thành 10 (phụ đề là ["1", "0"], sentence là "1 0").
3. Giơ 2 rồi 0 KHÔNG còn tự ghép thành 20 (phụ đề là ["2", "0"]).
4. Giơ 9 rồi 0 KHÔNG còn tự ghép thành 90 (phụ đề là ["9", "0"]).
5. Giơ cùng một số liên tục (ví dụ 1 rồi 1) không tạo chuỗi ghép 11 (phụ đề là ["1", "1"]).
6. TTS đọc số không có tiền tố "số" (trả về tts_text là "không", "một", "hai", ..., "chín", "mười").
7. API /api/recognize_landmarks trả về kết quả số độc lập, không có tiền tố "Số" trong new_token hay sentence.
"""

import json
import os
import sys
import unittest

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.venv/Lib/site-packages"))
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

from fastapi.testclient import TestClient

from backend.app import app
from backend.vision.recognizer import VSLRecognizer, DIGIT_TTS_MAP

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json")
FIXTURE_0_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_0_landmarks.json")


class TestSequentialDigitsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cls.samples_by_digit = {}
        for s in data.get("samples", []):
            d = s["digit"]
            if d not in cls.samples_by_digit:
                cls.samples_by_digit[d] = []
            cls.samples_by_digit[d].append(s["landmarks"])

        with open(FIXTURE_0_PATH, "r", encoding="utf-8") as f:
            d0_data = json.load(f)
        cls.samples_by_digit["0"] = [d0_data["samples"][0]["landmarks"]]

    def setUp(self):
        self.rec = VSLRecognizer()
        self.client = TestClient(app)

    def test_01_single_frame_digits_0_to_9_unaffected(self):
        """Khung hình độc lập của 0-9 tiếp tục nhận diện đúng 100%."""
        for d_str in [str(i) for i in range(10)]:
            self.assertIn(d_str, self.samples_by_digit)
            lm = self.samples_by_digit[d_str][0]
            res = self.rec.predict_landmarks(lm, mode="digits")
            self.assertEqual(res["sign"], d_str, f"Số {d_str} không nhận diện đúng")
            self.assertEqual(res["type"], "digit")
            self.assertGreater(res["confidence"], 0.70)

    def test_02_sequential_1_then_0_does_NOT_make_10(self):
        """Ký số 1 rồi hạ tay rồi ký số 0 KHÔNG ghép thành 10; phụ đề hiển thị '1 0'."""
        lm_1 = self.samples_by_digit["1"][0]
        lm_0 = self.samples_by_digit["0"][0]

        t = 1000.0
        # 1. Ký số 1
        for _ in range(12):
            self.rec.predict_landmarks(lm_1, mode="digits", timestamp=t)
            t += 0.05
        self.assertIn("1", self.rec.sentence_tokens)

        # 2. Thả tay (None)
        for _ in range(6):
            self.rec.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        # 3. Ký số 0
        for _ in range(12):
            res = self.rec.predict_landmarks(lm_0, mode="digits", timestamp=t)
            t += 0.05

        # Chờ thêm
        for _ in range(35):
            self.rec.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        # Phụ đề bỏ hoàn toàn từ "Số", hiển thị độc lập: ["1", "0"]
        self.assertEqual(self.rec.sentence_tokens, ["1", "0"])
        self.assertEqual(" ".join(self.rec.sentence_tokens), "1 0")
        self.assertNotIn("10", self.rec.sentence_tokens)
        self.assertNotIn("Số 10", self.rec.sentence_tokens)

    def test_03_sequential_2_then_0_does_NOT_make_20(self):
        """Ký số 2 rồi số 0 KHÔNG ghép thành 20; phụ đề là '2 0'."""
        lm_2 = self.samples_by_digit["2"][0]
        lm_0 = self.samples_by_digit["0"][0]

        t = 2000.0
        for _ in range(12):
            self.rec.predict_landmarks(lm_2, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(6):
            self.rec.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(12):
            self.rec.predict_landmarks(lm_0, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(35):
            self.rec.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        self.assertEqual(self.rec.sentence_tokens, ["2", "0"])
        self.assertEqual(" ".join(self.rec.sentence_tokens), "2 0")
        self.assertNotIn("20", self.rec.sentence_tokens)

    def test_04_sequential_9_then_0_does_NOT_make_90(self):
        """Ký số 9 rồi số 0 KHÔNG ghép thành 90; phụ đề là '9 0'."""
        lm_9 = self.samples_by_digit["9"][0]
        lm_0 = self.samples_by_digit["0"][0]

        t = 3000.0
        for _ in range(12):
            self.rec.predict_landmarks(lm_9, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(6):
            self.rec.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(12):
            self.rec.predict_landmarks(lm_0, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(35):
            self.rec.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        self.assertEqual(self.rec.sentence_tokens, ["9", "0"])
        self.assertEqual(" ".join(self.rec.sentence_tokens), "9 0")
        self.assertNotIn("90", self.rec.sentence_tokens)

    def test_05_repeating_same_digit_does_not_create_composed_number(self):
        """Ký cùng một số liên tục (1 rồi 1) tạo '1 1' không ghép thành 11."""
        lm_1 = self.samples_by_digit["1"][0]
        t = 4000.0
        for _ in range(12):
            self.rec.predict_landmarks(lm_1, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(6):
            self.rec.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(12):
            self.rec.predict_landmarks(lm_1, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(35):
            self.rec.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        self.assertEqual(self.rec.sentence_tokens, ["1", "1"])
        self.assertNotIn("11", self.rec.sentence_tokens)

    def test_06_tts_text_has_no_so_prefix(self):
        """TTS đọc số không có tiền tố 'số', chỉ đọc tên số ('không', 'một', ..., 'chín')."""
        expected_tts = {
            "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
            "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
        }
        for d_str, expected_word in expected_tts.items():
            lm = self.samples_by_digit[d_str][0]
            rec = VSLRecognizer()
            confirmed_res = None
            for i in range(12):
                res = rec.predict_landmarks(lm, mode="digits", timestamp=100.0 + i * 0.05)
                if res.get("confirmed"):
                    confirmed_res = res
            self.assertIsNotNone(confirmed_res, f"Không chốt được số {d_str}")
            self.assertEqual(confirmed_res["tts_text"], expected_word)
            self.assertFalse(confirmed_res["tts_text"].startswith("số"))
            self.assertFalse(confirmed_res["tts_text"].startswith("Số"))

    def test_07_api_recognize_landmarks_independent_digits(self):
        """API /api/recognize_landmarks trả về kết quả số độc lập, không có aggregator."""
        lm_3 = self.samples_by_digit["3"][0]
        payload = {
            "landmarks": lm_3,
            "mode": "digits",
            "timestamp": 1234.5
        }
        resp = self.client.post("/api/recognize_landmarks", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual(data["sign"], "3")
        self.assertEqual(data["type"], "digit")
        self.assertNotIn("aggregator", data)
        self.assertIn("tts_text", data)


if __name__ == "__main__":
    unittest.main()
