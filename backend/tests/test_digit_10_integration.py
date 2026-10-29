# -*- coding: utf-8 -*-
"""
Bộ kiểm thử tích hợp chuyên biệt cho nhận diện cử chỉ động Số 10 VSL và xử lý độc lập từng chữ số.
Bao gồm 12 kịch bản kiểm thử:
1. Video / chuỗi landmark thật từ D0539: Nhận diện thành công Số 10 qua predict_landmarks(..., mode="digits"), tts_text là "mười".
2. Dữ liệu số 1 tĩnh: Đảm bảo chỉ nhận diện là số 1, không nhận diện nhầm thành số 10, tts_text là "một" (không có tiền tố "số").
3. Không lặp token: Thực hiện cử chỉ số 10 chỉ ghi nhận đúng 1 token "10" vào phụ đề.
4. Xử lý độc lập số 1 và 1: Ký 1, hạ tay, ký 1 tiếp -> KHÔNG tự ghép thành 11; sentence_tokens là ['1', '1'].
5. Xử lý độc lập số 1 và 0: Ký 1, hạ tay, ký 0 -> KHÔNG tự ghép thành 10; sentence_tokens là ['1', '0'].
6. Xử lý độc lập số 2 và 0: Ký 2, hạ tay, ký 0 -> KHÔNG tự ghép thành 20; sentence_tokens là ['2', '0'].
7. Xử lý độc lập số 9 và 0: Ký 9, hạ tay, ký 0 -> KHÔNG tự ghép thành 90; sentence_tokens là ['9', '0'].
8. Gián đoạn / mất tracking: Chuỗi bị thiếu frame giữa chừng không sinh ra số 10 giả.
9. Timestamp thực tế không đều: Cử chỉ số 10 vẫn nhận diện đúng và không phát hành lặp.
10. Chuyển chế độ: Đổi sang letters, words, sentence rồi quay lại digits thì detector được reset sạch sẽ.
11. Độc lập chế độ: Chế độ letters nhận diện chữ cái bình thường, không bị ảnh hưởng bởi detector số 10.
12. Tích hợp API FastAPI: Endpoint /api/recognize_landmarks với mode="digits" trả về kết quả số 10 chính xác với tts_text="mười" và sentence không chứa "Số".
"""

import os
import sys
import json
import unittest

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.venv/Lib/site-packages"))
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

from fastapi.testclient import TestClient

from backend.vision.recognizer import VSLRecognizer
from backend.app import app


class TestDigit10Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")

        # Load fixture Số 10
        p10 = os.path.join(fixtures_dir, "real_digit_10_landmarks.json")
        with open(p10, "r", encoding="utf-8") as f:
            d10 = json.load(f)
        cls.d10_samples = [s for s in d10["samples"] if s.get("quality_status") == "ok"]

        # Load fixture Số 1..9
        p_digits = os.path.join(fixtures_dir, "real_digit_landmarks.json")
        with open(p_digits, "r", encoding="utf-8") as f:
            d_all = json.load(f)
        cls.digit_samples = {}
        for s in d_all["samples"]:
            digit = str(s["digit"])
            if digit not in cls.digit_samples:
                cls.digit_samples[digit] = s["landmarks"]

        # Load fixture Số 0
        p0 = os.path.join(fixtures_dir, "real_digit_0_landmarks.json")
        with open(p0, "r", encoding="utf-8") as f:
            d0 = json.load(f)
        cls.digit_samples["0"] = d0["samples"][0]["landmarks"]

        # Load fixture Chữ A thật chất lượng ok
        pa = os.path.join(fixtures_dir, "real_alphabet_landmarks.json")
        with open(pa, "r", encoding="utf-8") as f:
            da = json.load(f)
        ok_a = [s for s in da["samples"] if s.get("letter_upper") == "A" and s.get("quality_status") == "ok"]
        cls.letter_a_landmarks = ok_a[0]["landmarks"]

    def setUp(self):
        self.recognizer = VSLRecognizer()
        self.client = TestClient(app)

    # -------------------------------------------------------------
    # Kịch bản 1: Chuỗi landmark thật từ D0539 nhận diện thành công Số 10
    # -------------------------------------------------------------
    def test_scenario_01_real_digit_10_sequence_recognized(self):
        detected = False
        final_res = None
        for s in self.d10_samples:
            res = self.recognizer.predict_landmarks(
                landmarks_list=s["landmarks"],
                mode="digits",
                timestamp=s["timestamp_sec"]
            )
            if res.get("confirmed") and res.get("sign") == "10":
                detected = True
                final_res = res

        self.assertTrue(detected, "Chuỗi landmark thật D0539 phải nhận diện thành công Số 10")
        self.assertEqual(final_res["sign"], "10")
        self.assertEqual(final_res["new_token"], "10")
        self.assertEqual(final_res["type"], "digit")
        self.assertEqual(final_res["tts_text"], "mười")
        self.assertFalse(final_res["tts_text"].startswith("số"))
        self.assertGreaterEqual(final_res["confidence"], 0.85)

    # -------------------------------------------------------------
    # Kịch bản 2: Dữ liệu số 1 tĩnh không bị nhận thành số 10
    # -------------------------------------------------------------
    def test_scenario_02_static_digit_1_not_recognized_as_10(self):
        landmarks_1 = self.digit_samples["1"]
        detected_signs = []
        confirmed_res = None
        for i in range(15):
            t = 1.0 + i * 0.05
            res = self.recognizer.predict_landmarks(
                landmarks_list=landmarks_1,
                mode="digits",
                timestamp=t
            )
            if res.get("confirmed"):
                detected_signs.append(res.get("sign"))
                confirmed_res = res

        self.assertNotIn("10", detected_signs)
        self.assertIn("1", detected_signs)
        self.assertIn("1", self.recognizer.sentence_tokens)
        self.assertNotIn("10", self.recognizer.sentence_tokens)
        self.assertNotIn("Số 10", self.recognizer.sentence_tokens)
        self.assertEqual(confirmed_res["tts_text"], "một")
        self.assertFalse(confirmed_res["tts_text"].startswith("số"))

    # -------------------------------------------------------------
    # Kịch bản 3: Không lặp token khi thực hiện cử chỉ số 10
    # -------------------------------------------------------------
    def test_scenario_03_no_duplicate_token_for_digit_10(self):
        for s in self.d10_samples:
            self.recognizer.predict_landmarks(
                landmarks_list=s["landmarks"],
                mode="digits",
                timestamp=s["timestamp_sec"]
            )

        tokens = self.recognizer.sentence_tokens
        self.assertEqual(tokens, ["10"], f"Sentence tokens phải là ['10'], thực tế: {tokens}")
        self.assertNotIn("1", tokens)
        self.assertNotIn("11", tokens)

    # -------------------------------------------------------------
    # Kịch bản 4: Xử lý độc lập số 1 và số 1 (KHÔNG tự ghép thành 11)
    # -------------------------------------------------------------
    def test_scenario_04_independent_processing_1_then_1(self):
        landmarks_1 = self.digit_samples["1"]
        t = 100.0

        for _ in range(12):
            self.recognizer.predict_landmarks(landmarks_1, mode="digits", timestamp=t)
            t += 0.05

        for _ in range(6):
            self.recognizer.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        for _ in range(12):
            self.recognizer.predict_landmarks(landmarks_1, mode="digits", timestamp=t)
            t += 0.05

        for _ in range(35):
            self.recognizer.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        self.assertNotIn("11", self.recognizer.sentence_tokens)
        self.assertNotIn("Số 11", self.recognizer.sentence_tokens)
        self.assertEqual(self.recognizer.sentence_tokens, ["1", "1"])
        self.assertEqual(" ".join(self.recognizer.sentence_tokens), "1 1")

    # -------------------------------------------------------------
    # Kịch bản 5: Xử lý độc lập số 1 và số 0 (KHÔNG tự ghép thành 10)
    # -------------------------------------------------------------
    def test_scenario_05_independent_processing_1_then_0(self):
        landmarks_1 = self.digit_samples["1"]
        landmarks_0 = self.digit_samples["0"]
        t = 200.0

        for _ in range(12):
            self.recognizer.predict_landmarks(landmarks_1, mode="digits", timestamp=t)
            t += 0.05

        for _ in range(6):
            self.recognizer.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        for _ in range(12):
            self.recognizer.predict_landmarks(landmarks_0, mode="digits", timestamp=t)
            t += 0.05

        for _ in range(35):
            self.recognizer.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        # Giơ 1 rồi 0 không còn tự ghép thành 10, mà tạo 2 token độc lập '1' và '0'
        self.assertEqual(self.recognizer.sentence_tokens, ["1", "0"])
        self.assertEqual(" ".join(self.recognizer.sentence_tokens), "1 0")
        self.assertNotIn("10", self.recognizer.sentence_tokens)

    # -------------------------------------------------------------
    # Kịch bản 6: Xử lý độc lập số 2 và số 0 (KHÔNG tự ghép thành 20)
    # -------------------------------------------------------------
    def test_scenario_06_independent_processing_2_then_0(self):
        landmarks_2 = self.digit_samples["2"]
        landmarks_0 = self.digit_samples["0"]
        t = 300.0

        for _ in range(12):
            self.recognizer.predict_landmarks(landmarks_2, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(6):
            self.recognizer.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(12):
            self.recognizer.predict_landmarks(landmarks_0, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(35):
            self.recognizer.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        self.assertEqual(self.recognizer.sentence_tokens, ["2", "0"])
        self.assertEqual(" ".join(self.recognizer.sentence_tokens), "2 0")
        self.assertNotIn("20", self.recognizer.sentence_tokens)

    # -------------------------------------------------------------
    # Kịch bản 7: Xử lý độc lập số 9 và số 0 (KHÔNG tự ghép thành 90)
    # -------------------------------------------------------------
    def test_scenario_07_independent_processing_9_then_0(self):
        landmarks_9 = self.digit_samples["9"]
        landmarks_0 = self.digit_samples["0"]
        t = 400.0

        for _ in range(12):
            self.recognizer.predict_landmarks(landmarks_9, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(6):
            self.recognizer.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(12):
            self.recognizer.predict_landmarks(landmarks_0, mode="digits", timestamp=t)
            t += 0.05
        for _ in range(35):
            self.recognizer.predict_landmarks(None, mode="digits", timestamp=t)
            t += 0.05

        self.assertEqual(self.recognizer.sentence_tokens, ["9", "0"])
        self.assertEqual(" ".join(self.recognizer.sentence_tokens), "9 0")
        self.assertNotIn("90", self.recognizer.sentence_tokens)

    # -------------------------------------------------------------
    # Kịch bản 8: Gián đoạn / mất tracking không tạo Số 10 giả
    # -------------------------------------------------------------
    def test_scenario_08_interrupted_tracking_no_false_10(self):
        for s in self.d10_samples[:2]:
            self.recognizer.predict_landmarks(s["landmarks"], mode="digits", timestamp=s["timestamp_sec"])

        for i in range(20):
            res = self.recognizer.predict_landmarks(None, mode="digits", timestamp=1.0 + i * 0.05)
            self.assertNotEqual(res.get("sign"), "10")

        self.assertNotIn("10", self.recognizer.sentence_tokens)

    # -------------------------------------------------------------
    # Kịch bản 9: Timestamp không đều (jitter) vẫn nhận diện đúng và không lặp
    # -------------------------------------------------------------
    def test_scenario_09_irregular_timestamps_no_duplicate(self):
        confirmed_count = 0
        base_t = 10.0
        intervals = [0.03, 0.08, 0.04, 0.07, 0.05, 0.06, 0.03, 0.09]
        curr_t = base_t
        for i, s in enumerate(self.d10_samples):
            if i > 0:
                curr_t += intervals[(i - 1) % len(intervals)]
            res = self.recognizer.predict_landmarks(
                s["landmarks"],
                mode="digits",
                timestamp=curr_t
            )
            if res.get("confirmed") and res.get("sign") == "10":
                confirmed_count += 1

        self.assertEqual(confirmed_count, 1, "Cử chỉ số 10 phải được chốt chính xác đúng 1 lần duy nhất")
        self.assertEqual(self.recognizer.sentence_tokens, ["10"])

    # -------------------------------------------------------------
    # Kịch bản 10: Chuyển chế độ letters/words/sentence reset sạch detector
    # -------------------------------------------------------------
    def test_scenario_10_mode_switch_clears_detector_state(self):
        for s in self.d10_samples[:4]:
            self.recognizer.predict_landmarks(s["landmarks"], mode="digits", timestamp=s["timestamp_sec"])
        self.assertGreater(len(self.recognizer.digit_10_detector.buffer), 0)

        self.recognizer.predict_landmarks(self.letter_a_landmarks, mode="letters", timestamp=2.0)
        self.assertEqual(len(self.recognizer.digit_10_detector.buffer), 0)

        self.recognizer.predict_landmarks(None, mode="digits", timestamp=2.5)
        self.assertEqual(len(self.recognizer.digit_10_detector.buffer), 0)
        self.assertIsNone(self.recognizer.last_confirmed_sign)

    # -------------------------------------------------------------
    # Kịch bản 11: Độc lập chế độ: mode letters nhận diện A bình thường
    # -------------------------------------------------------------
    def test_scenario_11_letters_mode_independence(self):
        detected_letter = None
        for i in range(12):
            res = self.recognizer.predict_landmarks(
                self.letter_a_landmarks,
                mode="letters",
                timestamp=1.0 + i * 0.05
            )
            if res.get("confirmed"):
                detected_letter = res.get("sign")

        self.assertEqual(detected_letter, "A")
        self.assertEqual(self.recognizer.sentence_tokens, ["A"])
        self.assertNotIn("10", self.recognizer.sentence_tokens)
        self.assertNotIn("1", self.recognizer.sentence_tokens)

    # -------------------------------------------------------------
    # Kịch bản 12: Tích hợp API FastAPI /api/recognize_landmarks
    # -------------------------------------------------------------
    def test_scenario_12_fastapi_endpoint_recognize_landmarks(self):
        detected_api = False
        api_res = None
        for s in self.d10_samples:
            response = self.client.post(
                "/api/recognize_landmarks",
                json={
                    "landmarks": s["landmarks"],
                    "mode": "digits",
                    "timestamp": s["timestamp_sec"],
                    "session_id": "test_api_session_digit10"
                }
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data.get("success"))
            payload = data.get("data", {})
            if payload.get("confirmed") and payload.get("sign") == "10":
                detected_api = True
                api_res = payload

        self.assertTrue(detected_api, "API /api/recognize_landmarks phải trả về Số 10 thành công")
        self.assertEqual(api_res["sign"], "10")
        self.assertEqual(api_res["new_token"], "10")
        self.assertEqual(api_res["type"], "digit")
        self.assertEqual(api_res["tts_text"], "mười")
        self.assertNotIn("aggregator", api_res)
        self.assertEqual(api_res["sentence"], "10")


if __name__ == "__main__":
    unittest.main()
