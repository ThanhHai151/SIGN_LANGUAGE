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
        self.assertEqual(res3["sentence"], "D 3")

    # -------------------------------------------------------------------------
    # 11. Chu kỳ chuyển đổi toàn diện letters -> digits -> words -> sentence -> letters
    # -------------------------------------------------------------------------
    def test_11_full_cycle_mode_switching_sequence(self):
        """
        Kiểm tra chuyển đổi liên tục qua toàn bộ 4 chế độ:
        Mỗi lần đổi mode, bộ đệm làm mượt được reset độc lập,
        không có bất kỳ frame của mode trước nào kích hoạt confirm nhầm cho mode sau.
        """
        lm_letter = self.samples_by_digit["1"][0]
        lm_digit = self.samples_by_digit["3"][0]
        lm_word = self.samples_by_digit["5"][0]

        # 1. letters mode
        self.rec.predict_landmarks(lm_letter, mode="letters", session_id="cycle_sess")
        self.assertEqual(len(self.rec.prediction_history), 1)

        # 2. Chuyển sang digits mode -> buffer bị xóa và chỉ chứa 1 frame của digits
        self.rec.predict_landmarks(lm_digit, mode="digits", session_id="cycle_sess")
        self.assertEqual(len(self.rec.prediction_history), 1)
        self.assertEqual(self.rec.prediction_history[0][0], "3")

        # 3. Chuyển sang words mode -> buffer bị xóa và chỉ chứa 1 frame của words
        self.rec.predict_landmarks(lm_word, mode="words", session_id="cycle_sess")
        self.assertEqual(len(self.rec.prediction_history), 1)

        # 4. Chuyển sang sentence mode -> buffer bị xóa và chỉ chứa 1 frame của sentence
        self.rec.predict_landmarks(lm_word, mode="sentence", session_id="cycle_sess")
        self.assertEqual(len(self.rec.prediction_history), 1)

        # 5. Quay lại letters mode -> buffer bị xóa và chỉ chứa 1 frame của letters
        self.rec.predict_landmarks(lm_letter, mode="letters", session_id="cycle_sess")
        self.assertEqual(len(self.rec.prediction_history), 1)
        self.assertEqual(self.rec.prediction_history[0][0], "D")

    # -------------------------------------------------------------------------
    # 12. Cô lập trạng thái phiên nhận diện (Session Isolation)
    # -------------------------------------------------------------------------
    def test_12_session_isolation_smoothing(self):
        """
        Phiên A đổi mode không làm xóa hoặc ảnh hưởng đến bộ đệm mode của Phiên B.
        """
        lm_d = self.samples_by_digit["3"][0]
        # Phiên B đăng ký mode 'digits'
        self.rec.predict_landmarks(lm_d, mode="digits", session_id="session_B")
        self.assertEqual(self.rec._session_last_modes.get("session_B"), "digits")

        # Phiên A đăng ký mode 'letters' rồi đổi sang 'digits'
        self.rec.predict_landmarks(lm_d, mode="letters", session_id="session_A")
        self.assertEqual(self.rec._session_last_modes.get("session_A"), "letters")
        self.rec.predict_landmarks(lm_d, mode="digits", session_id="session_A")
        self.assertEqual(self.rec._session_last_modes.get("session_A"), "digits")

        # Mode của Phiên B vẫn được bảo toàn nguyên vẹn
        self.assertEqual(self.rec._session_last_modes.get("session_B"), "digits")



    # -------------------------------------------------------------------------
    # 13. Mô phỏng chuyển chế độ trực tiếp khi Camera đang chạy (Live Switching)
    # -------------------------------------------------------------------------
    def test_13_live_camera_mode_switching_letters_to_digits(self):
        """
        Mô phỏng chuỗi frame camera trực tiếp:
        - 5 frame đầu đang ở mode='letters' nhận diện chữ cái.
        - Người dùng đổi sang mode='digits' trong lúc camera tiếp tục đẩy frame.
        - Các frame ngay sau đó chuyển sang mode='digits' tức thì, không bị kẹt nhãn chữ cái.
        """
        lm_finger = self.samples_by_digit["1"][0]  # ngón trỏ duỗi: mode letters là D, mode digits là 1
        
        # 1. 5 frame ở mode letters
        for _ in range(5):
            res_letter = self.rec.predict_landmarks(lm_finger, mode="letters", session_id="live_cam")
            self.assertEqual(res_letter["type"], "alphabet")
            self.assertEqual(res_letter["sign"], "D")
            self.assertNotIn(res_letter["sign"], ALLOWED_DIGITS)

        # 2. Người dùng chuyển mode sang 'digits' trên cùng session live camera
        res_digit = self.rec.predict_landmarks(lm_finger, mode="digits", session_id="live_cam")
        self.assertEqual(res_digit["type"], "digit")
        self.assertEqual(res_digit["sign"], "1")
        self.assertNotIn(res_digit["sign"], ALLOWED_LETTERS)

    # -------------------------------------------------------------------------
    # 14. Bảo vệ chống Stale State & Out-of-Order: Reset debounce và history
    # -------------------------------------------------------------------------
    def test_14_mode_switch_clears_prediction_history_and_last_confirmed_sign(self):
        """
        Khi chuyển mode, prediction_history và last_confirmed_sign phải được reset sạch sẽ,
        ngăn chặn hoàn toàn việc trạng thái debounce của chữ cái cũ ảnh hưởng đến số mới.
        """
        lm_letter = self.samples_by_digit["1"][0]
        # Xác nhận chữ D trong mode letters
        self.rec.smoothing_window = 2
        for _ in range(3):
            res = self.rec.predict_landmarks(lm_letter, mode="letters", session_id="stale_test")
        self.assertEqual(self.rec.last_confirmed_sign, "D")
        self.assertGreater(len(self.rec.prediction_history), 0)

        # Đổi mode sang digits
        lm_digit = self.samples_by_digit["3"][0]
        res_d = self.rec.predict_landmarks(lm_digit, mode="digits", session_id="stale_test")
        # last_confirmed_sign đã được reset về None trước khi xử lý frame mới
        self.assertNotEqual(self.rec.last_confirmed_sign, "D")
        self.assertEqual(len(self.rec.prediction_history), 1)
        self.assertEqual(self.rec.prediction_history[0][0], "3")

    # -------------------------------------------------------------------------
    # 15. Kiểm tra tính độc lập tuyệt đối giữa các mode không để lộ nhãn chéo
    # -------------------------------------------------------------------------
    def test_15_strict_no_cross_mode_labels_during_rapid_switch(self):
        """
        Chuyển đổi liên tục giữa letters và digits:
        - Mode digits không bao giờ sinh ra nhãn nằm trong ALLOWED_LETTERS.
        - Mode letters không bao giờ sinh ra nhãn nằm trong ALLOWED_DIGITS.
        """
        lm = self.samples_by_digit["1"][0]
        for cycle in range(5):
            res_l = self.rec.predict_landmarks(lm, mode="letters", session_id=f"rapid_{cycle}")
            self.assertIn(res_l["sign"], ALLOWED_LETTERS | {"NEUTRAL"})
            self.assertNotIn(res_l["sign"], ALLOWED_DIGITS)

            res_d = self.rec.predict_landmarks(lm, mode="digits", session_id=f"rapid_{cycle}")
            self.assertIn(res_d["sign"], ALLOWED_DIGITS | {"NEUTRAL"})
            self.assertNotIn(res_d["sign"], ALLOWED_LETTERS)

    # -------------------------------------------------------------------------
    # 16. Chuyển đổi qua lại giữa số 0 và chữ O / C
    # -------------------------------------------------------------------------
    def test_16_zero_digit_vs_letter_mode_switching(self):
        """
        Kiểm tra số 0: Trong mode digits trả về '0', khi chuyển sang mode letters không bao giờ trả về '0'.
        """
        fixture_0_path = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_0_landmarks.json")
        if os.path.exists(fixture_0_path):
            with open(fixture_0_path, "r", encoding="utf-8") as f:
                d0_data = json.load(f)
            lm_0 = d0_data["samples"][0]["landmarks"]
            # Mode digits
            res_0 = self.rec.predict_landmarks(lm_0, mode="digits", session_id="zero_switch")
            self.assertEqual(res_0["sign"], "0")
            self.assertEqual(res_0["type"], "digit")

            # Chuyển trực tiếp sang mode letters
            res_l = self.rec.predict_landmarks(lm_0, mode="letters", session_id="zero_switch")
            self.assertNotEqual(res_l["sign"], "0")
            self.assertIn(res_l["sign"], ALLOWED_LETTERS | {"NEUTRAL"})


if __name__ == "__main__":
    unittest.main()
