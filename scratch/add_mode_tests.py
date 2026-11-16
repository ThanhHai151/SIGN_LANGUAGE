import os

test_code = '''
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
'''

with open("backend/tests/test_mode_separation.py", "r", encoding="utf-8") as f:
    content = f.read()

target = 'if __name__ == "__main__":\n    unittest.main()'
assert target in content, "Could not find target"

new_content = content.replace(target, test_code + '\n\n' + target)

with open("backend/tests/test_mode_separation.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("test_mode_separation.py updated successfully!")

