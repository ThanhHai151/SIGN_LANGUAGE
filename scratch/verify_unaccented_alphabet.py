import os, unittest
from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS

class TestUnaccentedAlphabetOnly(unittest.TestCase):
    def setUp(self):
        self.rec = VSLRecognizer(confidence_threshold=0.65)

    def test_allowed_letters_exact_26_latin(self):
        """Kiểm tra ALLOWED_LETTERS chỉ chứa 26 chữ cái không dấu A-Z."""
        expected = {chr(c) for c in range(ord("A"), ord("Z") + 1)}
        self.assertEqual(ALLOWED_LETTERS, expected)
        self.assertEqual(len(ALLOWED_LETTERS), 26)

    def test_no_accented_letters_in_allowed(self):
        """Tuyệt đối không có bất kỳ chữ cái có dấu nào trong ALLOWED_LETTERS."""
        accented = ["Ă", "Â", "Ê", "Ô", "Ơ", "Ư", "Đ", "Á", "À", "Ả", "Ã", "Ạ", "É", "È", "Ẻ", "Ẽ", "Ẹ", "Í", "Ì", "Ỉ", "Ĩ", "Ị", "Ó", "Ò", "Ỏ", "Õ", "Ọ", "Ú", "Ù", "Ủ", "Ũ", "Ụ", "Ý", "Ỳ", "Ỷ", "Ỹ", "Ỵ"]
        for ch in accented:
            self.assertNotIn(ch, ALLOWED_LETTERS, f"Chữ có dấu '{ch}' không được phép có trong ALLOWED_LETTERS!")

    def test_letters_mode_never_returns_accented_letters(self):
        """Mô phỏng các tư thế tay khác nhau trong mode='letters', kết quả trả về không bao giờ là chữ có dấu."""
        accented_set = {"Ă", "Â", "Ê", "Ô", "Ơ", "Ư", "Đ"}
        
        # Test với nhiều bộ landmarks giả lập khác nhau
        dummy_landmarks_list = [
            [[0.5, 0.5, 0.0]] * 21,
            [[0.5, 0.5 + i*0.01, 0.0] for i in range(21)],
            [[0.5 + (i % 5)*0.05, 0.5 - (i // 5)*0.08, 0.0] for i in range(21)],
        ]
        
        for idx, lm in enumerate(dummy_landmarks_list):
            res = self.rec.predict_landmarks(lm, mode="letters", session_id=f"test_accent_{idx}")
            self.assertNotIn(res["sign"], accented_set, f"res['sign'] trả về chữ có dấu '{res['sign']}'!")
            if res.get("label"):
                for acc in accented_set:
                    self.assertNotEqual(res["label"], acc, f"res['label'] trả về chữ có dấu '{acc}'!")
            self.assertTrue(res["sign"] in ALLOWED_LETTERS or res["sign"] == "NEUTRAL")

    def test_diacritic_fsm_bypassed_in_letters_mode(self):
        """Đảm bảo FSM dấu không can thiệp biến đổi A, E, O, U, D thành chữ có dấu trong mode letters."""
        # Giả lập khi classifier trả về A
        lm_a = [[0.5, 0.5, 0.0]] * 21
        # Cho dù classifier hoặc rules phán đoán gì, final_sign trong mode letters không bao giờ là Ă, Â, Ê, Ô, Ơ, Ư, Đ
        res = self.rec.predict_landmarks(lm_a, mode="letters")
        self.assertNotIn(res["sign"], {"Ă", "Â", "Ê", "Ô", "Ơ", "Ư", "Đ"})

if __name__ == "__main__":
    unittest.main()

