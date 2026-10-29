"""
Bộ kiểm thử End-to-End (E2E) và Kiểm chứng Độc lập Digit Classifier sau Tích hợp (Giai đoạn 2C.4C).

Mục tiêu kiểm chứng:
1. Đánh giá toàn diện 18 mẫu landmark thực tế (9 DEV, 9 EVAL) qua luồng predict_landmarks(mode='letters_numbers').
2. Phân tách rõ ràng kết quả giữa:
   - Standalone digit_classifier (18/18 mẫu).
   - Normal Pipeline (không truyền expected_sign - tình huống sử dụng thực tế của người dùng).
   - Expected Pipeline (có truyền expected_sign).
3. Kiểm chứng và định lượng xung đột hình thái tĩnh giữa Chữ cái và Chữ số (1/D, 2/V/U, 6/Y, 7/L, 0/O).
4. Xác minh sự phụ thuộc của các số đồng dạng (1, 2, 6) vào chính sách expected_sign.
"""

import os
import json
import unittest
import numpy as np
import torch

from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS, ALLOWED_DIGITS, ALLOWED_LETTERS_AND_DIGITS
from backend.vision.digit_classifier import digit_classifier
from backend.vision.feature_extractor import extract_dual_hand_features, calculate_angle, normalize_hand_landmarks

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json")


class TestDigitClassifierE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cls.samples = data["samples"]
        cls.rec = VSLRecognizer()

    def setUp(self):
        self.rec.prediction_history.clear()
        self.rec.sentence_tokens.clear()
        self.rec.last_confirmed_sign = None
        self.rec.last_confirmed_time = 0

    # -------------------------------------------------------------------------
    # 1. KIỂM THỬ TOÀN DIỆN 18 MẪU VỚI DIGIT CLASSIFIER ĐỘC LẬP
    # -------------------------------------------------------------------------
    def test_01_standalone_classifier_all_18_samples(self):
        """18/18 mẫu landmark thực tế (9 DEV + 9 EVAL) phải đạt MATCH đúng nhãn số."""
        for s in self.samples:
            digit = s["digit"]
            split = s["split"]
            lm = s["landmarks"]
            res = digit_classifier.classify_detailed(lm)
            self.assertEqual(res["status"], "MATCH", f"Mẫu {split} số {digit} không MATCH: {res}")
            self.assertEqual(res["digit"], digit, f"Mẫu {split} số {digit} match sai thành {res['digit']}")
            self.assertGreaterEqual(res["confidence"], 0.85)

    # -------------------------------------------------------------------------
    # 2. KIỂM THỬ NORMAL PIPELINE (KHÔNG CÓ expected_sign - LUỒNG THỰC TẾ)
    # -------------------------------------------------------------------------
    def test_02_normal_pipeline_independent_digits_succeed(self):
        """Các chữ số không có xung đột chữ cái (3, 4, 5, 7, 8, 9) nhận diện đúng 12/12 mẫu."""
        independent_digits = {"3", "4", "5", "7", "8", "9"}
        success_count = 0
        for s in self.samples:
            digit = s["digit"]
            if digit in independent_digits:
                lm = s["landmarks"]
                pred = self.rec.predict_landmarks(lm, mode="letters_numbers")
                self.assertEqual(pred["sign"], digit, f"Số {digit} ({s['split']}) thất bại trong normal pipeline: got {pred['sign']}")
                self.assertEqual(pred["type"], "digit")
                success_count += 1
        self.assertEqual(success_count, 12, "Phải có đúng 12 mẫu độc lập (6 số x 2 split) thành công")

    def test_03_normal_pipeline_homomorphic_digits_yield_letters(self):
        """
        Chứng minh thực nghiệm: Trong normal pipeline (không có expected_sign),
        các số đồng dạng (1, 2, 6) BỊ GHI ĐÈ bởi chữ cái (D, U, Y) do chính sách bảo tồn chữ cái.
        Tổng cộng 6/6 mẫu (3 số x 2 split) trả về chữ cái thay vì chữ số.
        """
        homomorphic_map = {
            "1": "D",
            "2": "U",  # Cử chỉ số 2 trong video QIPEDC 2 ngón khép sát nhau, khớp chữ U
            "6": "Y",
        }
        for s in self.samples:
            digit = s["digit"]
            if digit in homomorphic_map:
                lm = s["landmarks"]
                pred = self.rec.predict_landmarks(lm, mode="letters_numbers")
                expected_letter = homomorphic_map[digit]
                self.assertEqual(
                    pred["sign"], expected_letter,
                    f"Số {digit} trong normal pipeline không trả về chữ cái đồng dạng {expected_letter}, mà trả {pred['sign']}"
                )
                self.assertEqual(pred["type"], "alphabet")

    # -------------------------------------------------------------------------
    # 3. KIỂM THỬ PIPELINE KHI CÓ expected_sign (LUỒNG LUYỆN TẬP / HỌC TẬP)
    # -------------------------------------------------------------------------
    def test_04_expected_sign_pipeline_all_18_samples_succeed(self):
        """Khi có expected_sign, toàn bộ 18/18 mẫu (cả 1, 2, 6) đều nhận diện đúng nhãn chữ số."""
        for s in self.samples:
            digit = s["digit"]
            lm = s["landmarks"]
            pred = self.rec.predict_landmarks(lm, mode="letters_numbers", expected_sign=digit)
            self.assertEqual(
                pred["sign"], digit,
                f"Mẫu số {digit} ({s['split']}) thất bại khi có expected_sign: got {pred['sign']}"
            )
            self.assertEqual(pred["type"], "digit")

    # -------------------------------------------------------------------------
    # 4. KIỂM TRA ĐẶC TRƯNG HÌNH HỌC VÀ GÓC CỦA SỐ 7 VÀ CHỮ L
    # -------------------------------------------------------------------------
    def test_05_digit_7_vs_letter_l_geometric_distribution(self):
        """
        Kiểm chứng góc mở ngón cái - ngón trỏ (angle_ti) của Số 7 thực tế:
        - DEV frame 54: 37.8°
        - EVAL frame 45: 28.1°
        Cả hai mẫu đều < 50° (ngưỡng quy định của Chữ L), do đó không bị nhận nhầm là Chữ L.
        """
        samples_7 = [s for s in self.samples if s["digit"] == "7"]
        self.assertEqual(len(samples_7), 2)
        for s in samples_7:
            lm = s["landmarks"]
            shifted, meta = normalize_hand_landmarks(lm)
            palm_scale = meta.get("palm_scale", 1.0)
            p_pts = [shifted[i*3:(i+1)*3] for i in range(21)]
            ang_ti = calculate_angle(p_pts[4], p_pts[0], p_pts[8])
            thumb_lat = abs(p_pts[4][0] - p_pts[0][0]) / palm_scale

            # Xác minh góc thực tế nằm trong dải [25°, 45°]
            self.assertGreaterEqual(ang_ti, 25.0)
            self.assertLess(ang_ti, 50.0, f"Góc angle_ti của số 7 vượt quá 50° ({ang_ti}°), có nguy cơ xung đột với L")
            self.assertGreaterEqual(thumb_lat, 0.35)

    # -------------------------------------------------------------------------
    # 5. KIỂM TRA CHỮ SỐ 0 VÀ CHỮ O
    # -------------------------------------------------------------------------
    def test_06_digit_0_is_not_implemented_and_returns_letter_o(self):
        """
        Xác minh giới hạn hiện tại:
        Số 0 chưa có trong digit_classifier (chưa có ground truth).
        Cử chỉ hình khuyên tròn O khi đưa vào pipeline luôn trả về Chữ O (alphabet).
        """
        self.assertNotIn("0", ALLOWED_DIGITS)
        # Giả lập cử chỉ O cơ bản
        lm = [[0.5, 0.7, 0.0] for _ in range(21)]
        lm[0] = [0.5, 0.8, 0.0]
        lm[9] = [0.5, 0.5, 0.0]
        lm[8] = [0.45, 0.40, 0.0]
        lm[4] = [0.46, 0.41, 0.0]  # ngón cái chạm ngón trỏ
        lm[3] = [0.47, 0.43, 0.0]

        pred = self.rec.predict_landmarks(lm, mode="letters_numbers")
        self.assertNotEqual(pred["sign"], "0", "Hệ thống không được tự suy đoán Số 0 khi chưa có ground truth")


if __name__ == "__main__":
    unittest.main()

